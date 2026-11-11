"""全重みを更新するQwen3と、回答位置だけを投影する省メモリloss。"""

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str, sid_tokens: list[str], seed: int = 42):
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    original_token_count = len(tokenizer)
    added_count = tokenizer.add_tokens(sid_tokens, special_tokens=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    )
    model.resize_token_embeddings(
        max(model.config.vocab_size, len(tokenizer)), mean_resizing=False
    )
    model.tie_weights()
    # tokenizer未使用のpadding行も含め、全SIDを同じ分布で初期化する。
    # checkpointの再読込時には学習済みSIDを初期化し直さない。
    if added_count:
        with torch.no_grad():
            ids = tokenizer.convert_tokens_to_ids(sid_tokens)
            values = torch.empty(len(ids), model.config.hidden_size)
            values.normal_(mean=0.0, std=model.config.initializer_range)
            model.get_input_embeddings().weight[ids] = values
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError("固定されたパラメータがあります")
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise ValueError("入力と出力embeddingの共有が失われています")
    for token in sid_tokens:
        if tokenizer.encode(token, add_special_tokens=False) != [
            tokenizer.convert_tokens_to_ids(token)
        ]:
            raise ValueError(f"SIDが単一トークンになりません: {token}")
    return model, tokenizer, original_token_count


def parameter_digest(model) -> str:
    """初期パラメータ全体の一致を確認するハッシュ。"""
    import hashlib

    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().float().numpy().tobytes())
    return digest.hexdigest()


def configure_lora(model, rank=16, alpha=32):
    """TransformerはLoRAで適応し、既存語彙を含むembedding全体も更新する。

    adapter保存時にはsave_embedding_layers=Trueで全embeddingを必ず保存する。
    adapter再読込後の追加学習では、共有embeddingのrequires_gradも再度有効にする。
    """
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )
    adapted = get_peft_model(model, config)
    core = adapted.get_base_model()
    core.get_input_embeddings().weight.requires_grad_(True)
    if core.get_input_embeddings().weight is not core.get_output_embeddings().weight:
        raise ValueError("LoRA適用後にembeddingとlm_headの共有が失われています")
    return adapted


def response_losses(model, batch: dict) -> torch.Tensor:
    """通常のcausal CEと同値の位置を計算し、回答長で正規化した各例のlossを返す。

    大半がloss対象外のpromptなので、lm_headは教師のある位置にだけ適用する。
    各例を同じ重みにすることで長いメタデータ回答によるタスク重みの偏りを避ける。
    """
    core = model.get_base_model() if hasattr(model, "peft_config") else model
    hidden = core.model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    ).last_hidden_state
    targets = batch["labels"][:, 1:]
    positions = targets != -100
    if not positions.any() or not positions.any(dim=1).all():
        raise ValueError("教師トークンのない学習例があります")
    logits = core.lm_head(hidden[:, :-1][positions])
    token_losses = F.cross_entropy(logits.float(), targets[positions], reduction="none")
    example_ids = torch.where(positions)[0]
    sums = torch.zeros(targets.shape[0], device=hidden.device)
    sums.scatter_add_(0, example_ids, token_losses)
    return sums / positions.sum(dim=1)

"""全カタログの有効SIDを制約付きbeam searchで生成し、item単位で評価する。"""

import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

from src.experiments.common import write_json
from src.experiments.qwen_sid.data import read_examples, sid_text


def build_sid_trie(tokenizer, sids):
    allowed = {}
    item_by_tokens = {}
    for item, codes in enumerate(sids):
        tokens = tuple(tokenizer.encode(sid_text(codes), add_special_tokens=False))
        if len(tokens) != 4 or tokens in item_by_tokens:
            raise ValueError("SIDのトークン数または一意性が不正です")
        item_by_tokens[tokens] = item
        for length in range(4):
            allowed.setdefault(tokens[:length], set()).add(tokens[length])
    return {key: sorted(value) for key, value in allowed.items()}, item_by_tokens


def ranking_metrics(predictions: list[list[int]], targets: list[int]) -> dict:
    if not predictions or len(predictions) != len(targets):
        raise ValueError("予測件数と正解件数が一致しません")
    result = {
        name: 0.0 for name in ("hit@1", "recall@5", "recall@10", "ndcg@5", "ndcg@10")
    }
    for candidates, target in zip(predictions, targets, strict=True):
        if len(candidates) != len(set(candidates)):
            raise ValueError("候補商品が重複しています")
        rank = candidates.index(target) + 1 if target in candidates else math.inf
        result["hit@1"] += rank == 1
        for k in (5, 10):
            result[f"recall@{k}"] += rank <= k
            result[f"ndcg@{k}"] += 1 / math.log2(rank + 1) if rank <= k else 0
    return {key: value / len(targets) for key, value in result.items()}


class SidPrefixProcessor(LogitsProcessor):
    """prefix制約をGPU上で一括適用し、beamごとのCPU転送を避ける。"""

    def __init__(
        self, trie, base_id, width, prompt_length, device, check_valid_prefix=True
    ):
        self.base_id, self.width = base_id, width
        self.prompt_length = prompt_length
        self.check_valid_prefix = check_valid_prefix
        self.keys, self.allowed = [], []
        for depth in range(4):
            groups = []
            for prefix, tokens in trie.items():
                if len(prefix) != depth:
                    continue
                key = 0
                for level, token in enumerate(prefix):
                    key = key * width + token - base_id - level * width
                groups.append((key, tokens))
            groups.sort()
            keys = torch.tensor([key for key, _ in groups], device=device)
            mask = torch.zeros(len(groups), width, dtype=torch.bool)
            for row, (_, tokens) in enumerate(groups):
                mask[row, [t - base_id - depth * width for t in tokens]] = True
            self.keys.append(keys)
            self.allowed.append(mask.to(device))

    def __call__(self, input_ids, scores):
        depth = input_ids.shape[1] - self.prompt_length
        start = self.base_id + depth * self.width
        sid_scores = scores[:, start : start + self.width].clone()
        if self.check_valid_prefix:
            key = input_ids.new_zeros(input_ids.shape[0])
            for level in range(depth):
                key = (
                    key * self.width
                    + input_ids[:, self.prompt_length + level]
                    - self.base_id
                    - level * self.width
                )
            nodes = torch.searchsorted(self.keys[depth], key)
            sid_scores.masked_fill_(~self.allowed[depth][nodes], -torch.inf)
        scores.fill_(-torch.inf)
        scores[:, start : start + self.width] = sid_scores
        return scores


@torch.inference_mode()
def evaluate_recommendations(model, tokenizer, examples, sids, config, output: Path):
    model.eval()
    trie, item_by_tokens = build_sid_trie(tokenizer, sids)
    layer_tokens = [
        [
            tokenizer.convert_tokens_to_ids(f"<sid_{level}_{code}>")
            for code in range(config["sid_codebook_width"])
        ]
        for level in range(4)
    ]
    code_by_token = {
        token: code for layer in layer_tokens for code, token in enumerate(layer)
    }
    predictions, targets = [], []
    rows = [row for row in examples if row["task"] == "recommendation"]
    processor = SidPrefixProcessor(
        trie,
        layer_tokens[0][0],
        config["sid_codebook_width"],
        0,
        model.device,
        config["check_valid_prefix"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.padding_side = "left"
    with output.open("w") as stream:
        for start in range(0, len(rows), config["eval_batch_size"]):
            selected = rows[start : start + config["eval_batch_size"]]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for row in selected
            ]
            batch = tokenizer(prompts, padding=True, return_tensors="pt").to(
                model.device
            )
            prompt_length = batch.input_ids.shape[1]

            processor.prompt_length = prompt_length

            with torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.generate(
                    **batch,
                    do_sample=False,
                    num_beams=config["beam_width"],
                    num_return_sequences=config["beam_width"],
                    max_new_tokens=4,
                    min_new_tokens=4,
                    eos_token_id=None,
                    forced_eos_token_id=None,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                    length_penalty=0.0,
                    renormalize_logits=True,
                    logits_processor=[processor],
                    top_k=None,
                    top_p=None,
                    temperature=None,
                )
            generated = generated[:, prompt_length:].reshape(len(selected), -1, 4)
            for row, beams in zip(selected, generated.cpu().tolist(), strict=True):
                # 元TIGERと同じ非制約評価では無効SIDも候補枠を消費する。
                # 異なる無効SIDに異なる負のIDを付け、正解商品には一致させない。
                candidates = []
                for tokens in beams:
                    key = tuple(tokens)
                    invalid_code = sum(
                        code_by_token[t] * 256 ** (3 - i) for i, t in enumerate(tokens)
                    )
                    candidates.append(item_by_tokens.get(key, -1 - invalid_code))
                # 同じ商品が出た場合は隠して続行せず評価を失敗させる。
                if len(set(candidates)) != len(candidates):
                    raise ValueError("beam内の商品重複")
                predictions.append(candidates)
                targets.append(row["item_id"])
                stream.write(
                    json.dumps(
                        {
                            "user_id": row["user_id"],
                            "target": row["item_id"],
                            "predictions": candidates,
                        }
                    )
                    + "\n"
                )
            if start % (20 * config["eval_batch_size"]) == 0:
                print(f"evaluation {len(targets)}/{len(rows)}", flush=True)
    tokenizer.padding_side = "right"
    return {
        "examples": len(rows),
        "catalog_items": len(sids),
        "beam_width": config["beam_width"],
        "check_valid_prefix": config["check_valid_prefix"],
        "invalid_candidate_fraction": sum(
            item < 0 for row in predictions for item in row
        )
        / sum(map(len, predictions)),
        **ranking_metrics(predictions, targets),
    }


def evaluate_checkpoint(config: dict) -> dict:
    out = Path(config["output_dir"])
    checkpoint = out / "best"
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.float32, attn_implementation="sdpa"
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    sids = json.loads((Path(config["prepared_dir"]) / "sids.json").read_text())
    results = {}
    for split in ("evaluation", "testing"):
        examples = read_examples(Path(config["prepared_dir"]) / f"{split}.jsonl")
        results[split] = evaluate_recommendations(
            model, tokenizer, examples, sids, config, out / f"{split}_predictions.jsonl"
        )
        write_json(out / f"{split}_metrics.json", results[split])
    write_json(out / "results.json", results)
    return results

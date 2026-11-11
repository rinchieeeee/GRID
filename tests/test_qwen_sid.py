"""元GRIDとのデータ対応、回答loss、学習と再開の挙動を検証する。"""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

import src.utils  # noqa: F401  元GRIDの循環importを避けるためutilsを先に初期化する。
from scripts import run_qwen_sid_ablation
from src.data.loading.components.collate_functions import (
    collate_with_sid_causal_duplicate,
)
from src.data.loading.components.label_function import NextKTokenMasking
from src.experiments.common import read_json, write_json
from src.experiments.qwen_sid import training
from src.experiments.qwen_sid.data import (
    alignment_examples,
    collate_examples,
    prepare_recommendation_plan,
    sample_grid_subsequences,
    training_sequences,
    validate_histories,
)
from src.experiments.qwen_sid.evaluation import SidPrefixProcessor, ranking_metrics
from src.experiments.qwen_sid.model import (
    configure_lora,
    parameter_digest,
    response_losses,
)
from src.experiments.qwen_sid.optimization import auxiliary_groups


@pytest.mark.parametrize("cap", [5, 512, 10000])
def test_subsequences_match_original_grid_collate(cap):
    # 30商品を超える例も含め、trimと教師の境界まで元実装と比較する。
    rows = [
        {"user_id": 0, "items": list(range(35))},
        {"user_id": 1, "items": list(range(40, 46))},
    ]
    codes = {i: [i, i + 50, i + 100, i + 150] for i in range(46)}
    batch = [
        {
            "sequence_data": torch.tensor(
                [code for item in row["items"] for code in codes[item]]
            )
        }
        for row in rows
    ]
    torch.manual_seed(19)
    inputs, labels = collate_with_sid_causal_duplicate(
        batch,
        sequence_field_name="sequence_data",
        sid_hierarchy=4,
        labels={"sequence_data": SimpleNamespace(transform=NextKTokenMasking(4))},
        sequence_length=120,
        padding_token=-1,
        masking_token=-1,
        max_batch_size=cap,
    )
    actual_targets = labels.labels["sequence_data"].reshape(-1, 4)
    sampled = sample_grid_subsequences(rows, cap, torch.Generator().manual_seed(19))
    assert len(sampled) == len(actual_targets)
    for i, row in enumerate(sampled):
        sequence = row["items"][-30:]
        expected = [code for item in sequence[:-1] for code in codes[item]]
        actual = inputs.transformed_sequences["sequence_data"][i]
        assert actual[actual != -1].tolist() == expected
        assert actual_targets[i].tolist() == codes[sequence[-1]]


def test_recommendation_order_is_independent_of_auxiliary_rng():
    histories = [{"user_id": i, "items": [0, 1, 2, i + 3]} for i in range(33)]
    config = {
        "seed": 42,
        "source_history_batch_size": 32,
        "augmentation_cap": 512,
        "history_items": 29,
    }
    first = training_sequences(histories, config, 0)
    auxiliary_groups({"text_to_sid": [1, 2], "sid_to_text": [3, 4]}, 5, 42, 1, 1)
    assert first == training_sequences(histories, config, 0)
    assert {row["user_id"] for row in first} == set(range(33))


def test_auxiliary_weight_and_sampling():
    data = {"text_to_sid": [1, 2], "sid_to_text": [3, 4]}
    assert auxiliary_groups(data, 4, 42, 0, 0) == []
    groups = auxiliary_groups(data, 4, 42, 0, 1)
    assert groups == auxiliary_groups(data, 4, 42, 0, 1)
    assert [weight for _, _, weight in groups] == [0.5, 0.5]
    assert all(len(rows) == 4 for _, rows, _ in groups)


def tiny_model():
    torch.manual_seed(42)
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            tie_word_embeddings=True,
            attention_dropout=0.0,
            use_cache=False,
        )
    )


@pytest.mark.parametrize(
    ("max_steps", "patience", "expected_steps", "stop_reason"),
    [
        (9, 3, [2, 4, 6, 8, 9], "max_steps"),
        (13, 2, [2, 4, 6, 8, 10], "early_stopping"),
    ],
)
def test_step_validation_and_resume(
    tmp_path, monkeypatch, max_steps, patience, expected_steps, stop_reason
):
    prepared = tmp_path / "data"
    base = tmp_path / "base"
    write_json(prepared / "manifest.json", {"sid_tokens": [], "prepared_sha256": {}})
    write_json(
        prepared / "training_histories.json",
        [{"user_id": 0, "items": [0, 1, 2, 3]}, {"user_id": 1, "items": [2, 3]}],
    )
    write_json(prepared / "sids.json", [[i] * 4 for i in range(4)])
    for name in ("training.jsonl", "evaluation.jsonl"):
        (prepared / name).touch()
    revision = base / ".cache/huggingface/download/config.json.metadata"
    revision.parent.mkdir(parents=True)
    revision.write_text("test\n")
    config = {
        "prepared_dir": str(prepared),
        "output_dir": str(tmp_path / "resumed"),
        "model_path": str(base),
        "max_steps": max_steps,
        "eval_steps": 2,
        "early_stopping_patience": patience,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 2,
        "seed": 42,
        "history_items": 29,
        "source_history_batch_size": 2,
        "augmentation_cap": 512,
        "max_length": 32,
        "max_grad_norm": 1.0,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "warmup_ratio": 0.3,
        "alignment_weight": 0.0,
    }
    prepare_recommendation_plan(config)

    def load_model(path, sid_tokens, seed):
        model = (
            tiny_model()
            if path == str(base)
            else Qwen3ForCausalLM.from_pretrained(path)
        )
        model.to = lambda device: model
        return model, SimpleNamespace(save_pretrained=lambda path: None), 32

    def backward_groups(model, tokenizer, groups, config):
        assert model.training
        target = sum(row["item_id"] for row in groups[0][1])
        loss = (model.lm_head.weight[0, 0] - target).square()
        loss.backward()
        return {"recommendation": loss.item()}

    def evaluate(model, tokenizer, examples, sids, config, output):
        model.eval()
        output.write_text("[]\n")
        return {"recall@5": next(scores)}

    monkeypatch.setattr(training, "load_model", load_model)
    monkeypatch.setattr(training, "encode_example", lambda tokenizer, row, length: row)
    monkeypatch.setattr(training, "backward_groups", backward_groups)
    monkeypatch.setattr(training, "evaluate_recommendations", evaluate)
    monkeypatch.setattr(torch.cuda, "get_rng_state", torch.get_rng_state)
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state: None)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)

    scores = iter([0.2, 0.2, 0.3, 0.3, 0.3])
    paused = training.train(config, until_step=5)
    assert (paused["step"], paused["epoch_index"], paused["next_example"]) == (5, 1, 2)
    assert paused["evaluations_without_improvement"] == 1
    if stop_reason == "early_stopping":
        paused = training.train(config, resume=True, until_step=9)
        assert paused["best_step"] == 6
        assert paused["evaluations_without_improvement"] == 1

        def fail_save(*args, **kwargs):
            raise OSError("checkpoint書き込み失敗")

        out = tmp_path / "resumed"
        history_before = (out / "metrics.jsonl").read_text()
        with monkeypatch.context() as patch:
            patch.setattr(training.torch, "save", fail_save)
            with pytest.raises(OSError, match="checkpoint書き込み失敗"):
                training.train(config, resume=True)
        assert read_json(out / "last" / "progress.json") == paused
        assert (out / "metrics.jsonl").read_text() == history_before
        scores = iter([0.3])
    resumed = training.train(config, resume=True)
    out = tmp_path / "resumed"
    history = [
        json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in history] == expected_steps
    assert [row["evaluations_without_improvement"] for row in history] == [
        0,
        1,
        0,
        1,
        2,
    ]
    assert [row["recommendation_examples"] for row in history] == [
        4,
        3,
        4,
        3,
        2 if max_steps == 9 else 4,
    ]
    assert resumed["stop_reason"] == stop_reason
    assert resumed["step"] == expected_steps[-1]
    assert read_json(out / "best" / "selection.json")["step"] == 6

    scores = iter([0.2, 0.2, 0.3, 0.3, 0.3])
    training.train({**config, "output_dir": str(tmp_path / "continuous")})
    for checkpoint in ("best", "last"):
        resumed_model = Qwen3ForCausalLM.from_pretrained(out / checkpoint)
        continuous_model = Qwen3ForCausalLM.from_pretrained(
            tmp_path / "continuous" / checkpoint
        )
        assert parameter_digest(resumed_model) == parameter_digest(continuous_model)


def test_ablation_stops_both_conditions_at_the_same_step(tmp_path, monkeypatch):
    config = {"max_steps": 9, "eval_steps": 2, "early_stopping_patience": 1}
    variants = {"A": tmp_path / "A.json", "B": tmp_path / "B.json"}
    calls = []

    def run_stage(root, stage, arguments):
        name = stage[0]
        step = int(arguments[arguments.index("--until-step") + 1])
        assert "--defer-early-stopping" in arguments
        calls.append((name, step))
        write_json(
            root / name / "experiment.json",
            {
                "initial_parameter_sha256": "initial",
                "recommendation_plan_sha256": "plan",
            },
        )
        write_json(
            root / name / "last" / "progress.json",
            {
                "step": step,
                "evaluations_without_improvement": int(
                    step >= (4 if name == "A" else 6)
                ),
            },
        )

    monkeypatch.setattr(run_qwen_sid_ablation, "run_stage", run_stage)
    assert run_qwen_sid_ablation.train_variants(tmp_path, variants, config) == (
        6,
        "early_stopping",
    )
    assert calls == [(name, step) for step in (2, 4, 6) for name in variants]
    calls.clear()
    assert run_qwen_sid_ablation.train_variants(tmp_path, variants, config) == (
        6,
        "early_stopping",
    )
    assert calls == []


def test_sparse_response_loss_and_gradients_match_full_logits():
    model = tiny_model()
    reference = deepcopy(model)
    rows = [
        {"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]},
        {"input_ids": [1, 5, 6], "labels": [-100, -100, 6]},
    ]
    batch = collate_examples(rows, pad_id=0)
    actual = response_losses(model, batch)
    logits = reference(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    ).logits[:, :-1]
    targets = batch["labels"][:, 1:]
    tokens = F.cross_entropy(
        logits.reshape(-1, 32), targets.reshape(-1), ignore_index=-100, reduction="none"
    ).reshape_as(targets)
    expected = tokens.sum(1) / (targets != -100).sum(1)
    torch.testing.assert_close(actual, expected)
    actual.mean().backward()
    expected.mean().backward()
    for a, b in zip(model.parameters(), reference.parameters(), strict=True):
        assert a.grad is not None
        torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-4)


def test_full_update_and_checkpoint_reload_preserve_predictions(tmp_path):
    model = tiny_model()
    before = parameter_digest(model)
    batch = collate_examples(
        [{"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]}], pad_id=0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    response_losses(model, batch).mean().backward()
    optimizer.step()
    assert parameter_digest(model) != before
    model.save_pretrained(tmp_path)
    restored = Qwen3ForCausalLM.from_pretrained(tmp_path)
    assert parameter_digest(restored) == parameter_digest(model)
    assert restored.lm_head.weight is restored.model.embed_tokens.weight
    model.eval()
    restored.eval()
    torch.testing.assert_close(
        model(input_ids=batch["input_ids"]).logits,
        restored(input_ids=batch["input_ids"]).logits,
    )


def test_metadata_is_preserved_verbatim():
    item = {
        "item_id": 0,
        "text": "Title: abc; Brand: xyz; Categories: ['A']; Price: 5; ",
    }
    rows = alignment_examples(item, [0, 1, 2, 3])
    assert rows[0]["prompt"].endswith(item["text"])
    assert rows[1]["response"] == item["text"]


def test_history_validation_accepts_only_expected_truncation():
    splits = {
        "training": [{"user_id": 0, "items": list(range(25))}],
        "evaluation": [{"user_id": 0, "items": list(range(5, 26))}],
        "testing": [{"user_id": 0, "items": list(range(6, 27))}],
    }
    assert validate_histories(splits, 30)["training"] == 1
    splits["testing"][0]["items"][0] = 0
    with pytest.raises(ValueError, match="prefix/suffix"):
        validate_histories(splits, 30)


def test_ranking_metrics_use_item_rank_and_reject_duplicates():
    metrics = ranking_metrics([[2, 0, 1], [0, 1, 2]], [0, 3])
    assert metrics["recall@5"] == 0.5
    assert metrics["hit@1"] == 0
    assert metrics["ndcg@5"] == pytest.approx(
        0.5 / torch.log2(torch.tensor(3.0)).item()
    )
    with pytest.raises(ValueError, match="重複"):
        ranking_metrics([[1, 1]], [1])


def test_vectorized_prefix_mask_matches_trie():
    base, width = 10, 4
    codes = [[0, 1, 2, 0], [0, 2, 1, 1], [1, 0, 3, 0]]
    trie = {}
    for row in codes:
        tokens = tuple(base + level * width + code for level, code in enumerate(row))
        for depth in range(4):
            trie.setdefault(tokens[:depth], set()).add(tokens[depth])
    processor = SidPrefixProcessor(trie, base, width, 2, "cpu")
    for prefix, allowed in trie.items():
        inputs = torch.tensor([[1, 2, *prefix]])
        scores = torch.randn(1, 30)
        expected = torch.full_like(scores, -torch.inf)
        expected[:, list(allowed)] = scores[:, list(allowed)]
        torch.testing.assert_close(processor(inputs, scores.clone()), expected)


def test_lora_updates_existing_and_sid_embeddings_and_reloads(tmp_path):
    from peft import PeftModel

    model = configure_lora(tiny_model(), rank=2, alpha=4)
    core = model.get_base_model()
    frozen = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if not p.requires_grad
    }
    old_before = core.get_input_embeddings()(torch.tensor([1])).detach().clone()
    sid_before = core.get_input_embeddings()(torch.tensor([30])).detach().clone()
    batch = collate_examples(
        [{"input_ids": [1, 30, 2, 31], "labels": [-100, -100, 2, 31]}], pad_id=0
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    response_losses(model, batch).mean().backward()
    optimizer.step()
    for name, p in model.named_parameters():
        if name in frozen:
            assert torch.equal(p, frozen[name])
    assert not torch.equal(core.get_input_embeddings()(torch.tensor([1])), old_before)
    assert not torch.equal(core.get_input_embeddings()(torch.tensor([30])), sid_before)
    assert any(
        p.count_nonzero() > 0
        for name, p in model.named_parameters()
        if "lora_B" in name
    )
    embedded = core.get_input_embeddings()(torch.tensor([30]))
    projected = core.lm_head(torch.ones(1, 16))[0, 30]
    torch.testing.assert_close(projected, embedded.sum())
    model.save_pretrained(tmp_path, save_embedding_layers=True)
    restored = PeftModel.from_pretrained(tiny_model(), tmp_path)
    model.eval()
    restored.eval()
    torch.testing.assert_close(
        response_losses(model, batch), response_losses(restored, batch)
    )

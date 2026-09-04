"""Offline tests for the generic Ordered Compliance HF trainer interface.

These tests deliberately stop at argument parsing and recipe resolution: they
must never download a tokenizer/model or require an available accelerator.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.oc_completion.ordered_data import pi_tag
from src.oc_completion.ordered_train_hf import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_DATA_ROOT,
    DEFAULT_HF_CACHE,
    MODEL_SPECS,
    _requested_recipe_matches_stored,
    _resolve_batch_configuration,
    _validate_resume_configuration,
    build_hf_model,
    parse_args,
    pin_remote_revisions,
    resolve_training_recipe,
)


def _required_args() -> list[str]:
    return ["--pi", "0.3", "--model-seed", "9950"]


def _resume_configuration(
    recipe: dict[str, object],
    *,
    effective_batch: int = 16,
    micro_batch: int = 1,
    accumulation: int = 16,
) -> tuple[dict[str, object], dict[str, object]]:
    requested: dict[str, object] = {
        "recipe": recipe,
        "noise_pi": 0.3,
        "model_seed": 9950,
        "data_fingerprint": "abc123",
        "effective_batch": effective_batch,
        "micro_batch": micro_batch,
        "accumulation": accumulation,
        "eval_batch": 2,
        "max_epochs": 20,
        "patience": 3,
        "max_train_rows": 0,
        "smoke": False,
    }
    stored = {
        "model_recipe": dict(recipe),
        "noise_pi": requested["noise_pi"],
        "model_seed": requested["model_seed"],
        "data_fingerprint": requested["data_fingerprint"],
        "effective_batch_size": requested["effective_batch"],
        "micro_batch_size": requested["micro_batch"],
        "gradient_accumulation": requested["accumulation"],
        "eval_batch_size": requested["eval_batch"],
        "maximum_epochs": requested["max_epochs"],
        "early_stopping_patience": requested["patience"],
        "max_train_rows": requested["max_train_rows"],
        "smoke": requested["smoke"],
    }
    return requested, stored


def test_legacy_arm_defaults_and_run_identity_are_preserved() -> None:
    args = parse_args(["--arm", "llama_lora", *_required_args()])
    recipe = resolve_training_recipe(args, torch.device("cpu"))

    assert args.arm == "llama_lora"
    assert args.data_root == DEFAULT_DATA_ROOT
    assert args.checkpoint_root == DEFAULT_CHECKPOINT_ROOT
    assert args.hf_cache == DEFAULT_HF_CACHE
    assert args.run_dir is None
    assert recipe["model_name"] == MODEL_SPECS["llama_lora"]["model_name"]
    assert recipe["model_kind"] == "decoder"
    assert MODEL_SPECS["llama_lora"]["kind"] == "llama"
    assert recipe["model_tag"] == "llama_lora"

    # This is the historical layout consumed by ordered_eval and the SLURM
    # arrays. Generic model tags must not alter an untouched legacy command.
    expected = (
        DEFAULT_CHECKPOINT_ROOT
        / "llama_lora"
        / f"pi_{pi_tag(args.pi)}"
        / f"seed_{args.model_seed}"
    )
    actual = (
        args.checkpoint_root
        / recipe["model_tag"]
        / f"pi_{pi_tag(args.pi)}"
        / f"seed_{args.model_seed}"
    )
    assert actual == expected


def test_generic_qwen_qlora_arguments_and_directories() -> None:
    args = parse_args(
        [
            "--model-name",
            "Qwen/Qwen2.5-14B",
            "--model-kind",
            "decoder",
            "--model-tag",
            "qwen25_14b_qlora",
            "--model-revision",
            "main",
            "--tokenizer-name",
            "Qwen/Qwen2.5-14B",
            "--peft",
            "--quantization",
            "4bit",
            "--dtype",
            "bf16",
            "--compute-dtype",
            "bf16",
            "--gradient-checkpointing",
            "--lora-r",
            "16",
            "--lora-alpha",
            "32",
            "--lora-dropout",
            "0.05",
            "--lora-targets",
            "q_proj,k_proj,v_proj,o_proj",
            "--data-root",
            "/datasets/oc",
            "--checkpoint-root",
            "/checkpoints/oc",
            "--run-dir",
            "/runs/qwen",
            "--hf-cache",
            "/cache/hf",
            "--micro-batch",
            "1",
            "--gradient-accumulation",
            "16",
            "--eval-batch",
            "2",
            "--resume",
            *_required_args(),
        ]
    )

    assert args.arm is None
    assert args.model_name == "Qwen/Qwen2.5-14B"
    assert args.model_kind == "decoder"
    assert args.model_tag == "qwen25_14b_qlora"
    assert args.model_revision == "main"
    assert args.tokenizer_name == "Qwen/Qwen2.5-14B"
    assert args.peft is True
    assert args.quantization == "4bit"
    assert args.dtype == "bf16"
    assert args.compute_dtype == "bf16"
    assert args.gradient_checkpointing is True
    assert args.lora_r == 16
    assert args.lora_alpha == 32
    assert args.lora_dropout == pytest.approx(0.05)
    assert args.lora_targets == "q_proj,k_proj,v_proj,o_proj"
    assert args.data_root == Path("/datasets/oc")
    assert args.checkpoint_root == Path("/checkpoints/oc")
    assert args.run_dir == Path("/runs/qwen")
    assert args.hf_cache == Path("/cache/hf")
    assert args.micro_batch == 1
    assert args.gradient_accumulation == 16
    assert args.eval_batch == 2
    assert args.resume is True


def test_effective_batch_size_defaults_to_protocol_value() -> None:
    args = parse_args(
        [
            "--model-name",
            "Qwen/Qwen2.5-14B",
            "--model-kind",
            "decoder",
            *_required_args(),
        ]
    )

    assert args.effective_batch_size == 16
    assert _resolve_batch_configuration(args, None, "decoder") == (1, 16, 2, 16)


def test_effective_batch_size_64_is_accepted_and_derives_accumulation() -> None:
    args = parse_args(
        [
            "--model-name",
            "Qwen/Qwen2.5-14B",
            "--model-kind",
            "decoder",
            "--effective-batch-size",
            "64",
            "--micro-batch",
            "64",
            *_required_args(),
        ]
    )

    assert args.effective_batch_size == 64
    assert _resolve_batch_configuration(args, None, "decoder") == (64, 1, 2, 64)


def test_effective_batch_size_64_accepts_micro_batch_32_with_two_steps() -> None:
    args = parse_args(
        [
            "--model-name",
            "Qwen/Qwen2.5-14B",
            "--model-kind",
            "decoder",
            "--effective-batch-size",
            "64",
            "--micro-batch",
            "32",
            "--gradient-accumulation",
            "2",
            *_required_args(),
        ]
    )

    assert _resolve_batch_configuration(args, None, "decoder") == (32, 2, 2, 64)


def test_batch_configuration_checks_product_against_selected_effective_batch() -> None:
    args = parse_args(
        [
            "--model-name",
            "Qwen/Qwen2.5-14B",
            "--model-kind",
            "decoder",
            "--effective-batch-size",
            "64",
            "--micro-batch",
            "16",
            "--gradient-accumulation",
            "1",
            *_required_args(),
        ]
    )

    with pytest.raises(ValueError, match=r"(?i)effective.*64|64.*effective"):
        _resolve_batch_configuration(args, None, "decoder")


@pytest.mark.parametrize("value", ["0", "-1"])
def test_effective_batch_size_must_be_positive(value: str) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model-name",
                "Qwen/Qwen2.5-14B",
                "--model-kind",
                "decoder",
                "--effective-batch-size",
                value,
                *_required_args(),
            ]
        )


def test_remote_model_revision_is_pinned_and_requested_revision_retained(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from transformers import AutoConfig, AutoTokenizer

    calls: list[tuple[str, str, dict[str, object]]] = []

    def fake_config(name: str, **kwargs: object) -> SimpleNamespace:
        calls.append(("config", name, kwargs))
        return SimpleNamespace(_commit_hash="a" * 40)

    def forbidden_tokenizer(name: str, **kwargs: object) -> None:
        raise AssertionError(f"unexpected tokenizer resolution for {name}: {kwargs}")

    monkeypatch.setattr(AutoConfig, "from_pretrained", staticmethod(fake_config))
    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", staticmethod(forbidden_tokenizer)
    )
    requested = resolve_training_recipe(
        parse_args(
            [
                "--model-name",
                "ExampleOrg/Mutable-14B",
                "--model-kind",
                "decoder",
                "--model-revision",
                "release",
                *_required_args(),
            ]
        ),
        torch.device("cpu"),
    )

    pinned = pin_remote_revisions(requested, tmp_path)

    assert pinned["requested_model_revision"] == "release"
    assert pinned["model_revision"] == "a" * 40
    assert pinned["tokenizer_revision"] == "a" * 40
    assert calls == [
        (
            "config",
            "ExampleOrg/Mutable-14B",
            {
                "revision": "release",
                "cache_dir": str(tmp_path),
                "trust_remote_code": False,
                "local_files_only": False,
            },
        )
    ]
    assert _requested_recipe_matches_stored(requested, pinned)


def test_same_model_tokenizer_inherits_pinned_model_revision_and_resumes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from transformers import AutoConfig, AutoTokenizer

    model_name = "ExampleOrg/Shared-Model-And-Tokenizer"
    tokenizer_calls: list[object] = []
    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: SimpleNamespace(_commit_hash="b" * 40)),
    )
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: tokenizer_calls.append((args, kwargs))),
    )
    requested = resolve_training_recipe(
        parse_args(
            [
                "--model-name",
                model_name,
                "--model-kind",
                "decoder",
                "--model-revision",
                "main",
                "--tokenizer-name",
                model_name,
                *_required_args(),
            ]
        ),
        torch.device("cpu"),
    )

    pinned = pin_remote_revisions(requested, tmp_path)

    assert pinned["model_revision"] == "b" * 40
    assert pinned["tokenizer_revision"] == "b" * 40
    assert tokenizer_calls == []
    assert _requested_recipe_matches_stored(requested, pinned)


def test_legacy_recipe_pinning_does_not_import_or_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import builtins

    requested = resolve_training_recipe(
        parse_args(["--arm", "llama_lora", *_required_args()]),
        torch.device("cpu"),
    )
    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "transformers":
            raise AssertionError("legacy revision pinning imported Transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    pinned = pin_remote_revisions(requested, tmp_path)

    assert pinned == requested
    assert pinned is not requested


def test_generic_recipe_normalises_lora_targets() -> None:
    args = parse_args(
        [
            "--model-name",
            "Qwen/Qwen2.5-14B",
            "--model-kind",
            "decoder",
            "--lora-targets",
            " q_proj, k_proj, v_proj, o_proj ",
            *_required_args(),
        ]
    )
    recipe = resolve_training_recipe(args, torch.device("cuda"))

    assert recipe["lora_targets"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert recipe["peft"] is True
    assert recipe["quantization"] == "none"


@pytest.mark.parametrize(
    "targets",
    ["   ", "q_proj,q_proj"],
    ids=["empty", "duplicate"],
)
def test_generic_recipe_rejects_malformed_lora_targets(targets: str) -> None:
    args = parse_args(
        [
            "--model-name",
            "Qwen/Qwen2.5-14B",
            "--model-kind",
            "decoder",
            "--lora-targets",
            targets,
            *_required_args(),
        ]
    )

    with pytest.raises(ValueError, match="(?i)target"):
        resolve_training_recipe(args, torch.device("cpu"))


def test_generic_model_tag_is_deterministic_and_filesystem_safe() -> None:
    command = [
        "--model-name",
        "Qwen/Qwen2.5-14B",
        "--model-kind",
        "decoder",
        "--quantization",
        "4bit",
        *_required_args(),
    ]
    first = resolve_training_recipe(parse_args(command), torch.device("cuda"))
    second = resolve_training_recipe(parse_args(command), torch.device("cuda"))

    assert first["model_tag"] == second["model_tag"]
    assert "/" not in first["model_tag"]
    assert "qwen" in first["model_tag"].lower()
    assert "4bit" in first["model_tag"].lower()


@pytest.mark.parametrize(
    ("first_options", "second_options"),
    [
        (
            ["--model-name", "FirstOrg/Shared-14B"],
            ["--model-name", "SecondOrg/Shared-14B"],
        ),
        (
            ["--model-revision", "release-a"],
            ["--model-revision", "release-b"],
        ),
        (["--dtype", "bf16"], ["--dtype", "fp16"]),
        (["--compute-dtype", "bf16"], ["--compute-dtype", "fp16"]),
        (["--lora-r", "8"], ["--lora-r", "16"]),
        (["--lora-alpha", "16"], ["--lora-alpha", "32"]),
        (["--lora-dropout", "0.0"], ["--lora-dropout", "0.1"]),
        (
            ["--lora-targets", "q_proj,v_proj"],
            ["--lora-targets", "q_proj,k_proj,v_proj,o_proj"],
        ),
    ],
    ids=(
        "model-organisation",
        "model-revision",
        "storage-dtype",
        "compute-dtype",
        "lora-rank",
        "lora-alpha",
        "lora-dropout",
        "lora-targets",
    ),
)
def test_default_model_tag_separates_distinct_training_recipes(
    first_options: list[str],
    second_options: list[str],
) -> None:
    base = [
        "--model-name",
        "Qwen/Shared-14B",
        "--model-kind",
        "decoder",
        *_required_args(),
    ]

    first = resolve_training_recipe(
        parse_args([*base, *first_options]), torch.device("cuda")
    )
    second = resolve_training_recipe(
        parse_args([*base, *second_options]), torch.device("cuda")
    )

    assert first["model_tag"] != second["model_tag"]


def test_resume_configuration_accepts_an_identical_run(tmp_path: Path) -> None:
    recipe = resolve_training_recipe(
        parse_args(
            [
                "--model-name",
                "Qwen/Qwen2.5-14B",
                "--model-kind",
                "decoder",
                *_required_args(),
            ]
        ),
        torch.device("cpu"),
    )
    requested, stored = _resume_configuration(recipe)

    _validate_resume_configuration(
        tmp_path / "config.json", stored, **requested
    )


@pytest.mark.parametrize(
    ("stored_key", "different_value"),
    [
        ("noise_pi", 0.2),
        ("model_seed", 1234),
        ("data_fingerprint", "different-data"),
        ("effective_batch_size", 64),
        ("micro_batch_size", 2),
        ("gradient_accumulation", 8),
        ("eval_batch_size", 4),
        ("maximum_epochs", 10),
        ("early_stopping_patience", 5),
        ("max_train_rows", 100),
        ("smoke", True),
    ],
)
def test_resume_configuration_rejects_changed_run_identity(
    stored_key: str,
    different_value: object,
    tmp_path: Path,
) -> None:
    recipe = resolve_training_recipe(
        parse_args(
            [
                "--model-name",
                "Qwen/Qwen2.5-14B",
                "--model-kind",
                "decoder",
                *_required_args(),
            ]
        ),
        torch.device("cpu"),
    )
    requested, stored = _resume_configuration(recipe)
    stored[stored_key] = different_value

    with pytest.raises(ValueError, match=stored_key):
        _validate_resume_configuration(
            tmp_path / "config.json", stored, **requested
        )


def test_resume_configuration_rejects_changed_model_recipe(tmp_path: Path) -> None:
    recipe = resolve_training_recipe(
        parse_args(
            [
                "--model-name",
                "Qwen/Qwen2.5-14B",
                "--model-kind",
                "decoder",
                *_required_args(),
            ]
        ),
        torch.device("cpu"),
    )
    requested, stored = _resume_configuration(recipe)
    stored_recipe = dict(recipe)
    stored_recipe["lora_r"] = int(recipe["lora_r"]) * 2
    stored["model_recipe"] = stored_recipe

    with pytest.raises(ValueError, match="model_recipe"):
        _validate_resume_configuration(
            tmp_path / "config.json", stored, **requested
        )


@pytest.mark.parametrize(
    ("positive", "negative", "attribute"),
    [
        ("--peft", "--no-peft", "peft"),
        (
            "--gradient-checkpointing",
            "--no-gradient-checkpointing",
            "gradient_checkpointing",
        ),
    ],
)
def test_paired_boolean_flags(
    positive: str,
    negative: str,
    attribute: str,
) -> None:
    base = [
        "--model-name",
        "Qwen/Qwen2.5-14B",
        "--model-kind",
        "decoder",
        *_required_args(),
    ]
    assert getattr(parse_args([*base, positive]), attribute) is True
    assert getattr(parse_args([*base, negative]), attribute) is False


def test_remote_code_and_local_file_flags_are_explicit_opt_ins() -> None:
    base = [
        "--model-name",
        "Qwen/Qwen2.5-14B",
        "--model-kind",
        "decoder",
        *_required_args(),
    ]
    defaults = parse_args(base)
    opted_in = parse_args([*base, "--trust-remote-code", "--local-files-only"])

    assert defaults.trust_remote_code is False
    assert defaults.local_files_only is False
    assert opted_in.trust_remote_code is True
    assert opted_in.local_files_only is True


@pytest.mark.parametrize("quantization", ["none", "8bit", "4bit"])
def test_supported_quantization_choices_parse(quantization: str) -> None:
    args = parse_args(
        [
            "--model-name",
            "Qwen/Qwen2.5-14B",
            "--model-kind",
            "decoder",
            "--quantization",
            quantization,
            *_required_args(),
        ]
    )
    assert args.quantization == quantization


def test_unknown_quantization_is_rejected_by_parser() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model-name",
                "Qwen/Qwen2.5-14B",
                "--model-kind",
                "decoder",
                "--quantization",
                "3bit",
                *_required_args(),
            ]
        )


@pytest.mark.parametrize("quantization", ["8bit", "4bit"])
def test_quantized_training_requires_peft(quantization: str) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model-name",
                "Qwen/Qwen2.5-14B",
                "--model-kind",
                "decoder",
                "--no-peft",
                "--quantization",
                quantization,
                *_required_args(),
            ]
        )


@pytest.mark.parametrize("quantization", ["8bit", "4bit"])
def test_quantized_training_rejects_cpu(quantization: str) -> None:
    args = parse_args(
        [
            "--model-name",
            "Qwen/Qwen2.5-14B",
            "--model-kind",
            "decoder",
            "--quantization",
            quantization,
            "--cpu",
            *_required_args(),
        ]
    )
    with pytest.raises(ValueError, match="(?i)(cuda|gpu|cpu)"):
        resolve_training_recipe(args, torch.device("cpu"))


def test_generic_mode_requires_model_name_and_kind() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--model-name", "Qwen/Qwen2.5-14B", *_required_args()])
    with pytest.raises(SystemExit):
        parse_args(["--model-kind", "decoder", *_required_args()])


def test_quantized_explicit_cuda_zero_is_not_replaced_by_current_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``cuda:0`` remains device zero even when another device is current."""
    import peft
    import transformers

    events: list[object] = []
    load_kwargs: dict[str, object] = {}

    class DummyBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8, use_cache=True)

    class DummyQuantizationConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class DummyLoraConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    def fake_load(_name: str, **kwargs: object) -> DummyBackbone:
        events.append("load")
        load_kwargs.update(kwargs)
        return DummyBackbone()

    def fake_prepare(model: DummyBackbone, **kwargs: object) -> DummyBackbone:
        events.append(("prepare", kwargs))
        return model

    def fake_wrap(model: DummyBackbone, config: DummyLoraConfig) -> DummyBackbone:
        events.append(("peft", config.kwargs))
        return model

    monkeypatch.setattr(
        "src.oc_completion.ordered_train_hf.importlib.util.find_spec",
        lambda _name: object(),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(transformers, "BitsAndBytesConfig", DummyQuantizationConfig)
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        staticmethod(fake_load),
    )
    monkeypatch.setattr(peft, "prepare_model_for_kbit_training", fake_prepare)
    monkeypatch.setattr(peft, "LoraConfig", DummyLoraConfig)
    monkeypatch.setattr(peft, "get_peft_model", fake_wrap)

    args = parse_args(
        [
            "--model-name",
            "example/encoder",
            "--model-kind",
            "encoder",
            "--quantization",
            "4bit",
            *_required_args(),
        ]
    )
    recipe = resolve_training_recipe(args, torch.device("cuda:0"))
    build_hf_model(
        "encoder",
        SimpleNamespace(pad_token_id=0),
        torch.device("cuda:0"),
        torch.bfloat16,
        tmp_path,
        recipe=recipe,
    )

    assert events[0] == "load"
    assert events[1][0] == "prepare"
    assert events[2][0] == "peft"
    assert load_kwargs["device_map"] == {"": 0}

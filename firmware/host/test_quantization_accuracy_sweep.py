import json
import os


def test_quantization_accuracy_sweep_artifact_schema_and_floor():
    path = os.path.join("build", "reports", "quantization_accuracy_sweep.json")
    assert os.path.exists(path), f"Missing artifact: {path}"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = {
        "generated_at_utc",
        "model_name",
        "n_prompts",
        "k_tokens_per_prompt",
        "seed",
        "prompt_token_sequences",
        "fp32_outputs_hash",
        "int4_outputs_hash",
        "aggregate_top1_match_rate",
        "per_prompt_top1_match",
        "per_prompt_top1_match_histogram",
        "quantization_config",
    }
    assert required_keys.issubset(data.keys())

    assert isinstance(data["generated_at_utc"], str) and data["generated_at_utc"].endswith("Z")
    assert isinstance(data["model_name"], str) and data["model_name"]
    assert data["n_prompts"] == 20
    assert data["k_tokens_per_prompt"] == 16
    assert isinstance(data["seed"], int)

    prompts = data["prompt_token_sequences"]
    rates = data["per_prompt_top1_match"]
    hist = data["per_prompt_top1_match_histogram"]
    assert isinstance(prompts, list) and len(prompts) == data["n_prompts"]
    assert isinstance(rates, list) and len(rates) == data["n_prompts"]
    assert isinstance(hist, dict) and sum(hist.values()) == data["n_prompts"]

    for prompt in prompts:
        assert isinstance(prompt, list)
        assert len(prompt) > 0
        for tok in prompt:
            assert isinstance(tok, int)

    for rate in rates:
        assert isinstance(rate, (int, float))
        assert 0.0 <= float(rate) <= 1.0

    agg = float(data["aggregate_top1_match_rate"])
    assert 0.0 < agg <= 1.0
    assert agg >= 0.5

    assert isinstance(data["fp32_outputs_hash"], str) and len(data["fp32_outputs_hash"]) == 64
    assert isinstance(data["int4_outputs_hash"], str) and len(data["int4_outputs_hash"]) == 64

    qcfg = data["quantization_config"]
    assert isinstance(qcfg, dict)
    assert qcfg.get("enabled") is True
    assert qcfg.get("weight_bits") == 4
    assert qcfg.get("group_size") == 64

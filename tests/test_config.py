import pytest
from pydantic import ValidationError

from claimssql.config import EnvSettings, ModelsConfig, load_experiment, load_models


def test_repo_configs_load_and_match_spec():
    models = load_models()
    assert {m.provider for m in models.models} <= {"ollama", "groq"}
    assert models.get(models.models[0].key) is models.models[0]

    exp = load_experiment()
    # Pipeline contract in CLAUDE.md: k=5 samples at temperature 0.7, greedy at 0.
    assert exp.generation.k_samples == 5
    assert exp.generation.sample_seeds == [1, 2, 3, 4, 5]
    assert exp.generation.sample_temperature == 0.7
    assert exp.generation.greedy_temperature == 0.0
    assert exp.verify.result_row_cap == 10_000
    assert exp.metrics.rs_penalties == [0, 1, 10]


def test_duplicate_model_keys_rejected():
    entry = {"key": "a", "provider": "ollama", "model": "x"}
    with pytest.raises(ValidationError):
        ModelsConfig.model_validate({"models": [entry, entry]})


def test_env_settings_repr_hides_key():
    s = EnvSettings(groq_api_key="secret-value", ollama_host="http://localhost:11434")
    assert "secret-value" not in repr(s)
    assert "secret-value" not in str(s)

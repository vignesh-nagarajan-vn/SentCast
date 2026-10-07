"""LLM scorer: label cache, cost cap and the training-cutoff split. No request leaves the machine."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sentcast import experiments as ex
from sentcast import scorers

anthropic = pytest.importorskip("anthropic")


class FakeClient:
    """Stands in for anthropic.Anthropic; labels a headline 2 if it contains 'up', else -2."""
    calls = 0

    def __init__(self, **_):
        self.messages = self

    def create(self, *, messages, **_):
        FakeClient.calls += 1
        lines = messages[0]["content"].split("\n")
        labels = [2 if "up" in line else -2 for line in lines]
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=1000, output_tokens=100, cache_creation_input_tokens=0,
                                  cache_read_input_tokens=0),
            stop_reason="end_turn", content=[SimpleNamespace(type="text", text=json.dumps({"labels": labels}))])


@pytest.fixture()
def fake(monkeypatch, tmp_path):
    monkeypatch.setattr(scorers, "INTERIM", tmp_path)
    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    FakeClient.calls = 0
    return tmp_path


def _texts(n: int) -> list[str]:
    return [f"stocks {'up' if i % 2 else 'down'} {i}" for i in range(n)]


def test_each_distinct_text_is_labelled_once_and_reruns_use_the_cache(fake):
    texts = _texts(100) + _texts(10)                        # ten repeats
    scores, rep = scorers.score_llm(texts)
    assert np.isfinite(scores).all() and set(scores) == {-1.0, 1.0}
    assert scores[1] == 1.0 and scores[0] == -1.0 and (scores[100:] == scores[:10]).all()
    assert rep["labelled"] == 110 and rep["requested_this_run"] == 100 and rep["requests"] == FakeClient.calls == 3
    first_cost = rep["cost_usd"]
    again, rep2 = scorers.score_llm(texts)
    assert FakeClient.calls == 3 and rep2["requested_this_run"] == 0
    assert (again == scores).all() and rep2["cost_usd"] == first_cost


def test_the_cap_counts_spending_from_earlier_runs(fake, monkeypatch):
    cfg = scorers.llm_configs()[0]
    cache_path, usage_path = scorers._llm_cache_paths(cfg)
    cap = cfg["max_cost_usd"]
    usage_path.write_text(json.dumps({"cost_usd": cap - 0.001, "requests": 7}))
    scores, rep = scorers.score_llm(_texts(50))
    assert rep["stopped_at_cap"] and FakeClient.calls == 0
    assert np.isnan(scores).all() and rep["labelled"] == 0 and rep["requests"] == 7


def test_the_second_model_has_its_own_cache_and_settings(fake):
    first, second = scorers.llm_configs()
    assert second["model"] != first["model"] and second["batch_size"] == first["batch_size"]
    assert second["model"] in scorers.LLM_PRICES_USD_PER_MTOK and "second" not in second
    scorers.score_llm(_texts(40), first)
    _, rep = scorers.score_llm(_texts(40), second)
    assert rep["model"] == second["model"] and rep["requested_this_run"] == 40 and FakeClient.calls == 2
    assert scorers._llm_cache_paths(first)[0] != scorers._llm_cache_paths(second)[0]


def test_post_cutoff_origins_have_their_whole_window_after_the_cutoff_month():
    origins = pd.period_range("2025-05", "2025-11", freq="M")
    post = ex.llm_post_cutoff(origins, "2025-07")
    # the window of origin 2025-08 starts at the end of July, inside the cutoff month
    assert list(origins[post.values].astype(str)) == ["2025-09", "2025-10", "2025-11"]

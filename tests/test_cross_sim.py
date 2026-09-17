#!/usr/bin/env python3
"""
Offline checks for the cross-model simulation. No API calls: the providers are
replaced with a stub that echoes its input back, which also exercises the
echo-stripping path.

    python tests/test_cross_sim.py        # or: pytest tests/test_cross_sim.py
"""

import random
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cross_sim import simulate as sim_mod
from cross_sim.pairing import ModelPair, allocate_stories, build_pair_grid
from cross_sim.providers import BaseProvider, strip_input_echo, truncate_user_input
from cross_sim.story_table import build_story_table

MODELS = ["gpt", "claude", "llama", "qwen"]
CONFIGS = [{"id": m, "provider": "openai", "model_name": m, "env_key": "X"} for m in MODELS]


class StubProvider(BaseProvider):
    """Echoes the partner's input, then adds its own words."""

    def __init__(self, tag):
        self.tag = tag
        self.n = 0

    def generate(self, system_prompt, user_input, history=None, temperature=1.0, max_tokens=35):
        self.n += 1
        return f"{user_input} {self.tag} line {self.n} continues on from there"

    def reset_conversation(self):
        self.n = 0


def _stub_pool(model_configs):
    return {c["id"]: [StubProvider(f"{c['id']}-{s}") for s in (0, 1)] for c in model_configs}


def test_grid_sizes():
    assert len(build_pair_grid(MODELS, "all")) == 16
    assert len(build_pair_grid(MODELS, "cross")) == 12
    assert len(build_pair_grid(MODELS, "same")) == 4
    assert build_pair_grid(["only"], "cross") == []


def test_pair_labels():
    assert ModelPair("a", "b").pair_id != ModelPair("b", "a").pair_id
    assert ModelPair("a", "b").dyad_id == ModelPair("b", "a").dyad_id
    assert ModelPair("a", "a").is_self_pair
    assert not ModelPair("a", "b").is_self_pair


def test_allocation():
    grid = build_pair_grid(MODELS, "all")
    assert sum(allocate_stories(grid, stories_per_pair=7).values()) == 112
    assert sum(allocate_stories(grid, stories_per_pair=6).values()) == 96

    uneven = allocate_stories(grid, n_stories=100)
    assert sum(uneven.values()) == 100
    assert sorted(uneven.values()) == [6] * 12 + [7] * 4


def test_truncation_and_echo():
    assert truncate_user_input("one two three four five", 2) == "one two three"
    assert truncate_user_input("one two", 5) == ""
    assert strip_input_echo("Hello there friend", "hello there") == "friend"
    assert strip_input_echo("no overlap", "something") == "no overlap"


def _run(stories_per_pair=2, mode="all"):
    sim_mod.build_provider_pool = _stub_pool
    random.seed(42)
    allocation = allocate_stories(build_pair_grid(MODELS, mode), stories_per_pair=stories_per_pair)
    return sim_mod.simulate_cross_model_dataset(
        model_configs=CONFIGS,
        allocation=allocation,
        n_turns_per_story=10,
        delay_between_stories=0.0,
    )


def test_simulation_shape():
    df = _run()
    assert df["story_id"].nunique() == 32
    assert len(df) == 320
    assert set(df.columns) == {
        "turn", "agent_1", "agent_2", "story_id", "model_starter",
        "model_responder", "pair_id", "dyad_id", "is_self_pair", "timestamp",
    }
    assert df.groupby("story_id")["turn"].max().eq(10).all()


def test_missing_key_skips_cells():
    """A model with no key drops its cells rather than failing the run."""
    sim_mod.build_provider_pool = lambda cfgs: _stub_pool(
        [c for c in cfgs if c["id"] != "qwen"]
    )
    random.seed(42)
    allocation = allocate_stories(build_pair_grid(MODELS, "all"), stories_per_pair=1)
    df = sim_mod.simulate_cross_model_dataset(
        model_configs=CONFIGS, allocation=allocation,
        n_turns_per_story=2, delay_between_stories=0.0,
    )
    assert df["story_id"].nunique() == 9  # the 3x3 grid that remains
    assert "qwen" not in set(df["model_starter"]) | set(df["model_responder"])


def test_story_table():
    df = _run()
    table, id_map = build_story_table(df, condition="aa_cross", max_turns=10, seed=42)

    assert len(table) == 32 == len(id_map)
    assert table["conversation_id"].nunique() == 32
    assert table["condition"].eq("aa_cross").all()
    assert table["is_self_pair"].sum() == 8
    assert table["dyad_id"].nunique() == 10  # 6 cross dyads + 4 self
    assert (table["model_id"] == table["pair_id"]).all()

    # The primer never survives into the text.
    assert not table["full_story"].str.contains("This is the story of").any()
    assert not table["full_author_1"].str.contains("This is the story of").any()

    # The swap moves text and model labels together; starter stays truthful.
    for _, r in table.iterrows():
        if r["starter"] == "author_1":
            assert (r["model_author_1"], r["model_author_2"]) == (r["model_starter"], r["model_responder"])
        else:
            assert (r["model_author_1"], r["model_author_2"]) == (r["model_responder"], r["model_starter"])
    assert table["starter"].isin(["author_1", "author_2"]).all()
    assert table["starter_side"].equals(table["starter"])


def test_full_story_is_chronological():
    """full_story always opens with the starter's first turn, swap or no swap."""
    df = _run()
    table, id_map = build_story_table(df, condition="aa_cross", max_turns=10, seed=42)
    conv_of = dict(zip(id_map["story_id"], id_map["conversation_id"]))

    for story_id, grp in df.groupby("story_id"):
        row = table[table["conversation_id"] == conv_of[story_id]].iloc[0]
        opener = grp.sort_values("turn").iloc[0]["agent_1"]
        opener = " ".join(str(opener).replace("This is the story of", "").split())
        assert row["full_story"].startswith(opener)


def test_is_self_pair_survives_a_csv_roundtrip():
    """Resuming reads the checkpoint back as strings; bool("False") is True."""
    df = _run(stories_per_pair=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "turns.csv"
        df.to_csv(path, index=False)
        table, _ = build_story_table(pd.read_csv(path), seed=42)
    assert table["is_self_pair"].sum() == 4


def test_counterbalancing_is_seeded():
    df = _run()
    a, _ = build_story_table(df, seed=42)
    b, _ = build_story_table(df, seed=42)
    c, _ = build_story_table(df, seed=7)
    assert a["starter"].tolist() == b["starter"].tolist()
    assert c["starter"].tolist() != a["starter"].tolist()


def test_checkpoint_resume(tmp_path=None):
    ckpt = Path(tmp_path if tmp_path else tempfile.mkdtemp()) / "ckpt.csv"
    ckpt.unlink(missing_ok=True)

    sim_mod.build_provider_pool = _stub_pool
    allocation = allocate_stories(build_pair_grid(MODELS, "all"), stories_per_pair=1)
    kwargs = dict(
        model_configs=CONFIGS, allocation=allocation,
        n_turns_per_story=2, delay_between_stories=0.0, checkpoint_path=str(ckpt),
    )
    random.seed(42)
    first = sim_mod.simulate_cross_model_dataset(**kwargs)
    assert first["story_id"].nunique() == 16
    assert pd.read_csv(ckpt)["story_id"].nunique() == 16

    done = set(pd.read_csv(ckpt)["story_id"])
    again = sim_mod.simulate_cross_model_dataset(**kwargs, completed_story_ids=done)
    assert again.empty

    partial = sim_mod.simulate_cross_model_dataset(
        **kwargs, completed_story_ids=set(list(done)[:10])
    )
    assert partial["story_id"].nunique() == 6

    ckpt.unlink(missing_ok=True)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    with tempfile.TemporaryDirectory() as tmp:
        for fn in tests:
            if fn.__name__ in ("test_checkpoint_resume",):
                fn(tmp_path=tmp)
            else:
                fn()
            print(f"  ok  {fn.__name__}")
    print(f"\n{len(tests)} checks passed")

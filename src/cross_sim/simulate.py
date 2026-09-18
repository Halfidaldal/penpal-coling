"""
Running the model x model grid.

A single story is generated exactly as in PenPal-EMNLP
(src/nes/simulation.py simulate_single_story): ten turns, both sides on the
same prompt, echo stripped, both outputs word-truncated 2..5 words before they
are saved and handed to the partner. The one change is that the two sides can
be different models, so each turn records which model wrote it.

Chronology: agent_1 is always the starter and agent_2 always the responder.
Counterbalancing of the author_1 / author_2 column labels happens later, in
story_table.py, the same way the EMNLP pipeline did it.
"""

import os
import random
import time
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from .pairing import ModelPair
from .prompts import SystemPrompts
from .providers import (
    BaseProvider,
    get_provider,
    strip_input_echo,
    truncate_user_input,
)


def simulate_single_story(
    provider_starter: BaseProvider,
    provider_responder: BaseProvider,
    n_turns: int,
    story_id: str,
    pair: ModelPair,
    temperature: float = 1.0,
    max_tokens: int = 40,
    truncate_words_min: int = 2,
    truncate_words_max: int = 5,
) -> pd.DataFrame:
    """
    Simulate one story between two (possibly different) models.

    Args:
        provider_starter: provider for the model that writes turn 1.
        provider_responder: provider for the model that replies. Must be a
            distinct object from provider_starter even for a self-pair, because
            threaded providers keep per-conversation state.
        n_turns: turns per story; one turn is starter + responder.
        story_id: unique id for this story.
        pair: the grid cell, recorded on every row.
        temperature, max_tokens: sampling settings, shared by both sides.
        truncate_words_min, truncate_words_max: inclusive range for the random
            number of words trimmed off each output.

    Returns:
        One row per turn: turn, agent_1, agent_2, story_id, model_starter,
        model_responder, pair_id, dyad_id, is_self_pair, timestamp.
    """
    provider_starter.reset_conversation()
    provider_responder.reset_conversation()

    data = []

    # Each side keeps its own alternating history: "user" is the partner's
    # text, "assistant" is its own.
    starter_history: List[Dict[str, str]] = []
    responder_history: List[Dict[str, str]] = []

    responder_last_text = ""

    for turn_idx in range(n_turns):
        turn_number = turn_idx + 1
        system_prompt = SystemPrompts.get_system_prompt(turn_number)

        # === starter's turn ===
        if turn_idx == 0:
            starter_input = SystemPrompts.STORY_PREFIX
        else:
            starter_input = responder_last_text

        starter_text = provider_starter.generate(
            system_prompt=system_prompt,
            user_input=starter_input,
            history=starter_history if starter_history else None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        starter_text = strip_input_echo(starter_text, starter_input)
        starter_text = truncate_user_input(
            starter_text,
            random.randint(truncate_words_min, truncate_words_max),
        )

        starter_history.append({"role": "user", "content": starter_input})
        starter_history.append({"role": "assistant", "content": starter_text})

        if turn_idx == 0:
            starter_text_for_data = f"{SystemPrompts.STORY_PREFIX}\n{starter_text}"
        else:
            starter_text_for_data = starter_text

        # === responder's turn ===
        responder_input = starter_text

        responder_text = provider_responder.generate(
            system_prompt=system_prompt,
            user_input=responder_input,
            history=responder_history if responder_history else None,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        responder_text = strip_input_echo(responder_text, responder_input)
        responder_text = truncate_user_input(
            responder_text,
            random.randint(truncate_words_min, truncate_words_max),
        )

        responder_history.append({"role": "user", "content": responder_input})
        responder_history.append({"role": "assistant", "content": responder_text})
        responder_last_text = responder_text

        data.append({
            "turn": turn_number,
            "agent_1": starter_text_for_data,
            "agent_2": responder_text,
            "story_id": story_id,
            "model_starter": pair.starter,
            "model_responder": pair.responder,
            "pair_id": pair.pair_id,
            "dyad_id": pair.dyad_id,
            "is_self_pair": pair.is_self_pair,
            "timestamp": pd.Timestamp.now(),
        })

    return pd.DataFrame(data)


def retry_with_backoff(func, max_retries: int = 5, base_delay: float = 2.0):
    """Retry on rate-limit errors with exponential backoff. Other errors raise."""
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = (
                "429" in error_str
                or "rate limit" in error_str
                or "rate_limit" in error_str
                or "too many requests" in error_str
            )
            if not is_rate_limit:
                raise

            last_exception = e
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            if attempt < max_retries - 1:
                print(f"  rate limited, waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"  rate limit persists after {max_retries} attempts")

    raise last_exception


def build_provider_pool(model_configs: List[Dict]) -> Dict[str, List[BaseProvider]]:
    """
    Two provider objects per model, one for each side of a story.

    A self-pair needs two distinct objects because the OpenAI adapter threads
    the conversation server-side; sharing one would merge both sides of the
    dialogue into a single thread.

    Models whose API key is missing from the environment are skipped with a
    warning rather than failing the run.
    """
    pool: Dict[str, List[BaseProvider]] = {}

    for cfg in model_configs:
        model_id = cfg["id"]
        api_key = os.environ.get(cfg["env_key"])
        if not api_key:
            print(f"  skipping {model_id}: {cfg['env_key']} not set")
            continue
        pool[model_id] = [
            get_provider(cfg["provider"], api_key, cfg["model_name"], cfg.get("base_url"))
            for _ in range(2)
        ]

    return pool


def simulate_cross_model_dataset(
    model_configs: List[Dict],
    allocation: Dict[ModelPair, int],
    n_turns_per_story: int = 10,
    temperature: float = 1.0,
    max_tokens: int = 40,
    delay_between_stories: float = 2.0,
    checkpoint_path: Optional[str] = None,
    completed_story_ids: Optional[set] = None,
) -> pd.DataFrame:
    """
    Generate every story in the allocation, rotating through cells round-robin.

    Round-robin matters more here than in the same-model design: consecutive
    stories from one cell hammer the same one or two providers, whereas
    rotating spreads the load and keeps a rate limit on one provider from
    stalling the whole run.

    Args:
        model_configs: model config dicts (id, provider, model_name, env_key,
            optional base_url).
        allocation: cell -> number of stories, from pairing.allocate_stories.
        n_turns_per_story, temperature, max_tokens: generation settings.
        delay_between_stories: seconds to sleep between stories.
        checkpoint_path: if given, each finished story is appended here as soon
            as it completes, so an interrupted run loses at most one story.
        completed_story_ids: story ids already present in the checkpoint; these
            are skipped.

    Returns:
        All turns from this run, concatenated. Excludes anything skipped as
        already complete -- the caller reads those back from the checkpoint.
    """
    pool = build_provider_pool(model_configs)
    if not pool:
        print("\nNo models available. Check API keys.")
        return pd.DataFrame()

    completed_story_ids = completed_story_ids or set()

    # Drop cells whose models have no key, so the run does what it can rather
    # than dying on the first unavailable provider.
    runnable = {}
    for pair, n in allocation.items():
        missing = [m for m in (pair.starter, pair.responder) if m not in pool]
        if missing:
            print(f"  skipping cell {pair.pair_id}: no key for {', '.join(sorted(set(missing)))}")
            continue
        runnable[pair] = n

    if not runnable:
        print("\nNo runnable cells. Check API keys.")
        return pd.DataFrame()

    # Flatten to a round-robin queue: one story from each cell, then the next
    # from each cell, and so on.
    queue = []
    for story_num in range(1, max(runnable.values()) + 1):
        for pair, n in runnable.items():
            if story_num <= n:
                queue.append((pair, story_num))

    pending = [
        (pair, num) for pair, num in queue
        if _story_id(pair, num) not in completed_story_ids
    ]
    n_skipped = len(queue) - len(pending)
    if n_skipped:
        print(f"  resuming: {n_skipped} stories already in checkpoint, {len(pending)} to go")

    all_stories = []
    failures = []

    with tqdm(total=len(pending), desc="stories") as pbar:
        for i, (pair, story_num) in enumerate(pending):
            story_id = _story_id(pair, story_num)
            provider_starter = pool[pair.starter][0]
            provider_responder = pool[pair.responder][1]

            try:
                df_story = retry_with_backoff(
                    lambda: simulate_single_story(
                        provider_starter=provider_starter,
                        provider_responder=provider_responder,
                        n_turns=n_turns_per_story,
                        story_id=story_id,
                        pair=pair,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ),
                    max_retries=5,
                    base_delay=2.0,
                )
                all_stories.append(df_story)
                if checkpoint_path:
                    _append_checkpoint(df_story, checkpoint_path)
            except Exception as e:
                print(f"\n  failed {story_id}: {e}")
                failures.append((story_id, str(e)))

            pbar.update(1)
            pbar.set_postfix_str(pair.pair_id)

            if i < len(pending) - 1:
                time.sleep(delay_between_stories)

    if failures:
        print(f"\n{len(failures)} stories failed:")
        for story_id, err in failures:
            print(f"  {story_id}: {err}")

    if not all_stories:
        return pd.DataFrame()

    result = pd.concat(all_stories, ignore_index=True)
    print(f"\nGenerated {len(result)} turns across {len(all_stories)} stories")
    return result


def _story_id(pair: ModelPair, story_num: int) -> str:
    """Deterministic id, so a resumed run recognises what it already has."""
    return f"{pair.starter}__{pair.responder}__story_{story_num:03d}"


def _append_checkpoint(df_story: pd.DataFrame, path: str) -> None:
    exists = os.path.exists(path)
    df_story.to_csv(path, mode="a", header=not exists, index=False)

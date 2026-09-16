"""Rubric keys, in every language a judge prompt is written in.

The stage-3 rubric asks the judge to return a JSON object whose top-level keys
are the three quality dimensions, named in the prompt's own language. The
Chinese prompt asks for ``内容深度与价值（45分）``; the English one asks for
``Content Depth and Value (45 points)``. They are the same dimension, and the
scoring code has to read either without knowing which prompt produced the file.

Stage 2 needs none of this. Its output schema -- ``checklist`` /
``instruction_point`` / ``score`` / ``reason`` -- is ASCII in both prompts, so
its parser was already language-agnostic.

Matching is tried in three passes, narrowest first: an exact alias, then a
normalised alias (case, whitespace and both kinds of bracket folded away), then
the leading label before the bracket. The third pass is what absorbs a judge
that writes "Content Depth and Value (45 pts)" or drops the bracket entirely.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Dimension:
    """One quality dimension, and every name a judge has been asked to use.

    Attributes:
        name: The English label used in the paper's tables and in the
            leaderboard this repository prints.
        max_score: Points available. Also the denominator the judge is asked to
            write, as in ``"45/45"``.
        aliases: Exact top-level JSON keys, one per judge prompt language.
    """

    name: str
    max_score: int
    aliases: tuple[str, ...]


#: The three dimensions, in the order the rubric presents them. Their points sum
#: to 100, which is the stage-3 score.
DIMENSIONS: tuple[Dimension, ...] = (
    Dimension(
        name="Content Substance",
        max_score=45,
        aliases=("内容深度与价值（45分）", "Content Depth and Value (45 points)"),
    ),
    Dimension(
        name="Narrative Engagement",
        max_score=30,
        aliases=("结构与叙事设计（30分）", "Structure and Narrative Design (30 points)"),
    ),
    Dimension(
        name="Conversational Naturalness",
        max_score=25,
        aliases=(
            "语言表达与传播效果（25分）",
            "Language Expression and Communication Impact (25 points)",
        ),
    ),
)

#: Keys a dimension's score can arrive under, in the order they are tried. The
#: judge is asked for the first one in Chinese and the second in English; the
#: lowercase variant is a model that ignored the capital.
SCORE_KEYS: tuple[str, ...] = ("得分", "Score", "score")


def _normalise(text: str) -> str:
    """Fold a key down to what is common between the languages.

    Applies NFKC so full-width brackets become ASCII ones, lowercases, and drops
    everything that is not a letter or a digit. ``"Content Depth and Value (45
    points)"`` and ``"content depth & value (45 pts)"`` do not survive this as
    the same string -- that is what `_label` is for -- but casing and bracket
    width do.

    Args:
        text: A raw JSON key as the judge emitted it.

    Returns:
        The folded form, for comparison only.
    """
    folded = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[^0-9a-z一-鿿]+", "", folded)


def _label(text: str) -> str:
    """The part of a key before its bracketed point count.

    ``"Content Depth and Value (45 points)"`` -> ``"contentdepthandvalue"``. This
    is the loosest match, and it is why a judge that writes ``(45 pts)`` or omits
    the bracket is still read rather than silently scored zero.

    Args:
        text: A raw JSON key as the judge emitted it.

    Returns:
        The folded label, with any trailing bracketed group removed.
    """
    folded = unicodedata.normalize("NFKC", text)
    return _normalise(re.split(r"[(（]", folded)[0])


def find_dimension(result: dict, dimension: Dimension) -> dict | None:
    """The judge's entry for one dimension, whatever it called it.

    Args:
        result: The parsed stage-3 JSON object.
        dimension: The dimension being looked for.

    Returns:
        The dimension's own object, or None when the judge did not emit it.
        A None here means the judge answered in a shape nobody asked for, which
        is a failed evaluation rather than a zero.
    """
    for alias in dimension.aliases:
        entry = result.get(alias)
        if isinstance(entry, dict):
            return entry

    wanted_exact = {_normalise(a) for a in dimension.aliases}
    wanted_label = {_label(a) for a in dimension.aliases}
    for pass_ in (wanted_exact, wanted_label):
        for key, entry in result.items():
            if not isinstance(entry, dict):
                continue
            seen = _normalise(key) if pass_ is wanted_exact else _label(key)
            if seen in pass_:
                return entry
    return None


def read_score(entry: dict) -> tuple[float, float] | None:
    """One dimension's ``"<score>/<max>"`` string, as two numbers.

    Args:
        entry: A dimension's own object from the stage-3 JSON.

    Returns:
        ``(score, max_score)``, or None when no score key is present or the
        value is not a pair of numbers. None is a parse failure, not a zero.
    """
    for key in SCORE_KEYS:
        if key not in entry:
            continue
        parts = str(entry[key]).split("/")
        if len(parts) < 2:
            continue
        try:
            return float(parts[0].strip()), float(parts[1].strip())
        except ValueError:
            continue
    return None


def dimension_scores(result: dict) -> tuple[list[tuple[float, float]], bool]:
    """Every dimension's score, and whether all three were readable.

    Args:
        result: The parsed stage-3 JSON object.

    Returns:
        ``(pairs, complete)``. ``pairs`` holds ``(score, max_score)`` for each
        dimension that could be read, in rubric order; ``complete`` is False if
        any dimension was missing or malformed, which is the signal to exclude
        the sample rather than to score it low.
    """
    pairs: list[tuple[float, float]] = []
    for dimension in DIMENSIONS:
        entry = find_dimension(result, dimension)
        if entry is None:
            return pairs, False
        score = read_score(entry)
        if score is None:
            return pairs, False
        pairs.append(score)
    return pairs, True

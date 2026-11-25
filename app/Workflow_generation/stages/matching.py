"""Shared text-matching helpers used by the planner (legacy direct mode) and
the deterministic resolver.

Everything here is pure, deterministic and cheap: tokenisation, a loose domain
synonym table, and scoring functions that compare *intent text* against catalog
node metadata.  No LLM, no embeddings — the hybrid retriever supplies the
semantic signal; these functions supply the exact/lexical one.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

STOPWORDS = {
    "the", "a", "an", "for", "and", "or", "of", "to", "in", "on", "with", "by",
    "step", "node", "plan", "action", "record", "data", "when", "if",
    "this", "that", "it", "as", "at", "be", "is", "are", "from", "into",
    "string", "other", "value", "list", "item", "items",
    # NOTE: "send" is deliberately NOT a stopword — it is the verb that
    # distinguishes "send an email" from "edit a field".
}

#: Loose domain synonyms.  A 4B model writes "parse the pdf" while the catalog
#: calls the same thing "Extract ... with MUNTQ AI"; without this bridge the
#: lexical score is 0 for every real candidate and the fallback picks
#: whichever node happens to sort first.
CONCEPTS: dict[str, set[str]] = {
    "pdf": {"extract", "read", "file", "document", "text", "attachment"},
    "parse": {"extract", "read", "text", "ai", "analyze"},
    "extract": {"extract", "read", "ai", "text"},
    "ocr": {"extract", "read", "ai", "text"},
    "excel": {"spreadsheet", "sheet", "row", "rows", "file", "csv"},
    # "spreadsheet" deliberately does NOT bridge to "file": that generic word
    # is what let "Read spreadsheet" outrank "Add a row" for an *add* intent.
    "spreadsheet": {"spreadsheet", "sheet", "row", "rows"},
    "sheet": {"spreadsheet", "sheet", "row", "rows"},
    "add": {"add", "append", "row", "rows", "create", "store", "write"},
    "save": {"add", "row", "rows", "write", "create", "file", "appends"},
    "write": {"add", "row", "rows", "create", "file", "writes"},
    "store": {"add", "row", "rows", "create", "file"},
    "append": {"add", "row", "rows", "appends"},
    "fetch": {"get", "read", "email", "message", "inbox"},
    "get": {"get", "read", "email", "message"},
    "inbox": {"email", "message", "get"},
    "attachment": {"file", "email", "message", "attachment"},
    "attach": {"file", "email", "message"},
    "email": {"email", "message"},
    "mail": {"email", "message"},
    "outlook": {"outlook", "email", "message"},
    "gmail": {"gmail", "email", "message"},
    "row": {"row", "rows", "sheet", "spreadsheet"},
    "rows": {"row", "rows", "sheet", "spreadsheet"},
    "file": {"file", "files"},
    "if": {"condition", "branch", "tests", "true", "false"},
    "check": {"condition", "branch", "tests", "true", "false"},
    "condition": {"condition", "branch", "tests"},
    "branch": {"branch", "condition", "true", "false"},
    "schedule": {"schedule", "cron"},
    "webhook": {"webhook", "request"},
    "api": {"api", "request", "http"},
    "http": {"http", "api", "request"},
    "code": {"code", "javascript", "python", "script"},
    "script": {"code", "javascript", "python", "run"},
    "transform": {"transform", "edit", "fields", "convert"},
    "convert": {"convert", "file", "csv", "excel"},
    "classify": {"classify", "ai", "analyze"},
    "summarize": {"summarize", "ai", "analyze"},
    "ai": {"ai", "extract", "analyze"},
    "invoice": {"invoice", "bill"},
    "reply": {"reply", "respond", "email", "message"},
    "notify": {"send", "email", "message", "notify"},
    "send": {"send", "email", "message", "notify", "reply"},
    "wait": {"wait", "delay", "schedule"},
    "join": {"join", "merge"},
    "file_out": {"file", "csv", "excel", "convert"},
    # trigger phrasings a small model actually writes
    "every": {"schedule", "minute", "hour", "day", "week", "month"},
    "daily": {"schedule", "day"},
    "hourly": {"schedule", "hour"},
    "weekly": {"schedule", "week"},
    "monthly": {"schedule", "month"},
    "upload": {"file", "form", "submit"},
    "submit": {"form", "submit"},
    "form": {"form", "submit"},
    "request": {"http", "api", "webhook", "request"},
    "message": {"email", "message", "inbox"},
    "ticket": {"ticket", "support", "help"},
    "google": {"sheets", "spreadsheet"},
    "attached": {"attachment", "file", "email", "message"},
    "arrives": {"received", "message", "email", "request"},
    "arrive": {"received", "message", "email", "request"},
    "receives": {"received", "message", "email"},
    "uploaded": {"file", "form", "submit"},
}


def _stem(word: str) -> str:
    """Crude plural normaliser so 'attachment' matches 'attachments'."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def stems(wordset: set[str]) -> set[str]:
    return {_stem(w) for w in wordset}


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 2}


def expand(words_set: set[str]) -> set[str]:
    out = set(words_set)
    for word in words_set:
        out |= CONCEPTS.get(word, set()) or CONCEPTS.get(_stem(word), set())
    return out


def intent_score(probe: set[str], entry: dict[str, Any]) -> float:
    """Score one catalog entry against a set of intent words (0..~1)."""
    if not probe:
        return 0.0
    hay = " ".join(
        str(entry.get(k) or "")
        for k in ("name", "description", "summary", "appName", "nodeTypeId", "id")
    )
    target = words(hay)
    if not target:
        return 0.0
    probe_s, target_s = stems(probe), stems(target)
    direct = len(probe_s & target_s)
    # a concept match is worth less than a literal word match
    bridged = len(stems(expand(probe) - probe) & target_s)
    return (direct + 0.6 * bridged) / max(1, len(probe))


def plan_intent(parts: Iterable[Any]) -> set[str]:
    """Content words describing what a step should accomplish."""
    text = " ".join(str(x or "") for x in parts)
    return {w for w in words(text) if w not in STOPWORDS}


def concept_match(concept: str, target: str) -> float:
    """How well a plain-language data concept matches a label/path/description.

    Returns 0..1; 0 means "no relationship at all".  Used by the data-flow
    stage to pair consumer inputs with producer outputs.
    """
    probe = plan_intent([concept])
    if not probe:
        return 0.0
    target_words = words(target)
    if not target_words:
        return 0.0
    probe_s, target_s = stems(probe), stems(target_words)
    direct = len(probe_s & target_s)
    bridged = len(stems(expand(probe) - probe) & target_s)
    score = (direct + 0.6 * bridged) / max(1, len(probe))
    # a target that contains the concept's head word outright is a strong hit
    head = _stem(max(probe, key=len))
    if head and head in target_s:
        score = max(score, 0.8)
    return min(1.0, score)


def match_operation(hint: str | None, options: Sequence[str]) -> str | None:
    """Match an operation hint ("convert to excel") to a declared option.

    Returns the canonical option string when compatible, None when the node has
    an operation axis but no option fits the hint (incompatible), and None when
    there is no hint.  Callers distinguish "no operation axis" via the option
    list being empty.
    """
    if not hint or not options:
        return None
    hint_words = {w for w in words(hint) if w not in STOPWORDS}
    if not hint_words:
        return None
    hint_s = stems(hint_words)
    best: tuple[float, str] | None = None
    for option in options:
        option_words = words(option)
        if not option_words:
            continue
        option_s = stems(option_words)
        overlap = len(hint_s & option_s) / max(1, len(hint_s))
        # "excel" must not match "Convert to CSV"; require the hint's content
        # words to be substantially present in the option.
        if overlap >= 0.6 or hint_s <= option_s:
            if best is None or overlap > best[0]:
                best = (overlap, option)
    return best[1] if best else None

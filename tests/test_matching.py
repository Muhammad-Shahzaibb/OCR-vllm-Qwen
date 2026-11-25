"""Unit tests for the shared deterministic text-matching helpers."""

from __future__ import annotations

from app.Workflow_generation.stages.matching import (
    concept_match,
    expand,
    intent_score,
    match_operation,
    plan_intent,
    stems,
    words,
)


def _entry(name: str, **extra):
    return {"nodeTypeId": f"x.{name}", "name": name, **extra}


class TestTokenisation:
    def test_words_are_lowercased_and_length_filtered(self):
        assert words("Send an Email") == {"send", "email"}

    def test_stems_normalise_plurals(self):
        assert stems({"attachments", "rows"}) == {"attachment", "row"}

    def test_plan_intent_drops_stopwords(self):
        assert "the" not in plan_intent(["the invoice for the customer"])
        assert "invoice" in plan_intent(["the invoice for the customer"])


class TestIntentScore:
    def test_clear_match_scores_high(self):
        probe = plan_intent(["extract data from a pdf"])
        assert intent_score(probe, _entry("Extract with MUNTQ AI")) >= 0.5

    def test_unrelated_entry_scores_zero(self):
        probe = plan_intent(["extract data from a pdf"])
        assert intent_score(probe, _entry("On a schedule")) == 0.0

    def test_empty_probe_scores_zero(self):
        assert intent_score(set(), _entry("anything")) == 0.0

    def test_send_is_a_meaningful_word(self):
        """'send' must not be swallowed by the stopword list."""
        probe = plan_intent(["send an email to finance"])
        assert "send" in probe
        assert intent_score(probe, _entry("Send a message")) >= 0.5


class TestConceptBridging:
    def test_expand_adds_loose_synonyms(self):
        assert "extract" in expand({"pdf"})
        assert "attachment" in expand({"attached"})

    def test_concept_match_pairs_data(self):
        assert concept_match("extracted rows", "Rows from a field") >= 0.5

    def test_concept_match_rejects_unrelated(self):
        assert concept_match("the attached file", "Message.Attachments") >= 0.5
        assert concept_match("quarterly budget", "Message.Attachments") == 0.0


class TestMatchOperation:
    OPTIONS = ["Extract", "Classify", "Summarize"]

    def test_matches_a_declared_option(self):
        assert match_operation("convert to excel", ["Convert to CSV", "Convert to Excel"]) == "Convert to Excel"

    def test_rejects_an_incompatible_hint(self):
        assert match_operation("convert to excel", self.OPTIONS) is None

    def test_no_hint_is_not_a_rejection_signal(self):
        assert match_operation(None, self.OPTIONS) is None

    def test_no_operation_axis(self):
        assert match_operation("extract", []) is None

    def test_record_operation_still_matches(self):
        assert match_operation("create record", ["Get records", "Create record", "Update record"]) == "Create record"

"""The 6 shipped exports are structural reference fixtures.

They predate the current catalog (retired node types, legacy option values,
keys the catalog no longer declares), so they are validated under an
``include_retired`` policy: **structure** and **invented node types** must be
clean; availability and option-value findings are expected and tolerated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.Workflow_generation.catalog.manager import NodeCatalogManager
from app.Workflow_generation.config import WorkflowSettings
from app.Workflow_generation.stages.validator import validate_workflow

EXPORTS_DIR = Path(__file__).resolve().parents[1] / "app" / "existing_workflows"
EXPORTS = sorted(EXPORTS_DIR.glob("*.json"))

#: Findings that would mean the *structure* of a reference export is broken.
STRUCTURAL_CODES = {
    "INVALID_JSON",
    "MISSING_TOP_LEVEL",
    "INVALID_TYPE",
    "NO_NODES",
    "NODE_MISSING_FIELD",
    "NODE_INVALID_KEY",
    "DUPLICATE_NODE_KEY",
    "DANGLING_EDGE",
    "INVALID_EDGE_KIND",
    "DUPLICATE_EDGE",
    "MISSING_ROUTE_LABEL",
    "MISSING_ROUTE_ORDER",
    "DUPLICATE_ROUTE_ORDER",
    "NO_ROOT",
    "CYCLE",
    "UNREACHABLE_NODE",
    #: a node type the catalog has never heard of = an invented type
    "UNKNOWN_NODE_TYPE",
}

#: Expected under the retired/legacy policy — not structural failures.
LEGACY_CODES = {
    "UNAVAILABLE_NODE",
    "MISSING_REQUIRED_INPUT",
    "INVALID_OPTION_VALUE",
    "UNKNOWN_OUTPUT_PATH",
    "MISSING_TOP_LEVEL",
    "DUPLICATE_POSITION",
    "REQUIREMENT_NOT_COVERED",
    "TRIGGER_HAS_INCOMING",
}


def pytest_generate_tests(metafunc):
    if "export_path" in metafunc.fixturenames:
        metafunc.parametrize(
            "export_path", EXPORTS, ids=[p.name for p in EXPORTS]
        )


@pytest.mark.skipif(not EXPORTS, reason="no reference exports shipped")
class TestReferenceExports:
    def test_exports_exist(self, export_path):
        assert export_path.is_file()

    def test_structure_is_clean(self, export_path, catalog, settings):
        payload = json.loads(export_path.read_text(encoding="utf-8"))
        result = validate_workflow(payload, catalog, settings)
        structural = [e.code for e in result.errors if e.code in STRUCTURAL_CODES]
        assert structural == [], (
            f"{export_path.name}: structural errors {structural} "
            f"({[e.message for e in result.errors if e.code in STRUCTURAL_CODES]})"
        )

    def test_every_node_type_exists_in_the_catalog(
        self, export_path, catalog, settings
    ):
        """No invented node types — retired ones are allowed, unknown ones are not."""
        payload = json.loads(export_path.read_text(encoding="utf-8"))
        known = {
            e["nodeTypeId"]
            for e in catalog.all_nodes(include_unavailable=True)
        }
        used = {n.get("nodeTypeId") for n in payload.get("nodes", [])}
        assert used <= known, f"{export_path.name}: unknown {used - known}"

    def test_edges_reference_existing_nodes(self, export_path, catalog, settings):
        payload = json.loads(export_path.read_text(encoding="utf-8"))
        keys = {n.get("key") for n in payload.get("nodes", [])}
        for edge in payload.get("edges", []):
            assert edge.get("from") in keys, edge
            if edge.get("to") is not None:  # an explicit terminal branch is legal
                assert edge.get("to") in keys, edge

    def test_has_exactly_one_trigger_root(self, export_path, catalog, settings):
        payload = json.loads(export_path.read_text(encoding="utf-8"))
        triggers = [
            n
            for n in payload.get("nodes", [])
            if (catalog.get(str(n.get("nodeTypeId"))) or {}).get("isTrigger")
        ]
        assert len(triggers) == 1, [t.get("key") for t in triggers]

    def test_graph_is_acyclic_and_rooted(self, export_path, catalog, settings):
        payload = json.loads(export_path.read_text(encoding="utf-8"))
        result = validate_workflow(payload, catalog, settings)
        codes = {e.code for e in result.errors}
        assert "CYCLE" not in codes
        assert "NO_ROOT" not in codes

    def test_legacy_findings_are_reported_not_hidden(
        self, export_path, catalog, settings
    ):
        """Anything that is not a structural error must be a known legacy code."""
        payload = json.loads(export_path.read_text(encoding="utf-8"))
        result = validate_workflow(payload, catalog, settings)
        other = {
            e.code
            for e in result.errors
            if e.code not in STRUCTURAL_CODES | LEGACY_CODES
        }
        assert other == set(), f"{export_path.name}: unexpected errors {other}"

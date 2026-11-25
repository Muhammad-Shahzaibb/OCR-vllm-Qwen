"""Stage 4 — Node configuration generator.

The LLM proposes values; Python decides what is *allowed*.  Every key emitted is
checked against the catalog entry of that node type, every value is coerced to the
declared ``kind``, every ``select``/``segmented`` value is checked against the
declared options, secrets are never fabricated, and mode-conditional keys are
dropped when their mode is inactive.

Reference-workflow keys that the catalog does not declare (the exported files
carry a superset, e.g. ``maxEmails`` on the Outlook trigger) are **not** emitted
unless ``WORKFLOW_ALLOW_EXAMPLE_KEYS=true`` — the catalog is authoritative per
the feature spec, and the observed keys are kept behind one switch.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..catalog.manager import NodeCatalogManager
from ..config import WorkflowSettings
from ..llm import StructuredLLM, StructuredOutputError
from ..prompts import CONFIGURATOR_SYSTEM, build_configurator_user
from ..schemas import (
    ConfigurationBatch,
    DataFlowPlan,
    NodeConfigProposal,
    PlanNode,
    WorkflowPlan,
)

logger = logging.getLogger(__name__)

REF_PATTERN = re.compile(r"\{\{node:([A-Za-z0-9_.\-]+)/([^{}]*)\}\}")

#: Auth schemes and the credential fields each one needs (from the catalog text).
_AUTH_FIELDS = {
    "api": ["apiKeyName", "apiKeyValue", "apiKeyLocation"],
    "basic": ["basicUser", "basicPassword"],
    "oauth": [
        "oauthTokenUrl",
        "oauthClientId",
        "oauthClientSecret",
        "oauthScope",
    ],
}
_SECRET_INPUTS = {"apiKeyValue", "basicPassword", "oauthClientSecret"}


@dataclass
class ConfigurationOutcome:
    configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    used_llm: bool = True
    example_key_uses: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


async def configure_nodes(
    llm: StructuredLLM,
    plan: WorkflowPlan,
    catalog: NodeCatalogManager,
    settings: WorkflowSettings,
    *,
    prompt: str,
    request_id: str | None = None,
    max_tokens: int | None = None,
    dataflow: DataFlowPlan | None = None,
) -> ConfigurationOutcome:
    """Build one config per node.

    Precedence, highest first:

    1. ``PlanNode.preset_config`` — deterministic and catalog-backed (the
       resolver's operation option, condition rules, input type).  Merged into
       the raw proposal *before* sanitisation so the model can never override
       it and every value still goes through catalog coercion.
    2. data-flow **path selectors** (``rowsFrom``, ``approvalFields``)
       — the reference convention is a bare output path, so these overwrite a
       proposal that used the wrong syntax.
    3. the LLM proposal (sanitised: catalog keys only, coerced kinds, secrets
       never fabricated).
    4. data-flow **soft fills** — only for inputs the proposal left empty.
    5. deterministic catalog fallbacks (trigger schedule keys, ``schema``).
    """
    request_id = request_id or f"configure-{uuid.uuid4().hex[:8]}"
    entries = [_full_entry(catalog.get(n.node_type_id), settings) for n in plan.nodes]
    mappings = available_mappings(plan, catalog)
    #: The hard-coded shortcut is a rollback-path convenience only; the
    #: abstract pipeline must stay honest about what the model produced.
    direct_pattern = settings.planner_mode == "direct" and _is_outlook_pdf_to_excel(prompt)

    proposals: dict[str, NodeConfigProposal] = {}
    outcome = ConfigurationOutcome()
    try:
        if direct_pattern:
            # Node types and all required settings for this pattern are known
            # from the active catalog; avoid a second model call and apply the
            # catalog-backed configuration below.
            outcome.used_llm = False
        else:
            batch = await llm.complete_json(
                ConfigurationBatch,
                CONFIGURATOR_SYSTEM,
                build_configurator_user(plan, [e for e in entries if e], mappings, prompt,
                                        settings.configurator_prompt_budget),
                request_id=request_id,
                max_tokens=max_tokens,
            )
            proposals = {p.step_id: p for p in batch.nodes}
            outcome.unresolved.extend(batch.unresolved_requirements)
            outcome.issues.extend(f"assumption: {a}" for a in batch.assumptions)
    except StructuredOutputError as exc:
        outcome.used_llm = False
        outcome.issues.append(f"Configuration LLM output unusable ({exc}); defaults only.")
        logger.warning("Configurator structured output failed: %s", exc)

    for plan_node in plan.nodes:
        entry = catalog.get(plan_node.node_type_id)
        if entry is None:
            continue
        proposal = proposals.get(plan_node.id)
        raw = dict(proposal.config) if proposal else {}
        # 1. deterministic preset beats whatever the model proposed
        for key, value in (plan_node.preset_config or {}).items():
            if value not in (None, "", [], {}):
                raw[key] = value
        config, missing, notes = sanitize_config(
            entry, raw, plan_node, plan, catalog, settings, outcome
        )
        # For this well-defined workflow pattern, use deterministic catalog-backed
        # configuration instead of trusting Qwen to invent keys or omit binary flow.
        if direct_pattern:
            _apply_outlook_pdf_excel_config(plan_node.node_type_id, config)
        # 2./4. deterministic data-flow wiring
        _apply_data_flow(config, dataflow, plan_node, entry, notes)
        # 5. deterministic fallbacks for keys nothing else filled
        _apply_catalog_fallbacks(config, entry, notes)
        outcome.configs[plan_node.id] = config
        for key in missing:
            if key in config:
                continue
            line = f"Node '{plan_node.name or plan_node.node_type_id}' ({plan_node.id}) needs a value for '{key}'."
            if line not in outcome.unresolved:
                outcome.unresolved.append(line)
        outcome.issues.extend(
            f"{plan_node.id}: {n}" for n in notes
        )

    # condition `left` values and any other mapping are checked here as well
    outcome.issues.extend(validate_mappings(plan, catalog, outcome.configs))
    return outcome


def _apply_data_flow(
    config: dict[str, Any],
    dataflow: DataFlowPlan | None,
    plan_node: PlanNode,
    entry: dict[str, Any],
    notes: list[str],
) -> None:
    """Apply the deterministic pairing: selectors overwrite, the rest fills gaps."""
    if dataflow is None or not dataflow.mappings:
        return
    # local import: ``dataflow`` imports this module's helpers
    from .dataflow import _uses_bare_path, reference_value

    declared = {
        i.get("key")
        for i in entry.get("inputs") or []
        if isinstance(i, dict) and i.get("key")
    }
    inactive = inactive_keys(entry, config)
    for mapping in dataflow.for_step(plan_node.id):
        key = mapping.input_key
        if key not in declared or key in inactive:
            continue
        value = reference_value(mapping)
        if key in config and not _uses_bare_path(key):
            continue  # proposal wins for ordinary inputs
        if config.get(key) == value:
            continue
        replaced = config.get(key)
        config[key] = value
        notes.append(
            f"'{key}' wired to {value} ({mapping.basis or 'deterministic data-flow'})."
            + (f" Replaced proposal '{replaced}'." if replaced not in (None, "") else "")
        )


_TRIGGER_POLL_DEFAULTS = {
    "pollHour": "9",
    "pollMinute": "5",
    "pollDayOfMonth": "1",
    "cron": "0 9 * * *",
}


def _apply_catalog_fallbacks(
    config: dict[str, Any], entry: dict[str, Any], notes: list[str]
) -> None:
    """Deterministic last-resort values taken from the catalog, never invented."""
    inputs = [i for i in entry.get("inputs") or [] if isinstance(i, dict)]
    by_key = {i.get("key"): i for i in inputs}

    if entry.get("isTrigger"):
        for key, value in _TRIGGER_POLL_DEFAULTS.items():
            if key in config or key not in by_key:
                continue
            default = by_key[key].get("defaultValue")
            config[key] = default if isinstance(default, str) and default.strip() else value
            notes.append(f"'{key}' defaulted to '{config[key]}' (schedule default).")
        # an "Every X minutes" interval of 0 would never fire
        if (
            str(config.get("pollMode") or "") == "Every X minutes"
            and str(config.get("pollMinute") or "").strip() in ("", "0")
        ):
            config["pollMinute"] = "5"
            notes.append("pollMinute defaulted to 5: an interval of 0 minutes would never fire.")

    if entry.get("nodeTypeId") == "muntq.ai-operation" and "schema" in by_key:
        spec = by_key["schema"]
        if not config.get("schema") and spec.get("required") and spec.get("kind") == "code":
            default = spec.get("defaultValue")
            if isinstance(default, str) and default.strip():
                config["schema"] = default
                notes.append("'schema' filled from the catalog's own default for Extract.")



# --------------------------------------------------------------------------- #
# catalog helpers
# --------------------------------------------------------------------------- #


def _full_entry(node: dict[str, Any] | None, settings: WorkflowSettings) -> dict[str, Any] | None:
    if not node:
        return None
    example_keys = load_example_config_keys(settings)
    declared = {
        i.get("key")
        for i in (node.get("inputs") or [])
        if isinstance(i, dict)
    }
    extra = sorted(example_keys.get(node.get("nodeTypeId"), set()) - declared)
    return {
        "nodeTypeId": node.get("nodeTypeId"),
        "inputs": node.get("inputs") or [],
        "outputs": node.get("outputs") or [],
        "poll": node.get("poll"),
        "branching": node.get("branching"),
        "example_only_keys": extra if settings.allow_example_keys else [],
    }


def load_example_config_keys(settings: WorkflowSettings) -> dict[str, set[str]]:
    """Config keys observed in the app's own exported workflows.

    Used only when ``allow_example_keys`` is enabled; never invented here.
    """
    directory = Path(settings.examples_dir)
    result: dict[str, set[str]] = {}
    if not directory.exists():
        return result
    for path in directory.glob("*.muntq.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for node in data.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            type_id = node.get("nodeTypeId")
            cfg = node.get("config")
            if type_id and isinstance(cfg, dict):
                result.setdefault(type_id, set()).update(cfg.keys())
    return result


def available_mappings(plan: WorkflowPlan, catalog: NodeCatalogManager) -> dict[str, list[str]]:
    """For every step: the reference tokens its ancestors can supply."""
    order = topological_order(plan)
    position = {sid: i for i, sid in enumerate(order)}
    ancestors: dict[str, set[str]] = {}
    for sid in order:
        node = plan.node(sid)
        if node is None:
            continue
        preds = _predecessors(plan, sid)
        acc: set[str] = set()
        for p in preds:
            acc.add(p)
            acc |= ancestors.get(p, set())
        ancestors[sid] = acc

    out: dict[str, list[str]] = {}
    for sid in order:
        node = plan.node(sid)
        if node is None:
            continue
        tokens: list[str] = []
        for anc in sorted(ancestors[sid], key=lambda x: position.get(x, 0)):
            src = plan.node(anc)
            if src is None:
                continue
            paths = catalog.output_paths(src.node_type_id)
            if paths:
                tokens.extend(f"{{{{node:{anc}/{p}}}}}" for p in paths)
            else:
                tokens.append("{{node:<field-from-%s>}}" % anc)
        out[sid] = tokens
    return out


def _predecessors(plan: WorkflowPlan, node_id: str) -> list[str]:
    preds: list[str] = []
    for node in plan.nodes:
        if node_id in node.next or any(r.next == node_id for r in node.routes):
            preds.append(node.id)
    return preds


def topological_order(plan: WorkflowPlan) -> list[str]:
    """Kahn ordering seeded from the trigger; falls back to declaration order."""
    indeg = {n.id: 0 for n in plan.nodes}
    adj: dict[str, list[str]] = {n.id: [] for n in plan.nodes}
    for node in plan.nodes:
        for target in list(node.next) + [r.next for r in node.routes]:
            if target in indeg:
                adj[node.id].append(target)
                indeg[target] += 1
    starts = [n.id for n in plan.nodes if n.role == "trigger"] or [
        n.id for n in plan.nodes if indeg[n.id] == 0
    ]
    queue = list(starts)
    seen: list[str] = []
    while queue:
        cur = queue.pop(0)
        if cur in seen:
            continue
        seen.append(cur)
        for nxt in adj.get(cur, []):
            indeg[nxt] -= 1
            if indeg[nxt] <= 0 and nxt not in seen:
                queue.append(nxt)
    for node in plan.nodes:  # any node left out of the DAG order (defensive)
        if node.id not in seen:
            seen.append(node.id)
    return seen


# --------------------------------------------------------------------------- #
# sanitisation
# --------------------------------------------------------------------------- #


def sanitize_config(
    entry: dict[str, Any],
    raw: dict[str, Any],
    plan_node: PlanNode,
    plan: WorkflowPlan,
    catalog: NodeCatalogManager,
    settings: WorkflowSettings,
    outcome: ConfigurationOutcome,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Return ``(config, missing_required_keys, notes)`` for one node."""
    notes: list[str] = []
    inputs: list[dict[str, Any]] = entry.get("inputs") or []
    declared = [i for i in inputs if isinstance(i, dict) and i.get("key")]
    example_keys = set(entry.get("example_only_keys") or [])
    allowed = {i["key"] for i in declared} | example_keys

    # 1. keep only supported keys (in catalog order)
    config: dict[str, Any] = {}
    ordered_keys = [i["key"] for i in declared] + sorted(example_keys)
    for key in ordered_keys:
        if key not in raw:
            continue
        spec = next((i for i in declared if i.get("key") == key), None)
        if spec is None:
            notes.append(
                f"key '{key}' is not declared by the catalog but appears in the reference "
                "workflows; emitted because allow_example_keys is on."
            )
            outcome.example_key_uses.append(f"{plan_node.node_type_id}.{key}")
            value = raw[key]
        else:
            value = coerce(spec, raw[key])
            if value is _INVALID:
                notes.append(f"dropped '{key}': value does not match the declared kind/options.")
                continue
        if value is None:
            continue
        config[key] = value

    dropped = [k for k in raw if k not in allowed]
    if dropped:
        notes.append("ignored unsupported keys: " + ", ".join(sorted(dropped)))

    # 2. mode-conditional keys
    config = _drop_inactive_mode_keys(entry, config, notes)

    # 3. example-derived behaviour (gmail gates sender/subject behind `filters`)
    config = _apply_example_rules(entry, config, notes)

    # 4. secrets are never generated
    for key in list(config):
        if key in _SECRET_INPUTS:
            config.pop(key)
            notes.append(f"secret field '{key}' left unset — must be provided by a connection.")

    # 5. required keys that are still missing
    missing: list[str] = []
    inapplicable = inactive_keys(entry, config)
    for spec in declared:
        key = spec["key"]
        if not spec.get("required") or key in config:
            continue
        if key in _SECRET_INPUTS or key in inapplicable:
            continue
        default = spec.get("defaultValue")
        if isinstance(default, str) and default.strip():
            coerced = coerce(spec, default)
            config[key] = default if coerced is _INVALID else coerced
            continue
        if default not in (None, "", [], {}) and not isinstance(default, str):
            config[key] = default
            continue
        options = parse_options(spec.get("options"))
        if options and spec.get("kind") in ("select", "segmented", "multiSelect"):
            config[key] = options[0]
            notes.append(f"'{key}' defaulted to the first catalog option '{options[0]}'.")
            continue
        if key == "pollMode" and options:
            config[key] = options[0]
            continue
        missing.append(key)

    # 6. map the plan step ids in `{{node:…}}` references is done by the builder,
    #    but make sure a mapped value never points at a step that does not exist.
    config = _guard_references(config, plan, notes)
    return config, missing, notes


_INVALID = object()


def parse_options(value: Any) -> list[str]:
    """Catalog options are a `` · `` (U+00B7) separated string."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value.strip():
        return [p.strip() for p in value.split("·") if p.strip()]
    return []


def coerce(spec: dict[str, Any], value: Any) -> Any:
    """Coerce a proposed value to the declared kind; ``_INVALID`` when impossible."""
    kind = spec.get("kind") or "text"
    if value is None:
        return None
    options = parse_options(spec.get("options"))

    if kind in ("checkbox", "toggle"):
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false", "yes", "no", "on", "off"):
            return value.strip().lower() in ("true", "yes", "on")
        if isinstance(value, (int, float)):
            return bool(value)
        return _INVALID

    if kind in ("select", "segmented"):
        if not isinstance(value, (str, int, float)):
            return _INVALID
        text = str(value).strip()
        if not options:
            return text
        if text in options:
            return text
        lowered = {o.lower(): o for o in options}
        if text.lower() in lowered:
            return lowered[text.lower()]
        return _INVALID

    if kind == "multiSelect":
        values = value if isinstance(value, list) else [value]
        out: list[str] = []
        for item in values:
            text = str(item).strip()
            if options and text not in options:
                lowered = {o.lower(): o for o in options}
                if text.lower() not in lowered:
                    return _INVALID
                text = lowered[text.lower()]
            out.append(text)
        return out

    if kind == "collection":
        if isinstance(value, list):
            out = [str(v) for v in value]
            if options:
                bad = [v for v in out if v not in options]
                if bad:
                    return _INVALID
            return out
        if isinstance(value, str) and value.strip():
            return [v.strip() for v in value.split(",") if v.strip()]
        return _INVALID if value not in (None, "") else None

    if kind == "kv":
        if isinstance(value, dict):
            return [[str(k), "" if v is None else v] for k, v in value.items()]
        if isinstance(value, list):
            rows: list[list[Any]] = []
            for item in value:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    rows.append([str(item[0]), item[1]])
                elif isinstance(item, dict):
                    keys = list(item.keys())
                    if len(keys) == 1:
                        rows.append([str(keys[0]), item[keys[0]]])
                else:
                    return _INVALID
            return rows
        return _INVALID

    if kind == "conditions":
        if not isinstance(value, list):
            return _INVALID
        groups: list[dict[str, Any]] = []
        for group in value:
            if not isinstance(group, dict):
                return _INVALID
            rules = group.get("rules")
            if not isinstance(rules, list) or not rules:
                return _INVALID
            clean_rules = []
            for rule in rules:
                if not isinstance(rule, dict) or "left" not in rule:
                    return _INVALID
                clean_rules.append(
                    {
                        "left": str(rule.get("left", "")),
                        "operator": str(rule.get("operator", "is equal to")),
                        "right": rule.get("right", ""),
                    }
                )
            groups.append(
                {"combinator": str(group.get("combinator") or "AND"), "rules": clean_rules}
            )
        return groups

    if kind == "routerRules":
        if not isinstance(value, list):
            return _INVALID
        return value

    if kind == "formFields":
        if not isinstance(value, list):
            return _INVALID
        fields = []
        for f in value:
            if not isinstance(f, dict) or not f.get("label"):
                return _INVALID
            fields.append(
                {
                    "label": str(f.get("label")),
                    "type": str(f.get("type") or "Short text"),
                    "required": bool(f.get("required", False)),
                    "options": list(f.get("options") or []),
                }
            )
        return fields

    # text / textarea / code
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return _INVALID


# --------------------------------------------------------------------------- #
# mode-conditional keys (documented rules, derived from catalog descriptions)
# --------------------------------------------------------------------------- #


def inactive_keys(entry: dict[str, Any], config: dict[str, Any]) -> set[str]:
    """Keys that do not apply given the node's current mode/auth/method.

    Derived from the catalog's own descriptions (see ``TECHNICAL_ASSESSMENT.md``
    §4) — the catalog marks many inputs ``required`` while showing them only for a
    particular mode.
    """
    type_id = entry.get("nodeTypeId") or ""
    inactive: set[str] = set()

    if type_id in ("http.api-request", "http.csrf-token"):
        auth = str(config.get("authType") or "None")
        if auth == "None":
            inactive |= {f for group in _AUTH_FIELDS.values() for f in group}
        elif auth == "API Key":
            inactive |= set(_AUTH_FIELDS["basic"] + _AUTH_FIELDS["oauth"])
        elif auth == "Basic Auth":
            inactive |= set(_AUTH_FIELDS["api"] + _AUTH_FIELDS["oauth"])
        elif auth == "OAuth":
            inactive |= set(_AUTH_FIELDS["api"] + _AUTH_FIELDS["basic"])
        if type_id == "http.api-request" and str(config.get("method") or "GET") in ("GET", "DELETE"):
            inactive |= {"bodyMode", "bodyRows", "rawBody"}

    if type_id == "transform.convert-to-file":
        if str(config.get("operation") or "Convert to CSV") != "Base64 to file":
            inactive.add("base64")

    if type_id == "muntq.ai-operation":
        operation = str(config.get("operation") or "Extract")
        if operation == "Classify":
            inactive.add("schema")
        elif operation == "Summarize":
            inactive |= {"schema", "labels"}

    if type_id == "flow-control.wait" and str(config.get("mode") or "For a time") == "For a time":
        inactive |= {"notifyEmail", "approvalMessage", "approvalFields"}

    if type_id == "code.run-code":
        if str(config.get("language") or "JavaScript") == "JavaScript":
            inactive.add("pythonCode")
        elif str(config.get("language")) == "Python":
            inactive.add("jsCode")
    return inactive


def missing_required_inputs(entry: dict[str, Any], config: dict[str, Any]) -> list[str]:
    """Required inputs that are in play for this mode but not populated."""
    skip = inactive_keys(entry, config)
    missing: list[str] = []
    for spec in entry.get("inputs") or []:
        if not isinstance(spec, dict) or not spec.get("required"):
            continue
        key = spec.get("key")
        if not key or key in skip or key in _SECRET_INPUTS:
            continue
        if key in config:
            continue
        missing.append(key)
    return missing


def _drop_inactive_mode_keys(
    entry: dict[str, Any], config: dict[str, Any], notes: list[str]
) -> dict[str, Any]:
    for key in sorted(inactive_keys(entry, config)):
        if key in config:
            config.pop(key)
            notes.append(f"omitted '{key}': not applicable to the selected mode.")
    return config


def _apply_example_rules(
    entry: dict[str, Any], config: dict[str, Any], notes: list[str]
) -> dict[str, Any]:
    """Behaviour evidenced by the reference workflows (not guessed)."""
    if entry.get("nodeTypeId") == "gmail.message-received" and "filters" in (
        i.get("key") for i in entry.get("inputs") or []
    ):
        active = [k for k in ("subject", "sender") if str(config.get(k) or "").strip()]
        if active and not config.get("filters"):
            config["filters"] = active
            notes.append(f"set filters={active} so the matching fields apply (reference workflow behaviour).")
    return config


# --------------------------------------------------------------------------- #
# reference guarding
# --------------------------------------------------------------------------- #


def _guard_references(config: dict[str, Any], plan: WorkflowPlan, notes: list[str]) -> dict[str, Any]:
    """A mapped token must name a step that exists in this plan."""

    def walk(value: Any) -> Any:
        if isinstance(value, str):
            def repl(match: re.Match[str]) -> str:
                step = match.group(1)
                if plan.node(step) is None:
                    notes.append(f"reference to unknown plan step '{step}' removed.")
                    return ""
                return match.group(0)

            return REF_PATTERN.sub(repl, value)
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        return value

    return {k: walk(v) for k, v in config.items()}


def validate_mappings(
    plan: WorkflowPlan, catalog: NodeCatalogManager, configs: dict[str, dict[str, Any]]
) -> list[str]:
    """Deterministic data-mapping checks (issues, not failures)."""
    issues: list[str] = []
    order = topological_order(plan)
    position = {sid: i for i, sid in enumerate(order)}

    for plan_node in plan.nodes:
        config = configs.get(plan_node.id) or {}
        for token, step, path in iter_references(config):
            source = plan.node(step)
            if source is None:
                issues.append(
                    f"{plan_node.id}: mapping '{token}' points at a step that does not exist."
                )
                continue
            if position.get(step, -1) >= position.get(plan_node.id, 1 << 30):
                issues.append(
                    f"{plan_node.id}: mapping '{token}' comes from a step that does not run before it."
                )
                continue
            declared = catalog.output_paths(source.node_type_id)
            if declared and not _path_supported(path, declared):
                issues.append(
                    f"{plan_node.id}: mapping '{token}' uses a path the catalog does not declare "
                    f"for {source.node_type_id} (declared: {', '.join(declared)})."
                )
    return issues


def _path_supported(path: str, declared: Sequence[str]) -> bool:
    path = (path or "").strip().strip("/")
    if not path:
        return False
    for candidate in declared:
        if path == candidate or path.startswith(candidate + "."):
            return True
    return False


def iter_references(value: Any) -> Iterable[tuple[str, str, str]]:
    """Yield ``(token, step_id, path)`` for every mapped reference."""
    if isinstance(value, str):
        for match in REF_PATTERN.finditer(value):
            yield match.group(0), match.group(1), match.group(2)
    elif isinstance(value, list):
        for item in value:
            yield from iter_references(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_references(item)



def _is_outlook_pdf_to_excel(prompt: str) -> bool:
    text = (prompt or "").lower()
    return (
        "outlook" in text
        and "pdf" in text
        and "excel" in text
        and any(word in text for word in ("attachment", "attached"))
    )


def _apply_outlook_pdf_excel_config(node_type_id: str, config: dict[str, Any]) -> None:
    """Apply only settings whose semantics are explicitly present in the catalog."""
    if node_type_id == "outlook.message-received":
        config.update({
            "hasAttachment": True,
            "downloadAttachments": True,
        })
    elif node_type_id == "muntq.ai-operation":
        schema = {
            "type": "object",
            "properties": {
                "document_type": {"type": "string"},
                "key_value_pairs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "value": {"type": "string"},
                        },
                        "required": ["field", "value"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["document_type", "key_value_pairs"],
            "additionalProperties": False,
        }
        config.update({
            "inputType": "File",
            "operation": "Extract",
            "schema": json.dumps(schema, ensure_ascii=False, indent=2),
            "instruction": (
                "Read the attached PDF and extract its meaningful information. "
                "Return one key_value_pairs entry for each field/value pair found. "
                "Preserve values accurately, do not invent missing values, and use "
                "the document_type field to identify the kind of document."
            ),
            "failOnInvalidSchema": True,
        })
    elif node_type_id == "transform.convert-to-file":
        config.update({
            "operation": "Convert to Excel",
            "fileName": "extracted-data",
            "rowsFrom": "Intelligence.Result.key_value_pairs",
            "columns": [["Field", "field"], ["Value", "value"]],
            "outputField": "data",
        })

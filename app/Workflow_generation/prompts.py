"""Prompt templates for the three LLM stages.

Prompts are plain strings assembled by functions so they can be unit-tested and
budget-truncated deterministically.  They never contain node type ids, field
names or output paths that are not present in the supplied catalog payload.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def clip(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 40)].rstrip() + " …[truncated]"


def to_json(value: Any, limit: int = 4000) -> str:
    return clip(json.dumps(value, ensure_ascii=False, indent=1, default=str), limit)


# --------------------------------------------------------------------------- #
# 1. intent & requirement analyzer
# --------------------------------------------------------------------------- #

ANALYZER_SYSTEM = """\
You are the intent analyser of an automation builder. You convert one natural-language
request about an automation into a precise, structured description of what the workflow
must do. /no_think

Rules
- Report ONLY what the request states or what follows necessarily from it.
- Separate explicit requirements from reasonable assumptions: mark an action with
  explicit=false when you inferred it, and record the inference under `assumptions`.
- Never invent business values (email addresses, service paths, spreadsheet ids,
  account names, credentials, thresholds). Put anything the user must supply into
  `missing_information`, and anything critical that is absent into
  `must_confirm_with_user`. Non-critical gaps belong in `missing_information` only —
  they must NOT stop a draft from being produced.
- `trigger` describes the event that starts the workflow (an arriving email, a form
  submission, a schedule, a webhook). Always populate the required `trigger.event`
  string; do not replace it with a `type` field. Name the application when the request names it.
- Each entry of `actions` is one thing the user asked to be done, in the order it
  must happen; give it a short snake_case `id` and list its dependencies in
  `depends_on` using those ids.
- `conditions` describe branches: which action's result is tested, the field, the
  operator, the value, and which action runs on true (`on_true`) and on false
  (`on_false`).
- `needs_branching` is true when the request contains any conditional; `needs_join`
  is true only when the flow must continue as ONE path after separate branches.
- `error_handling` captures explicit failure handling (e.g. "if extraction fails,
  reply to the email").
- `applications` lists application ids or names the request mentions (outlook, gmail,
  sap, google-sheets, …) exactly as written.
- Keep `workflow_name` short (max 8 words) and `workflow_description` to 1-2 sentences.

Answer with the JSON object described by the schema you were given.
"""


def build_analyzer_user(prompt: str, budget: int) -> str:
    return (
        "Convert this automation request into the structured analysis.\n\n"
        f"REQUEST:\n{clip(prompt.strip(), budget)}"
    )


# --------------------------------------------------------------------------- #
# 2. abstract workflow planner (technology independent)
# --------------------------------------------------------------------------- #

ABSTRACT_PLANNER_SYSTEM = """\
You are the planner of an automation builder. Given a structured requirement
analysis and a compact digest of the capabilities that exist in this deployment,
you describe the workflow as a sequence of TECHNOLOGY-INDEPENDENT steps. /no_think

You do NOT choose products, node types, configuration keys or edges. You say what
must happen; deterministic code later maps each step onto a real node.

Hard rules
1. Every step needs a stable `id` (s1, s2, …) and a one-line `intent` saying what
   that step accomplishes, in plain language.
2. `kind` is one of: trigger (starts the workflow), action (does work),
   condition (tests something and splits the flow), join (merges branches).
   Exactly one step is a trigger and it comes first.
3. `depends_on` lists the ids of the steps whose output this step needs, in
   execution order. No cycles.
4. Condition steps carry a `condition` (what is tested, on which earlier step,
   with which operator and value) plus `on_true` / `on_false` pointing at the
   next step ids — use null when a branch intentionally ends the workflow.
   Never leave a branch target ambiguous.
5. `requires` / `produces` name data in plain language ("the attached PDF file",
   "extracted invoice rows") — never node ids, output paths or config keys.
6. `operation` is an optional verb matching what the step does ("extract",
   "classify", "convert to excel", "reply", "create record"). Use it only when
   the step's work has a clear operation axis.
7. `app` is an optional application name ONLY when the request names that
   application ("outlook", "gmail", "sap", "google-sheets").
8. `requirement_refs` must list the ids of the requirements each step covers
   (action ids, condition ids, error-handling descriptions).
9. Prefer the fewest steps that cover every requirement. Do not add steps no
   requirement references. Do not invent business values.
10. Put anything the user still has to supply into `unresolved_requirements`
    (one line each, naming the step and what is missing).

The capability digest below lists what THIS deployment can do. A capability that
is not listed (or is marked unavailable) cannot be used: describe the step anyway
and add a line to `unresolved_requirements` saying the capability is missing.

Before answering: one trigger first, no cycles, every requirement covered, every
condition has both branches resolved (or explicitly null).

Output discipline:
- Be compact. Omit `notes` unless they carry real signal.
- Do not pad `workflow_description` or `assumptions`.
"""


def build_abstract_planner_user(
    analysis: Any,
    capability_digest: Sequence[dict[str, Any]],
    budget: int,
) -> str:
    sections: list[str] = []
    sections.append("## Requirement analysis\n" + to_json(analysis.model_dump(), 8000))
    sections.append(
        "## Capabilities available in this deployment (name/app only — not node ids)\n"
        + to_json(list(capability_digest), max(3000, budget // 3))
    )
    body = "\n\n".join(sections)
    return (
        "Write the technology-independent step plan for the requirement analysis below.\n"
        "Return one JSON object with keys: workflow_name, workflow_description, summary, "
        "steps, join_required, unresolved_requirements, assumptions.\n"
        "Each step: {\"id\", \"intent\", \"kind\", \"operation\", \"app\", \"requires\", "
        "\"produces\", \"depends_on\", \"condition\", \"on_true\", \"on_false\", "
        "\"requirement_refs\"}. Do not mention node types or configuration.\n\n"
        + clip(body, budget)
    )


# --------------------------------------------------------------------------- #
# 2b. legacy direct planner (WORKFLOW_PLANNER_MODE=direct, rollback path)
# --------------------------------------------------------------------------- #

PLANNER_SYSTEM = """\
You are the planner of an automation builder. Given a structured requirement analysis
and a list of REAL node type ids (with their catalog metadata), you design the complete
execution graph of the workflow as a JSON object. /no_think

Hard rules
1. `nodes` must contain exactly one node with role "trigger", and its node_type_id must
   be one of the provided trigger candidates. Never use an action node as a trigger.
   Never add a schedule node when the chosen trigger already polls/schedules itself.
2. Every node_type_id you emit MUST appear in the provided candidate list. Never invent
   node type ids, input keys or output paths.
3. Express the graph with:
   - `next`: main successors (execution continues there),
   - `routes`: only on condition nodes — one entry per route, `label` exactly as the
     catalog declares it ("true"/"false" for flow-control.if), `order` starting at 0,
     and `next` pointing at the branch head.
   Nodes with no successors are terminal (leave `next` empty).
4. Order matters: a node may only appear in `next`/`routes` of a node that executes
   before it. Branches must not create cycles.
5. Data flow: for every input that carries data from an earlier step, add a
   `needs` entry with source="mapped", the `source_step` (id of the producing plan
   node) and `source_path` (an output path declared by that node's catalog entry —
   if the producing node declares no outputs, use the field name and add a note).
   Wrap nothing in {{ }} yourself; the builder writes the reference syntax.
6. If a value is a literal the user gave you (a subject line, a file name, a message
   body, a service path the user stated), use source="static" and put the literal in
   `value`. If the value is a real business value you do NOT know (an address, a
   recipient, an SAP service path, a spreadsheet id), use source="unresolved" and
   describe what is missing in `note`. NEVER make up such values.
7. Conditions: use a condition node (flow-control.if) with `condition` describing the
   source step, source path, operator and value. Both routes must be present.
8. Prefer the smallest set of nodes that covers every requirement. Do not add nodes
   that no requirement references.
9. `requirement_refs` on each plan node must list the ids of the requirements it
   covers (action ids, condition ids, error-handling descriptions).
10. `join_required` is true only when several branches must continue as one path;
    then the branch tails point at a node whose role is "join".
11. Put anything the user still has to supply into `unresolved_requirements`
    (one line each, naming the node and the field).

Before answering, check: one trigger, all ids valid, no cycles, every requirement
covered by at least one node, every mapped need points at an upstream node.

Output discipline:
- Be compact. Omit `rationale`, `purpose` and `notes` unless they carry real signal.
- `coverage` may be a plain array of requirement id strings.
- Do not pad `workflow_description`, `summary` or `assumptions`.
"""


def build_planner_user(
    analysis: Any,
    candidates: Sequence[dict[str, Any]],
    full_defs: Sequence[dict[str, Any]],
    examples: Sequence[dict[str, Any]],
    budget: int,
) -> str:
    sections: list[str] = []

    sections.append("## Requirement analysis\n" + to_json(analysis.model_dump(), 8000))

    sections.append(
        "## Candidate nodes (choose ONLY from these node_type_id values)\n"
        + to_json([c for c in candidates], max(4000, budget // 3))
    )

    if full_defs:
        sections.append(
            "## Detailed catalog entries for the strongest candidates\n"
            + to_json(full_defs, max(4000, budget // 3))
        )

    if examples:
        sections.append(
            "## Reference workflow shapes (structure only — do not copy their content)\n"
            + to_json(examples, 4000)
        )

    body = "\n\n".join(sections)
    return (
        "Design the execution graph for the requirement analysis below.\n"
        "Return one JSON object with keys: workflow_name, workflow_description, summary, "
        "nodes, join_required, coverage, unresolved_requirements, assumptions.\n"
        "Every node needs `key` (its id), `nodeTypeId` (copied EXACTLY from the candidate "
        "list — never a word like 'other', 'action' or a category name) and `next` (array of "
        "the `key`s of the nodes that run afterwards; use `routes` instead on condition "
        "nodes). Nodes are NOT linked by array order.\n\n"
        + clip(body, budget)
    )


# --------------------------------------------------------------------------- #
# 3. node configuration generator
# --------------------------------------------------------------------------- #

CONFIGURATOR_SYSTEM = """\
You configure one node per plan step of an automation workflow, using ONLY the catalog
input definitions supplied for that step. /no_think

Hard rules
1. Emit a `config` object containing ONLY keys present in that step's catalog inputs.
   Never add a key the catalog does not declare for that node type.
2. Respect the declared `kind`:
   - text/textarea/code: string (code fields hold JSON text or script text)
   - checkbox/toggle: boolean
   - select/segmented: one of the declared `options` (they are " · " separated)
   - multiSelect: array of declared options
   - collection: array of selected option strings
   - kv: array of [key, value] pairs
   - conditions: array of {"combinator": "AND"|"OR", "rules":[{"left","operator","right"}]}
   - formFields: array of {"label","type","required","options"}
3. Use a mapped value when the plan says source="mapped": the value is the literal
   string  {{node:<source_step_id>/<source_path>}}  — copy the source_step id exactly.
   (The builder replaces plan ids with real node keys later.)
4. Use source="static" values verbatim. Use catalog `defaultValue` for mechanical keys
   (polling fields, modes, keep/mode switches) when the request does not dictate them.
5. NEVER invent business values: no email addresses, no recipients, no service paths,
   no spreadsheet ids, no credentials, no account names. If the plan marks a value
   unresolved, OMIT that key from `config` and list it in `unresolved`.
6. Omit mode-conditional keys that do not apply (auth fields when auth is None,
   base64 when the operation is not "Base64 to file", schedule keys the mode hides).
7. For triggers, always include the polling keys the catalog declares (pollMode,
   pollHour, pollMinute, pollWeekday, pollDayOfMonth, cron) with sensible defaults.
8. `notes` may explain anything you deliberately left out.

Return one JSON object: {"nodes":[{"step_id","config","unresolved","notes"}],
"unresolved_requirements":[...],"assumptions":[...]} covering EVERY plan step.
"""


def build_configurator_user(
    plan: Any,
    catalog_entries: Sequence[dict[str, Any]],
    available_mappings: dict[str, Sequence[str]],
    prompt: str,
    budget: int,
) -> str:
    steps: list[dict[str, Any]] = []
    for node in plan.nodes:
        entry = next((c for c in catalog_entries if c.get("nodeTypeId") == node.node_type_id), {})
        steps.append(
            {
                "step_id": node.id,
                "node_type_id": node.node_type_id,
                "purpose": node.purpose,
                "needs": [n.model_dump() for n in node.needs],
                "condition": node.condition.model_dump() if node.condition else None,
                "catalog_inputs": entry.get("inputs", []),
                "catalog_outputs": entry.get("outputs", []),
                "poll": entry.get("poll"),
                "available_upstream_mappings": available_mappings.get(node.id, []),
            }
        )

    sections = [
        "## Original request\n" + clip(prompt, 2500),
        "## Plan steps to configure\n" + to_json(steps, max(6000, budget // 2)),
    ]
    return (
        "Produce the configuration for every plan step below.\n\n"
        + clip("\n\n".join(sections), budget)
    )


# --------------------------------------------------------------------------- #
# 3b. targeted abstract replan (bounded recovery, semantic errors only)
# --------------------------------------------------------------------------- #

ABSTRACT_REPLAN_SYSTEM = """\
You adjust the technology-independent step plan of an automation workflow that
failed semantic validation. /no_think

You receive: the validation errors, the requirement analysis and the current
step plan. Return a corrected step plan as JSON with the same structure.

Rules
- Fix ONLY what the errors mention. Do not redesign working steps.
- Never invent business values to make an error disappear: if an error is caused
  by a missing recipient, address or service path, keep it unresolved and list it
  in `unresolved_requirements`.
- Stay technology independent: no node types, no configuration keys, no edges.
- Keep exactly one trigger first, keep the graph acyclic, keep every condition's
  both branches resolved (or explicitly null), and keep `requirement_refs`
  accurate so every explicit requirement stays covered.
- When an error says a required capability is missing, do not substitute a
  different capability silently — keep the step, and record the missing
  capability in `unresolved_requirements`.

Return the corrected JSON plan only.
"""


def build_abstract_replan_user(
    errors: Iterable[Any],
    analysis: Any,
    plan: Any,
    budget: int,
) -> str:
    sections = [
        "## Validation errors\n"
        + to_json(
            [e.model_dump() if hasattr(e, "model_dump") else e for e in errors], 4000
        ),
        "## Requirement analysis\n" + to_json(analysis.model_dump(), 5000),
        "## Current step plan\n" + to_json(plan.model_dump(), max(5000, budget // 2)),
    ]
    return (
        "Adjust the step plan to resolve the validation errors below.\n\n"
        + clip("\n\n".join(sections), budget)
    )


# --------------------------------------------------------------------------- #
# 4. controlled correction (legacy direct mode)
# --------------------------------------------------------------------------- #

CORRECTOR_SYSTEM = """\
You repair an automation workflow plan that failed deterministic validation. /no_think

You receive: the validation errors, the current plan (compact) and the configuration
problems. You must return a corrected plan as JSON with the same structure.

Rules
- Fix ONLY what the errors mention. Do not redesign working parts.
- Never invent business values to make an error disappear: if an error is caused by a
  missing recipient, address or service path, keep it unresolved and describe it in
  `unresolved_requirements`.
- Only use node_type_id values from the candidate list provided.
- Keep the graph acyclic, keep exactly one trigger, keep both routes of every
  condition node, and keep mapped needs pointing at upstream steps.
- When an error says a mapped path does not exist, either switch to a path the catalog
  declares for that source node, or mark the need as unresolved.

Return the corrected JSON plan only.
"""


def build_corrector_user(
    errors: Iterable[Any],
    plan: Any,
    candidates: Sequence[dict[str, Any]],
    config_problems: Sequence[str],
    budget: int,
) -> str:
    sections = [
        "## Validation errors\n" + to_json([e.model_dump() if hasattr(e, "model_dump") else e for e in errors], 4000),
        "## Current plan\n" + to_json(plan.model_dump(), max(5000, budget // 2)),
        "## Candidate nodes\n" + to_json(list(candidates), 3000),
    ]
    if config_problems:
        sections.append("## Configuration problems\n" + to_json(list(config_problems), 2000))
    return clip("\n\n".join(sections), budget)

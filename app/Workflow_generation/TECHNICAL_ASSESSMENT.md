# Technical Assessment — AI Workflow Generation Backend

Produced before implementation, from inspection of:

- `app/Workflow_generation/node_catalog/nodes-catalog.json` (moved from `app/existing_workflows/`)
- `app/existing_workflows/*.muntq.json` (6 reference workflows)

---

## 1. Discovered workflow JSON schema (canonical)

All six reference files share the same top-level shape:

```jsonc
{
  "muntqScenario": 1,          // number, always 1 in every example
  "exportedAt": "…ISO-8601…",  // present in all six
  "name": "…",                 // workflow display name
  "credentialsRemoved": [],    // array; empty in every example
  "nodes": [ /* node objects */ ],
  "edges":  [ /* edge objects */ ]
}
```

### Node object

| field           | type                        | notes |
|-----------------|-----------------------------|-------|
| `key`           | string (unique)             | the node id used by edges and by `{{node:…}}` references |
| `nodeTypeId`    | string                      | must exist in the catalog |
| `position`      | number                      | **equals the index of the node inside `nodes[]`** (0-based) |
| `x`, `y`        | number or `null`            | canvas coordinates; `null` occurs in 2 of 6 examples |
| `laneKey`       | string                      | `"main"`, `"true"`, `"false"` |
| `note`          | `null` (or string)          | always `null` in examples |
| `name`          | string or `null`            | display name; `null` in a few auto-generated nodes |
| `config`        | object                      | node-type-specific configuration |
| `credentialType`| string or `null`            | e.g. `"microsoft-oauth"`, `"google-oauth"`, `"sap"` |

### Edge object

| field         | type                    | notes |
|---------------|-------------------------|-------|
| `from`        | string                  | source node `key` |
| `to`          | string                  | target node `key` |
| `kind`        | `"MAIN"` \| `"ROUTE"`   | |
| `routeLabel`  | string or `null`        | `"true"` / `"false"` on `ROUTE` edges; `null` on `MAIN` |
| `routeOrder`  | number or `null`        | `0` for the first route, `1` for the second; `null` on `MAIN` |
| `conditions`  | always `null`           | never populated in any example |

### Two edge/node formatting variants (difference between files)

1. **"Full" form** — `sales-order.muntq (1).json`, `vendor-onboarding-intake-and-review-old.muntq.json`:
   every node carries `laneKey`, `note`, `x`, `y`, `credentialType`, `name`; every edge carries
   `routeLabel`, `routeOrder`, `conditions` (explicitly `null` on `MAIN`).
2. **"Export" form** — `23…26-*.muntq.json`: null-valued keys are omitted on `MAIN` edges and on
   main-lane nodes (`laneKey`/`note` only present on branch nodes).

**Decision:** variant (1) is the canonical target for generation. It is the superset of (2) — every
key the app reads is present, with explicit `null`s where the export form omits them. The style is
configurable (`WORKFLOW_EDGE_STYLE=full|minimal`) in case the importer prefers variant (2).

---

## 2. Dynamic data references

The only reference syntax used by any example:

```
{{node:<source-node-key>/<output.path>[.<sub-path>]}}
```

Examples observed:

- `{{node:outlook-message-received-0yj9f6/Message.Id}}` — declared trigger output
- `{{node:extract-invoice/Intelligence.Result.route}}` — declared output + schema property
- `{{node:report/File.Rows}}` — declared output of a `transform.convert-to-file`
- `{{node:group/Rows}}` — field produced by a `code.run-code` script (source declares **no** outputs)
- `{{node:read/Intelligence.SchemaValid}}` — declared output of a `muntq.file-extract`

The catalog's own `outputNote` text documents a *different*, positional form (`{{N.Schedule.RunAt}}`,
N = 1-based position). **No example uses it**; it is documentation prose, not the runtime syntax.
Therefore `{{node:<key>/…}}` is treated as the canonical mapping syntax.

Implication for validation: when the source node declares outputs, a referenced path must be a
declared path (or a sub-path beneath one). When the source declares no outputs (pass-through nodes
such as `code.run-code`, `transform.edit-fields`, `sap.create-record`), the path cannot be verified
against the catalog — this is reported as a **warning**, not an error, because the catalog explicitly
says "Pass-through or unknown shape — map from a recorded run".

---

## 3. Branching and triggers

- **Trigger:** exactly one trigger node, first in the document, `nodeTypeId.isTrigger == true` in the
  catalog. Triggers in the catalog: `schedule.every`, `webhook.on-request`, `form.on-submit`,
  `gmail.message-received`, `outlook.message-received`. Trigger nodes carry their own polling keys
  (`pollMode`, `pollHour`, `pollMinute`, `pollWeekday`, `pollDayOfMonth`, `cron`) — the catalog says
  "Its Poll Times ARE the scenario schedule", so **no separate schedule node is added**.
- **Condition:** `flow-control.if`, config key `conditions`:
  ```json
  [{"combinator":"AND","rules":[{"left":"{{node:…/…}}","operator":"is equal to","right":"…"}]}]
  ```
  It emits two routes labelled `true` and `false` (`branching.routeLabels` in the catalog), carried
  as `ROUTE` edges with `routeOrder` 0 and 1.
- **Router:** `flow-control.router` also declares branching but has `labelPrefix: "Route"` and **no**
  `routeLabels`, and no example uses it. Route labels for a router are therefore not derivable from
  the provided files → router support is isolated behind a configurable label source
  (`WORKFLOW_ROUTER_ROUTE_LABELS`) and defaults to the two labels observed, with a warning.
- **Lanes:** branch nodes carry `laneKey: "true"` / `"false"`; main-line nodes `"main"`.
- **Joins:** `flow-control.join` exists (`joins: true`) and its catalog body states its wiring "is not
  drawn and cannot be edited: it is recomputed whenever the graph changes". **No example uses a
  join**, so its stored edge representation cannot be established. Implementation: connect the last
  node of each branch to the join with `MAIN` edges and the join onward with a `MAIN` edge, and emit
  a validation **warning** flagging this as inferred rather than demonstrated.

---

## 4. Catalog structure

Top level: `generatedAt`, `source`, `appCount`, `nodeCount`, `activeNodeCount`, `retiredNodeCount`,
`apps[]`, `nodes[]` (46 nodes / 16 apps).

Per node:

```
nodeTypeId, appId, appName, category, name, description, summary, body, runMessage,
isTrigger, retired, implemented, credentialType, credentialOptional, managedService,
publicAddress, itemIdentity[], poll{...} | null, joins: bool, branching{...} | null,
wizard{role,prompt,order,group?,fields?} | null,
inputs[]{ key,label,kind,required,secret,defaultValue?,options?,hint?,description },
outputs[]{ path, description }, outputNote, sampleOutput
```

Observed input `kind` values (13): `text, textarea, select, segmented, multiSelect, checkbox,
toggle, collection, kv, code, conditions, routerRules, formFields`.
`options` is a **` · ` (U+00B7) separated string** — not an array.

Availability flags: 9 nodes `retired`, 5 nodes `implemented=false` (all `doc-ai.*`).
Some nodes are `retired=true, implemented=true` (e.g. `muntq.file-extract`, `muntq.text-parse`) —
these appear in the reference workflows but are retired in the current catalog.

**Required-flag caveat:** several inputs are marked `required: true` but are only shown for a
particular mode (e.g. `http.api-request.apiKeyName` only when `authType = "API Key"`,
`transform.convert-to-file.base64` only when `operation = "Base64 to file"`). The reference workflows
also omit catalog-required keys (e.g. `outlook.reply-email.emailType`). Required therefore means
"required when the owning mode is active", not "must always appear in `config`". Mode-conditional
keys are handled by an explicit, documented rule table rather than a blanket rule.

---

## 5. Assumptions that cannot be established from the provided files

| # | Uncertainty | Handling |
|---|-------------|----------|
| 1 | Exact edge/node key set the importer requires (variant 1 vs 2) | configurable `WORKFLOW_EDGE_STYLE`, default `full` |
| 2 | Router route labels | configurable `WORKFLOW_ROUTER_ROUTE_LABELS`, warning emitted |
| 3 | Stored edge representation of `flow-control.join` | inferred `MAIN` edges + warning |
| 4 | Whether an unresolved *required* input must be `""`, `null` or absent | absent + recorded in `unresolved_requirements` (examples never contain empty required business values) |
| 5 | Credential ids | never present in any example — only `credentialType` is emitted; required connections are reported in `unresolved_requirements` |
| 6 | Whether `position` must equal array index in all cases | enforced (holds in 6/6 examples) |
| 7 | Behaviour of `x/y = null` | always emit numeric coordinates; layout never affects execution |
| 8 | Approval/wait node placeholder emails | never invented; reported as unresolved |

Nothing above is spread through the codebase: each lives in one settings field or one warning code.

---

## 6. Architectural summary

```
prompt → validate → analyze (LLM/Pydantic) → retrieve (hybrid, deterministic)
       → plan (LLM/Pydantic) → configure (LLM + deterministic sanitisation)
       → build graph (deterministic) → emit JSON (deterministic)
       → validate (deterministic) ⇄ correct (bounded, deterministic-first)
       → response
```

LangGraph `StateGraph` with the shared state below; exactly one LLM call per semantic stage, and no
stage may emit the final workflow document directly — only the deterministic builder does.

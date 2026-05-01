# ADR-2026-04-11 — Meta-Mode Positioning and RTX 3090 Load Verification

- **Status:** Accepted
- **Date:** 2026-04-11
- **Project:** `llama-cpp-turboquant-guide`
- **Session artifact:** `SESSION-NOTES-2026-04-11.md`
- **Decision owners:** Session operator + agent
- **Scope:**
  1. Claude Code plugin strategy for `meta-skills`
  2. Operational verification of suspected GPU load on host `.90`

---

## 1. Context

Two separate but session-relevant questions were raised.

### A. Claude Code / meta-skills
The question was whether the `meta-skills` Claude Code plugin can introduce a real additional mode comparable to Claude Code's built-in planning mode, bypass mode, or normal/default operation.

The product motivation was valid: planning mode is useful not only because of permissions, but because it creates a visible planning phase and a cleaner, reviewable plan output.

### B. RTX 3090 load on `.90`
There was concern that the agent might still have an active test or workload running on the RTX 3090 host at `joe@10.40.10.90`.

---

## 2. Questions to answer

### A. Plugin / Meta-Mode
1. Is a new native Claude Code mode technically possible from a plugin?
2. Is it supported / allowed in the documented Claude Code extension model?
3. Is it a good product decision for `meta-skills`?

### B. GPU load
1. Did this agent session leave any managed background job running?
2. Was the 3090 under active compute load at inspection time?
3. Did the observed GPU state look like a test/inference workload or normal desktop residency?

---

## 3. Sources

### Local sources
- `C:/Users/Legion/Documents/phantom-ai/meta-skills/README.md`
- `C:/Users/Legion/Documents/phantom-ai/meta-skills/.claude-plugin/plugin.json`
- `C:/Users/Legion/Documents/phantom-ai/meta-skills/hooks/hooks.json`
- `C:/Users/Legion/Documents/phantom-ai/meta-skills/plans/hardening-refactoring-2026-04-10.md`

### Official Claude Code documentation
Retrieved through Context7 from:
- `/anthropics/claude-code`
- `/websites/code_claude`

Topics checked:
- permission modes
- plan mode
- plugin manifest / plugins reference
- output styles
- hooks

### Runtime checks
- local managed process inventory via `bg_shell list`
- remote host inspection via SSH to `joe@10.40.10.90`
- remote GPU inspection via `nvidia-smi`

---

## 4. Evidence

## A. Plugin capability evidence

### Local plugin structure
The current `meta-skills` plugin is structured around:
- slash commands
- hooks
- skills
- agents
- scripts
- statusline/session lifecycle behavior

The local manifest file contains metadata only and exposes no custom mode registration.

The local hook configuration currently defines:
- `UserPromptSubmit`
- `Stop`
- `PostToolUse`

### Official plugin model
Official Claude Code plugin references document plugin manifest support for fields such as:
- `commands`
- `agents`
- `skills`
- `hooks`
- `mcpServers`
- `outputStyles`
- `lspServers`

No documented manifest field for custom host-level modes was found.

### Official permission mode model
Official Claude Code docs describe a fixed set of permission modes, including:
- `default`
- `acceptEdits`
- `plan`
- `auto`
- `dontAsk`
- `bypassPermissions`

No documentation was found for plugin-defined native permission modes.

### Output style support
Official Claude Code docs explicitly support custom output styles. This is relevant because the desired user-facing benefit is partly presentation and workflow clarity, not only permissions.

---

## B. GPU verification evidence

### Managed background jobs
Local process manager result:
- `No background processes.`

### Remote host confirmation
Remote hostname returned:
- `DESKTOP-K4P6IFT`

### GPU state at inspection time
`nvidia-smi` returned:
- GPU utilization: `0%`
- Memory utilization: `13%`
- Memory used: `1447 MiB`
- Total VRAM: `24576 MiB`
- Temperature: `42 C`
- Power draw: `42.63 W`

### GPU process list
Visible processes were Windows/UI-oriented, including:
- `dwm.exe`
- `explorer.exe`
- `SearchHost.exe`
- `StartMenuExperienceHost.exe`
- `msedgewebview2.exe`
- `Taskmgr.exe`
- `Docker Desktop.exe`
- shell/application frame hosts

No obvious inference/training process was visible.

---

## 5. Decision

## Decision A — Meta-Mode positioning
`meta-skills` will **not** be treated as capable of introducing a new native Claude Code mode.

Instead, the correct strategic model is:
- a **plugin-defined workflow**
- optionally branded as **Meta Mode** in product language
- implemented through supported extension points such as:
  - commands
  - hooks
  - plugin-local settings/state
  - statusline indicators
  - output styles

### Approved framing
Acceptable framing:
- “Meta Planning workflow”
- “Meta Session Layer”
- “Meta Mode” as a plugin UX concept

### Rejected framing
Rejected framing:
- “new native Claude Code mode”
- “new permission mode next to plan/auto/bypassPermissions”

## Decision B — 3090 load attribution
The suspected 3090 load was **not** caused by an active managed test from this session.

At inspection time the GPU was not under active compute load. Observed VRAM usage was consistent with normal Windows desktop/UI process residency.

---

## 6. Rationale

## A. Why the Meta-Mode decision was made
The desired user benefit is real:
- clearer phase separation
- visible planning state
- cleaner plan output
- more confidence that the agent is operating inside a structured workflow

However, these benefits do **not** require a native Claude Code mode.

Trying to present a plugin workflow as if it were a host-native permission mode would create ambiguity around:
- edit rights
- command execution rights
- approval semantics
- safety expectations

That would be misleading UX and weak governance.

Using supported plugin primitives achieves the product goal without inventing unsupported host behavior.

## B. Why the GPU conclusion was made
The available runtime evidence points away from an active agent workload:
- no agent-managed background jobs existed
- GPU utilization was `0%`
- no obvious compute process appeared in the GPU process list
- listed processes matched normal desktop/UI activity

Therefore the safest conclusion is that no active test from this session was running on the 3090.

---

## 7. Consequences

## A. For the meta-skills plugin
If a future implementation is desired, the plugin should use a supported architecture such as:
- `/meta-plan` or `/meta-start`
- plugin-local state: `mode: meta-plan`
- statusline badge: `META`, `META-PLAN`, or equivalent
- deterministic plan output structure
- optional output style for presentation consistency

### Recommended output skeleton
A future Meta Planning output should likely include:
- goal
- context
- assumptions
- risks
- proposed steps
- excluded scope
- approval / handoff point

## B. For operations on `.90`
No remediation was required from this session.

If future unexplained GPU residency needs deeper attribution, use a second-level investigation:
1. identify top VRAM holders in more detail
2. inspect CPU-side inference processes
3. sample `nvidia-smi` over time for spikes
4. inspect scheduled services or user-launched workloads

---

## 8. Risks and caveats

### Documentation drift risk
An internal `meta-skills` planning document claims `SessionStart` is not a valid Claude Code hook event. Current official references surfaced during this session indicate broader hook support, including `SessionStart` / `SessionEnd` in some contexts.

This indicates at least one of the following:
- internal documentation drift
- product-version drift
- a context-specific difference that is not documented clearly in internal notes

This should be re-verified before any hook-model redesign.

### Observation timing caveat
The GPU conclusion reflects the observed state at inspection time. It rules out an active visible compute load at that moment, but not a transient workload that started and ended before inspection.

---

## 9. Non-decisions

This session did **not**:
- modify the `meta-skills` plugin
- add a new command or output style
- update ERPNext
- update open-notebook
- push to GitHub
- change any remote system state on `.90`

---

## 10. Final summary

### Final conclusion — plugin strategy
A new native Claude Code mode is **not** supported by the documented plugin model. A plugin-defined Meta workflow is both feasible and strategically sound.

### Final conclusion — GPU operations
No managed job from this session remained active, and the 3090 was not under active compute load when checked.

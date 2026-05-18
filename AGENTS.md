# AGENTS.md

## Mission
Keep analysis token-efficient and evidence-based. Avoid broad scans unless explicitly requested.

## Startup Order (Always)
1. Read `context/BATON.md`.
2. Read `context/PROJECT_CONTEXT.md`.
3. Read `context/EVIDENCE_MAP.md`.
4. Read `context/DEEP_CONTEXT.md` only when detailed context is needed.
5. Read only minimal files required for the active claim/task.

## Scope Discipline
- Do not run full-repo scans by default.
- Do not scan source files for resume or improvement tasks unless explicitly asked.
- Treat README and resume bullets as unverified until tied to code/tests/bench evidence.
- Mark uncertainty as `TODO/VERIFY`.

## Editing Rules
- Prefer small diffs.
- Do not modify source/tests/build config unless explicitly requested.
- Keep docs concise and high-signal.

## Validation Rules
- Run narrow tests only when explicitly asked.
- Avoid expensive builds/benchmarks unless explicitly requested.
- Do not commit and do not use `git add .`.

## Handoff Rules
After meaningful work:
1. Append `AUDIT_LOG.md`.
2. Update `NEXT_TASK.md`.
3. Update `CLAIMS_MATRIX.md` with evidence deltas.
4. Update `RESUME_CLAIMS.md` only if resume wording or evidence status changes.

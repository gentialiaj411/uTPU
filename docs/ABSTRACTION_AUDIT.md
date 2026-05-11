# Abstraction Boundary Audit

Date: 2026-05-10
Scope: IR/lowering/planner and immediate orchestration paths in `firmware/host`.

## Query Results

Target-leakage grep counts requested:

- `#ifdef CUDA`: **0** matches (Python/RTL scope reviewed).
- `if target ==`: **0** matches.
- `if backend ==`: **0** matches.
- Backend-name string compares in IR/lowering/planner layers: **3** matches.
  - `firmware/host/backend_lowering.py:31` (`if n == "utpu":`)
  - `firmware/host/backend_lowering.py:33` (`if n == "cuda":`)
  - `firmware/host/program_loader.py:551` (`if self.backend_name.strip().lower() == "cuda":`)

## Target-Specific Concept Leakage Outside Backend-Specific Files

Backend-specific files treated as allowed:
- `firmware/host/lowering_blocked_fc_utpu.py`
- `firmware/host/cuda_blocked_fc_backend.py`

Everything below is outside those files.

### 1) Backend switch in runtime path
- Location: `firmware/host/program_loader.py:551`
- Leak: Direct `"cuda"` branch in `ProgramLoader.execute_fc_layer_blocked`.
- Impact: Runtime path knows backend names and CUDA execution object directly.
- Decision: **TODO**
- TODO: Move execution dispatch behind `BackendLowerer`/backend runtime protocol so `ProgramLoader` does not branch on backend string.

### 2) Backend factory string routing
- Location: `firmware/host/backend_lowering.py:29-35`
- Leak: `create_backend_lowerer(name)` maps `"utpu"` / `"cuda"` via string compares.
- Impact: Centralized coupling point by backend names.
- Decision: **Justified**
- Justification: This is the intended seam where target selection happens; one explicit switch is acceptable and preferable to distributed string checks.

### 3) Hardware terms in shared abstraction layer
- Location: `firmware/host/compiler_abstractions.py`
  - `shared_mem_bytes` and `warp_or_lane_width` fields in `TargetDesc`.
  - Comment references BRAM mapping semantics.
- Leak: Shared abstraction names include target-oriented capacity terms.
- Impact: Mild semantic bias toward GPU/uTPU memory models.
- Decision: **Justified (for now)**
- Justification: These fields are metadata only and do not force backend behavior in schedule builder.
- TODO: If third backend diverges strongly, split `TargetDesc` into strict generic core + backend extension metadata.

### 4) BRAM wording in generic runtime docs/comments
- Location: `firmware/host/program_loader.py:82` comment mentions instruction BRAM.
- Leak: Terminology leak in orchestration layer documentation.
- Impact: Naming, not logic.
- Decision: **Justified**
- Justification: Program loader is currently uTPU-centric for upload/start/fetch path; comment accurately describes deployed path.

## Clean Areas

- `firmware/host/lowering_types.py` is fully backend-agnostic request typing.
- `firmware/host/compiler_abstractions.py::build_blocked_fc_schedule` contains no backend-name branch logic.
- `firmware/host/lowering_blocked_fc_utpu.py` owns uTPU ISA details cleanly.
- `firmware/host/cuda_blocked_fc_backend.py` owns CUDA/PTX/NVRTC details cleanly.

## Summary

Abstraction boundaries are mostly clean with one meaningful leakage to fix: the loader-level runtime backend branch (`firmware/host/program_loader.py:551`). The remaining target mentions are either the explicit factory seam or metadata/comment-level coupling with limited behavioral risk.

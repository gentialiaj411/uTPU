# Abstraction Boundary Audit

Date: 2026-05-10
Scope: IR/lowering/planner and immediate orchestration paths in `firmware/host`.

## Methodology

### Path boundaries used in this audit

- IR/planner/lowering/shared-orchestration scope (audited):
  - `firmware/host/compiler_abstractions.py`
  - `firmware/host/lowering_types.py`
  - `firmware/host/backend_lowering.py`
  - `firmware/host/program_loader.py`
  - `firmware/host/lowering_blocked_fc_utpu.py`
- Backend-specific implementation files (allowed target-specific content):
  - `firmware/host/lowering_blocked_fc_utpu.py`
  - `firmware/host/cuda_blocked_fc_backend.py`

### Re-runnable commands and raw counts

Run from repo root (`uTPU`):

1. `rg -n '#ifdef CUDA' firmware/host/compiler_abstractions.py firmware/host/lowering_types.py firmware/host/backend_lowering.py firmware/host/program_loader.py firmware/host/lowering_blocked_fc_utpu.py`
- raw count: `0`

2. `rg -n 'if\s+target\s*==' firmware/host/compiler_abstractions.py firmware/host/lowering_types.py firmware/host/backend_lowering.py firmware/host/program_loader.py firmware/host/lowering_blocked_fc_utpu.py`
- raw count: `0`

3. `rg -n 'if\s+backend\s*==' firmware/host/compiler_abstractions.py firmware/host/lowering_types.py firmware/host/backend_lowering.py firmware/host/program_loader.py firmware/host/lowering_blocked_fc_utpu.py`
- raw count: `0`

4. `rg -n 'if n ==' firmware/host/backend_lowering.py`
- raw count: `2`
- matches:
  - `firmware/host/backend_lowering.py:31` `if n == "utpu":`
  - `firmware/host/backend_lowering.py:33` `if n == "cuda":`

5. `Select-String -Path firmware/host/program_loader.py -Pattern 'backend\.strip\(\)\.lower\(\) == "cuda"'`
- raw count: `1`
- match:
  - `firmware/host/program_loader.py:45` `... if backend.strip().lower() == "cuda" else None`

6. `Select-String -Path firmware/host/program_loader.py -Pattern 'backend_name\.strip\(\)\.lower\(\) == "cuda"'`
- raw count: `1`
- match:
  - `firmware/host/program_loader.py:551` `if self.backend_name.strip().lower() == "cuda":`

7. `rg -n 'warp|shared_mem|BRAM|PTX|ptx' firmware/host/compiler_abstractions.py firmware/host/lowering_types.py firmware/host/backend_lowering.py firmware/host/program_loader.py firmware/host/lowering_blocked_fc_utpu.py`
- raw count: `8`
- matches:
  - `firmware/host/program_loader.py:82` (BRAM wording in comment)
  - `firmware/host/compiler_abstractions.py:17` (`shared_mem_bytes`)
  - `firmware/host/compiler_abstractions.py:19` (`warp_or_lane_width`)
  - `firmware/host/compiler_abstractions.py:51` (BRAM mapping comment)
  - `firmware/host/compiler_abstractions.py:80` (`shared_mem_bytes=1024`)
  - `firmware/host/compiler_abstractions.py:82` (`warp_or_lane_width=array_size`)
  - `firmware/host/compiler_abstractions.py:91` (`shared_mem_bytes=48*1024`)
  - `firmware/host/compiler_abstractions.py:93` (`warp_or_lane_width=32`)

## Query Results

Requested pattern counts:

- `#ifdef CUDA`: **0**
- `if target ==`: **0**
- `if backend ==`: **0**
- Backend-name string compares in IR/lowering/planner layers: **4**
  - `firmware/host/backend_lowering.py:31`
  - `firmware/host/backend_lowering.py:33`
  - `firmware/host/program_loader.py:45`
  - `firmware/host/program_loader.py:551`

## Target-Specific Concept Leakage Outside Backend-Specific Files

### 1) Backend switch in runtime path (execute-time)
- Location: `firmware/host/program_loader.py:551`
- Leak: Direct `"cuda"` branch in `ProgramLoader.execute_fc_layer_blocked`.
- Impact: Runtime path knows backend names and CUDA execution object directly.
- Decision: **TODO**
- TODO: Move execution dispatch behind `BackendLowerer`/backend runtime protocol so `ProgramLoader` does not branch on backend string.

### 2) Backend switch in runtime construction (init-time)
- Location: `firmware/host/program_loader.py:45`
- Leak: Constructor creates `CUDABlockedFCExecutor` via explicit `backend.strip().lower() == "cuda"` check.
- Impact: Backend-specific construction logic sits in shared orchestration class.
- Decision: **TODO**
- TODO: Move backend runtime object construction behind backend registry/factory, not `ProgramLoader` string checks.

### 3) Backend factory string routing
- Location: `firmware/host/backend_lowering.py:29-35`
- Leak: `create_backend_lowerer(name)` maps `"utpu"` / `"cuda"` via string compares.
- Impact: Centralized coupling point by backend names.
- Decision: **Justified**
- Justification: This is the intended seam where target selection happens; one explicit switch is acceptable and preferable to distributed checks.

### 4) Hardware terms in shared abstraction layer
- Location: `firmware/host/compiler_abstractions.py`
  - `shared_mem_bytes` and `warp_or_lane_width` fields in `TargetDesc`.
  - Comment references BRAM mapping semantics.
- Leak: Shared abstraction names include target-oriented capacity terms.
- Impact: Mild semantic bias toward GPU/uTPU memory models.
- Decision: **Justified (for now)**
- Justification: These fields are metadata only and do not force backend behavior in schedule builder.
- TODO: If third backend diverges strongly, split `TargetDesc` into strict generic core + backend extension metadata.

### 5) BRAM wording in generic runtime docs/comments
- Location: `firmware/host/program_loader.py:82`
- Leak: Terminology leak in orchestration layer documentation.
- Impact: Naming, not logic.
- Decision: **Justified**
- Justification: Program loader is currently uTPU-centric for upload/start/fetch path; comment accurately describes deployed path.

## New Leaks Surfaced In This Rerun

- Added one additional backend-name string compare not listed in the prior audit: `firmware/host/program_loader.py:45` (constructor-time CUDA executor selection).

## Clean Areas

- `firmware/host/lowering_types.py` remains backend-agnostic request typing.
- `firmware/host/compiler_abstractions.py::build_blocked_fc_schedule` contains no backend-name branch logic.
- `firmware/host/lowering_blocked_fc_utpu.py` owns uTPU ISA details cleanly.
- `firmware/host/cuda_blocked_fc_backend.py` owns CUDA/PTX/NVRTC details cleanly.

## Summary

Abstraction boundaries are mostly clean with two meaningful loader-level leakages to fix: backend-specific construction and execute-time branching in `ProgramLoader`. The remaining target mentions are either the explicit factory seam or metadata/comment-level coupling with limited behavioral risk.

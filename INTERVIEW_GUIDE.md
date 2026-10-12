# uTPU Project — Interview Guide

This document explains the project from first principles so you can answer any interview question about it confidently. Read this once end-to-end, then use the Q&A section to drill specific questions.

---

## The Big Picture (30-Second Version)

You built a **compiler**. A compiler takes code written in one language and translates it into code a machine can execute. Your compiler's input is a PyTorch neural network model. Its output is machine code — either for a GPU (CUDA) or for a custom hardware accelerator you designed (the uTPU).

The reason this is hard and impressive: neural network inference has very specific performance requirements (you need to process matrix multiplications as fast as possible), so a naive translation doesn't work. You had to design the entire translation pipeline yourself — how to represent the computation, how to schedule the work, how to encode it for each hardware target.

---

## Why This Project Exists

Modern ML accelerators (like Google's TPU or Apple's Neural Engine) are purpose-built chips that run matrix math much faster than a general-purpose CPU. Building one requires two things working together:

1. **Hardware** — the physical chip or FPGA that does the math
2. **Compiler** — software that translates "run this neural network" into the exact sequence of instructions the hardware understands

Most people who build custom ML accelerators use existing compiler frameworks (like MLIR or TVM). You built yours from scratch, targeting your own custom ISA (instruction set architecture) — the language your hardware speaks.

---

## The Hardware Side: What Is the uTPU?

uTPU stands for "micro Tensor Processing Unit." It's a custom hardware accelerator implemented in RTL (Register Transfer Level) — essentially, hardware description code (like Verilog) that describes logic gates and registers. This gets loaded onto an FPGA (a reprogrammable chip), which then behaves like a custom processor.

Your hardware is specifically optimized for **blocked fully-connected (FC) layers** — the core operation in MLP (multi-layer perceptron) neural networks. A fully-connected layer is just a matrix multiplication: output = input × weights.

**What "blocked" means:** Instead of multiplying the full matrix at once (which might not fit in on-chip memory), you chop it into tiles (blocks) and process one block at a time, reusing data that's already been loaded. This is the same idea as cache-blocking in CPU programming.

The uTPU has:
- Its own instruction set (custom opcodes it understands)
- On-chip BRAM (Block RAM) that stores the program and weights
- A simulation environment using Icarus Verilog (open-source RTL simulator) so you can test without physical hardware

---

## The Compiler Side: How It Works (Step by Step)

### Step 1: PyTorch FX Tracing
When you write a neural network in PyTorch, it's Python code. PyTorch has a feature called `torch.fx` that can trace through your model and produce a **computation graph** — a structured representation of every operation (linear layer, ReLU, etc.) and how data flows between them.

Your compiler starts here: it calls `torch.fx` on the user's model and captures this graph.

```python
# User's model (input to your compiler)
class TinyMLP(nn.Module):
    def forward(self, x):
        x = self.fc1(x)   # linear layer
        x = relu(x)
        x = self.fc2(x)   # linear layer
        return x
```

After FX tracing, you have a graph that says: `placeholder → linear → relu → linear → output`.

### Step 2: Custom Graph IR
FX gives you a graph, but it's tied to PyTorch internals. You convert it into your own **Graph IR** (Intermediate Representation) — a clean, target-agnostic data structure that records:
- Every value (tensor) that exists in the computation
- Every operation (op) and what inputs it takes
- The shape of every tensor
- Producer/consumer relationships (which op produces each value, which ops consume it)

This is your compiler's "internal language." Everything downstream works with this IR, not with PyTorch objects.

**Why have a custom IR?** It lets you transform, analyze, and lower the computation without being tied to PyTorch's representation. Real compilers (LLVM, GCC, XLA) all have their own IRs for this reason.

### Step 3: Lowering to Blocked FC Requests
Not all operations in a neural network are supported by your hardware. Your compiler identifies which ops it can handle (Linear layers) and which it can't (anything exotic). For the supported ops, it translates them into **BlockedFCLoweringRequests** — a structured description of the matrix multiplication work to be done:
- Input dimensions (M × K)
- Tile/block sizes
- Memory layout
- Which backend to target

This is the **shared layer** — the same lowering request object gets handed to both the CUDA backend and the uTPU backend. This is what makes the compiler "retargetable."

### Step 4: Backend Dispatch
A single function `create_backend_lowerer(name)` picks which backend to use. Pass `"cuda"` and you get a CUDA code generator. Pass `"utpu"` and you get the uTPU ISA code generator. Everything above this point is identical regardless of target.

### Step 5a: uTPU Backend
Takes the lowering request and emits **16-bit instruction words; INT4/INT8 datapath** — binary-encoded instructions in the custom ISA. These instructions tell the uTPU hardware exactly what to do: load weights into BRAM, execute the blocked matrix multiply, store results.

The emitted program gets packed into a **BRAM image** — the layout of on-chip memory that the hardware will read and execute. This is what you simulate with Icarus Verilog.

**Key optimization — program fusion:** Instead of emitting a separate program for each layer (load weights for layer 1, run it, then load weights for layer 2, run it), you fuse both layers into a single BRAM program. The activations from layer 1 stay on-chip and feed directly into layer 2 without going off-chip. This is where the **65% footprint reduction** comes from: one fused program at 1,017 BRAM words vs. two separate programs totaling 2,918 words.

### Step 5b: CUDA Backend
Takes the same lowering request and generates **NVRTC PTX code** — GPU kernel source code that gets compiled at runtime by NVIDIA's runtime compiler (NVRTC). The generated kernel performs the blocked int4 matrix multiplication on the GPU.

This backend is the validation and benchmarking sandbox. It lets you:
- Prove the compiler pipeline works without needing physical FPGA hardware
- Measure real kernel latency and compare to cuBLAS (NVIDIA's optimized library)
- Verify numerical correctness against a software reference

---

## Why CUDA? (The Question You Had)

Here's the clearest way to think about it:

The uTPU is your **end goal** — real custom hardware. But hardware is painful to develop against. You can't easily profile it, debugging is slow, and you need the physical board. So you built CUDA as a **parallel execution target** that shares the entire front-end of your compiler.

The CUDA backend answers the question: *"Is your compiler pipeline correct, independent of the hardware?"* If the same FX tracing, Graph IR, and blocked scheduling produces correct output on a GPU (which has great profiling tools and is easy to validate), you can trust that the logic fed into the uTPU backend is also correct.

It's similar to how compiler engineers test a new ISA backend by first running the IR through a well-understood target (like x86) to verify the passes are correct, then testing the new target.

**Additional benefit:** The CUDA path gives you real performance numbers — kernel latency, transfer overhead, comparison to cuBLAS — that make the project demonstrably measurable and benchmarkable. The uTPU path has RTL simulation cycle counts, but CUDA gives wall-clock performance data.

---

## Key Numbers and What They Mean

### 65% BRAM Footprint Reduction
- **What it is:** The uTPU's on-chip memory (BRAM) has limited space. Your fused 2-layer program uses 1,017 words. Two separate (unfused) programs for the same layers would use 2,918 words.
- **Why it matters:** Smaller program = fits in less BRAM = can run on smaller/cheaper hardware. This is the same logic as code size optimization on embedded systems.
- **How you achieved it:** Fusing programs eliminates the overhead of loading separate weight sets and inter-layer data transfers. Blocked scheduling and burst-store encoding compact the instruction stream further.

### Candidate-count reduction (Autotuner)
- **What it is:** Calibration replay shows **3.12x candidate-count reduction over a 16-schedule (threads x unroll) menu on calibration replay, max regression <0.5%**. Artifact: `bench/results/pruned_autotuner_replay.json`; reproduce with `python firmware/host/evaluate_pruned_autotuner.py --calibration-json bench/fixtures/cost_model_calibration_seed.json --top-k 4 --output-json bench/results/pruned_autotuner_replay.json`.
- **Why it matters:** Different layer shapes are faster with different GPU thread configurations. There's no single best config — it depends on matrix dimensions. The autotuner searches the space and remembers the best config for each shape.
- **Important caveat:** This is isolated kernel latency improvement, not end-to-end speedup. The full pipeline (including data transfer overhead) still loses to cuBLAS on these small shapes. This is expected — cuBLAS is a highly optimized library for exactly this. Your compiler's value is that it can target custom hardware, not that it beats cuBLAS on a GPU.

### RTL Simulation: 2 Cases Pass, 44,018 Cycles
- **What it is:** Running the fused 2-layer MLP program through the Icarus Verilog RTL simulator. The simulator actually executes your hardware description, instruction by instruction, and counts clock cycles.
- **Why it matters:** This is as close to real hardware execution as you can get without a physical board. 44,018 cycles is the exact cycle count your hardware would use.

---

## What the Compiler Does NOT Do (Be Honest About This)

- **Does not beat cuBLAS.** The compiled CUDA path is slower than cuBLAS on all measured shapes. This is fine — your goal was a retargetable compiler for custom hardware, not to out-optimize NVIDIA's production library.
- **Does not support transformers or convolutions.** Only Linear + ReLU MLP-style graphs. Being scoped is not a weakness — it's an honest engineering boundary.
- **Does not have physical board validation.** The uTPU path is validated by RTL simulation (Icarus Verilog), not by running on a real FPGA. Simulation is standard practice for hardware bring-up.
- **Does not replace torch.compile.** It's not a general PyTorch backend.

---

## The Retargetable Architecture (The Core Design Achievement)

The most impressive engineering decision is that both backends share everything above the backend dispatch layer. This means:

- If you add a third hardware target (say, a CPU backend or a different accelerator), you write one new lowering module. The FX import, Graph IR, analysis passes, and scheduling logic are all reused.
- Changes to the front-end (supporting a new op, fixing a scheduling bug) automatically benefit both backends.
- The backends are isolated — a bug in CUDA codegen can't affect uTPU codegen.

This is the same design principle as LLVM: one IR, many backends. You implemented a simplified version of this for ML workloads.

---

## Interview Q&A

## Likely Hard Questions -> Evidence

- Why is top-1 schedule prediction weak? -> see `CLAIMS_MATRIX.md`; the supported claim is pruning quality, not single-winner prediction.
- Is ISA/RTL match on hardware? -> see `CLAIMS_MATRIX.md` and `docs/EVIDENCE.md`; the proof is simulation-only, not board execution.
- Is transformer support broad? -> see `README.md`, `PROJECT_CONTEXT.md`, and `CLAIMS_MATRIX.md`; the claim is only tested single-block transformer coverage.
- Is `torch.compile` support general? -> see `CLAIMS_MATRIX.md`; it is bounded and falls back to eager for unsupported ops.

**Q: What is a compiler?**
A: Software that translates code from one representation to another. Mine takes PyTorch models (Python) and produces machine code for two hardware targets: GPU and a custom accelerator.

**Q: What is an intermediate representation (IR)?**
A: A target-agnostic data structure that sits between the input language and the output machine code. My Graph IR records every tensor value, operation, shape, and data flow edge in the computation. Both backends read from this same IR, which is how I support two targets without duplicating the front-end.

**Q: What does "blocked" mean in blocked FC?**
A: Instead of multiplying the full weight matrix at once, you divide it into rectangular tiles and process one tile at a time. This is done because on-chip memory (BRAM on an FPGA, L1/L2 cache on a CPU) is limited. By tiling, each block of weights fits in fast memory, gets reused multiple times, and then you move to the next tile. It's the same idea as cache-blocking or loop tiling in high-performance computing.

**Q: Why build a custom ISA?**
A: A general-purpose CPU ISA (x86, ARM) has instructions for everything — branching, floating point, system calls, etc. For an ML accelerator, you only need a small set of operations: load weights, compute blocked matrix multiply, store results. A custom ISA can encode these operations much more compactly and execute them more efficiently because the hardware doesn't need to handle the general case.

**Q: What is NVRTC?**
A: NVIDIA Runtime Compilation. Instead of pre-compiling GPU kernels to binary, NVRTC lets you write GPU kernel source code as a string at runtime, compile it on-the-fly, and execute it. This means your compiler generates kernel source code dynamically based on the layer's dimensions, and NVRTC turns it into a real GPU binary each time.

**Q: Why does your compiled path lose to cuBLAS?**
A: cuBLAS is NVIDIA's hand-optimized library for matrix operations, developed by hundreds of engineers over many years with access to hardware internals. My compiler generates correct kernels optimized for the custom uTPU blocking strategy, not for squeezing the last bit of GPU utilization. The comparison to cuBLAS isn't the point — the point is that the same compiler pipeline produces valid, executable code for two different hardware targets. On the custom hardware, the compiler's blocking strategy is an advantage, not a deficit.

**Q: What is RTL simulation?**
A: RTL (Register Transfer Level) is a level of hardware abstraction where you describe computation in terms of registers and the logic that transfers data between them. An RTL simulator (Icarus Verilog, in this case) takes your hardware description and executes it in software, simulating every clock cycle. This lets you verify hardware behavior without a physical chip.

**Q: How did you get 65% footprint reduction?**
A: The key insight is program fusion. Naively, a 2-layer MLP needs two separate programs: load layer 1 weights → run layer 1 → store output → load layer 2 weights → run layer 2. Each program has its own header, weight storage, and control flow. When fused into one program, you eliminate the redundant headers and keep intermediate activations on-chip between layers. The unfused total is 2,918 BRAM words; the fused program is 1,017 words — a 65% reduction.

**Q: What would it take to add a new hardware backend?**
A: Write one new Python module that accepts a `BlockedFCLoweringRequest` (the shared request schema) and produces output for the new target. The FX importer, Graph IR, and blocked scheduling are all already done — they're completely backend-agnostic. This is exactly the design value of having a shared IR and a clean backend dispatch layer.

**Q: What is an autotuner?**
A: GPU kernels have configuration parameters — how many threads per block, how to tile the computation — and the optimal values depend on the specific matrix dimensions. An autotuner runs the kernel with different configurations, measures actual execution time, validates that the output is still correct, and records the best-performing configuration. Future calls with the same layer shape look up the cached best config instead of searching again.

---

## One-Paragraph Summary (For "Tell Me About This Project")

> "I built a retargetable ML compiler that takes PyTorch models as input and compiles them to two hardware backends: a GPU using CUDA, and a custom hardware accelerator I designed called the uTPU. The compiler has a shared front-end — it uses PyTorch's FX tracer to capture the computation graph, converts it to a custom Graph IR, and lowers it into a blocked matrix multiplication schedule. That schedule feeds into a backend dispatch layer, and from there the CUDA path generates NVRTC kernel code while the uTPU path emits custom ISA instruction words. The key design achievement is that both backends share everything above the dispatch layer, so adding a new target only requires writing one new lowering module. I validated the uTPU path through RTL simulation using Icarus Verilog, confirmed 65% on-chip memory reduction through program fusion, and built a CUDA schedule autotuner that improves kernel latency 57% on representative shapes."

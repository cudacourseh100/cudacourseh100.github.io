---
title: "Lesson 8.2 - Kernel Launch"
lesson_number: "8.2"
lesson_slug: "kernel-launch"
instructor: "Prateek Shukla"
course: "CUDA Programming for NVIDIA H100s"
source_page: "pages/lesson-8.2.html"
source_slide_pdf: "H100-Course/slides/8.2 Kernel Launch.pdf"
published_lesson_page: "/pages/lesson-8.2.html"
published_markdown_path: "/markdown/lessons/lesson-08-2-kernel-launch.md"
topics:
  - "Launch Bounds"
  - "Grid Constants"
  - "Dependent Grids"
  - "GDC"
  - "L2 Prefetch"
code_refs:
  - "matmul_12.cu"
  - "pingpong_experimental.cu"
generated_from:
  - "pages/lesson-8.2.html"
  - "H100-Course/slides/8.2 Kernel Launch.pdf"
---

# Lesson 8.2 - Kernel Launch

This file combines the published lesson page with full slide-deck text so agents can fetch and search it directly.

## Sources

- Lesson page: `pages/lesson-8.2.html`
- Slide deck: `H100-Course/slides/8.2 Kernel Launch.pdf`
- Published lesson URL: `https://cudacourseh100.github.io/pages/lesson-8.2.html`
- Published markdown URL: `https://cudacourseh100.github.io/markdown/lessons/lesson-08-2-kernel-launch.md`

## Lesson Summary

Lesson 8.2 is the handoff supplement. The point is not just how to launch a kernel, but how Hopper lets you shape launch residency, mark grid-lifetime parameters, and relax stream serialization so the next grid can boot before the previous one has globally retired.

## Why This Lesson Matters

**Grid start time and data safety become separate control points.**

Hopper lets launch policy become part of optimization. The lesson isolates the point where a multi-kernel pipeline stops being "kernel A then kernel B" and becomes a coordinated overlap problem with scheduler, memory, and host-launch consequences.

## Topics

- Launch Bounds
- Grid Constants
- Dependent Grids
- GDC
- L2 Prefetch

## Lesson Facts

- **Course Position:** Lesson 08.2, after Stream-K and before Multi GPU
- **Main Shift:** The machine can admit a dependent grid early while still fencing the exact reads that would break correctness.
- **Companion Deck:** `8.2 Kernel Launch.pdf`
- **Concrete Anchors:** `matmul_12.cuh`, `pingpong_experimental.cuh`, `cudaLaunchKernelEx`

## Key Takeaways

- Multi-kernel boundaries are often the correct design, not a failure to fuse hard enough.
- Dependent launch works by separating scheduler eligibility from memory-read safety.
- Overlap only helps when the launch window is large enough to hide startup cost but not so large that it creates damaging contention.

## Web Lesson Content

### Why kernel launch deserves its own lesson

The notes are explicit that multi-kernel design is not just cleanup work after the "real" optimization. Some post-processing stages genuinely need a boundary. Epilogues hit a practical ceiling, fused work raises register pressure, and cross-tile operations like LayerNorm or Softmax need a point where one phase is complete before the next phase reasons over its output.

#### What forces the boundary

Cross-tile reductions, heavy epilogues, and state that no longer fits comfortably beside the accumulator mainloop make a clean handoff the right architecture.

#### What Hopper changes

The handoff no longer has to mean "producer fully retires, then consumer cold-starts." Hopper can overlap the boot path if the launch packet and instruction stream cooperate.

#### The lesson's core rule

Start early, read late. Let the dependent grid reserve context and do safe setup work before the producer is globally finished, but fence dependency-sensitive reads until it is actually safe.

> **Optimization target:** do not ask "how do I avoid multiple kernels?" Ask "how do I make the boundary cheap enough that the right decomposition still runs close to full speed?"

### Launch controls span both compile-time shape and runtime authority

Some launch decisions are embedded into the kernel signature and codegen assumptions. Others live in the host launch packet. This supplement keeps those layers separate because Hopper performance depends on knowing which knob changes occupancy, which knob changes parameter lifetime, and which knob changes scheduler policy.

| Control | What it expresses | Why it matters here |
| --- | --- | --- |
| `__cluster_dims__(x, y, z)` | Compile-time thread block cluster shape. | Important on SM90 because cluster residency and multicast behavior are part of real kernel design. |
| `__launch_bounds__(...)` | Bounds on thread count and occupancy assumptions. | Launch shape is part of the resource budget that later overlap decisions have to live inside. |
| `__maxnreg__(N)` | Register cap per thread. | Matters when fused work and dependent-launch overlap are both pressuring residency. |
| `__grid_constant__` | Read-only grid-lifetime kernel parameter. | Exactly the kind of metadata path Hopper uses for descriptors and launch-stable control data. |
| `cudaLaunchKernelEx` + attributes | Host-side runtime authority over launch behavior. | This is where programmatic stream serialization permission is actually granted. |

The local fast.cu examples are useful here even though they are not dependent-launch demos. Files like `matmul_12.cuh` and `pingpong_experimental.cuh` show the launch-shape side of the story directly: cluster dimensions, launch bounds, and grid-constant descriptors are encoded into the kernels before any runtime launch packet ever appears.

```text
__global__ __launch_bounds__(NUM_THREADS)
void __cluster_dims__(CLUSTER_M * CLUSTER_N, 1, 1)
matmulKernel12(
    int M,
    int N,
    int K,
    const __grid_constant__ CUtensorMap tensorMapC,
    const __grid_constant__ CUtensorMap tensorMapA,
    const __grid_constant__ CUtensorMap tensorMapB,
    int* dspace);
```

### Default stream serialization leaves a tail-latency hole

Under the default rules, kernels launched on the same stream behave like an ordered queue. Kernel B is not eligible until Kernel A fully retires. That means the handoff is tied to whole-grid retirement rather than to the earlier moment when enough of the producer's work is already complete for the next stage to begin preparing.

#### Why the tail hurts

Near the end of the producer grid, only a shrinking subset of SMs still has work. The rest of the machine can sit idle even when the dependent grid is already known and ready to go.

#### What is wasted

The cost is not just empty cycles. The dependent grid also pays a cold-start penalty later because no context was admitted and no safe prefetch work was allowed to begin early.

```text
Default behavior:
producer grid fully retires
-> dependent grid becomes eligible
-> dependent grid starts booting
```

That is the exact serialization wall this supplement is trying to soften. The goal is not to violate correctness. The goal is to untie "eligible to boot" from "safe to read dependency-sensitive data."

### Dependent launch works because Hopper splits scheduler permission from read permission

Grid dependency control introduces two control points. One is producer-side and talks to the scheduler. The other is consumer-side and fences execution until the dependency has actually resolved. That split is the mechanism that makes overlap possible without turning the handoff into a race.

| Mechanism | What it does | What it does not do |
| --- | --- | --- |
| `griddepcontrol.launch_dependents` | Signals that a dependent grid may begin booting before the producer grid has globally retired. | It does not make dependency-sensitive loads safe on its own. |
| `griddepcontrol.wait` | Stops dependent warps at the exact fence where unresolved reads would become unsafe. | It does not itself grant early eligibility; it only fences execution inside the dependent grid. |
| `cudaLaunchAttributeProgrammaticStreamSerialization` | Host-side permission bit that tells the launch machinery to honor the dependent-launch protocol. | It does not replace the device-side instructions; the packet and the instruction stream are both required. |

The notes make the host-authority point strongly: device instructions do not override stream policy by themselves. If the host launch descriptor does not opt in, strict same-stream serialization remains in force even if the kernel contains the relevant instructions.

```text
// producer kernel on the stream
producer_kernel<<<gridA, blockA, 0, stream>>>(...);

// dependent kernel with programmatic stream serialization permission
cudaLaunchAttribute attr[1];
attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attr[0].val.programmaticStreamSerializationAllowed = 1;

cudaLaunchConfig_t cfg{};
cfg.gridDim = gridB;
cfg.blockDim = blockB;
cfg.attrs = attr;
cfg.numAttrs = 1;

cudaLaunchKernelEx(&cfg, consumer_kernel, ...);
```

> **The conceptual split:** producer code says "the dependent may boot now." Dependent code says "I will stop right here until the unsafe reads are truly safe." Both sides are required.

### The overlap window is where the optimization either pays off or collapses

Once the dependent grid can arrive early, the interesting question becomes timing. Launch too late and you hide almost none of the startup cost. Launch too early and the producer and consumer fight over the same execution slots, registers, shared memory, and cache capacity.

#### The dependent's wall

Warps in the dependent grid can run setup code and then park at `griddepcontrol.wait`. That reserves execution context early, which is useful but not free.

#### The producer's green light

`launch_dependents` should be placed at the latest safe point that still gives real overlap. Too conservative wastes the feature. Too aggressive creates contention and can lower total throughput.

#### The prefetch split

The notes highlight an asymmetric dependent kernel where dependency-bound warps stop at the wait fence while separate warps prefetch static weights into L2. That keeps the memory system busy while still respecting correctness for activation reads.

| Too little overlap | Too much overlap | Healthy target |
| --- | --- | --- |
| The dependent grid still pays most of its cold-start cost. | Producer and consumer compete for issue slots, residency, L2, and memory queues. | Launch at the latest safe moment that still hides meaningful startup latency. |
| Little net gain from the feature. | Prefetched lines can churn out of L2 before compute needs them. | Prefetch only stable, reusable tensors whose safety is independent of the producer's completion. |
| The dependent grid remains mostly idle until the producer is done anyway. | Total runtime can rise even though concurrency looks more impressive on paper. | Watch L2 hit rate, queue pressure, and residency headroom rather than judging overlap by appearance alone. |

```text
Dependent-launch protocol:
producer issues launch_dependents
-> dependent grid boots
-> dependency-bound warps stop at wait
-> dependency resolves
-> stalled warps continue and consume the produced data
```

### Practical guidance

1. **Do not treat multi-kernel boundaries as failure.** If the math needs a real boundary, the right question is how to hide its cost, not how to deny that it exists.
2. **Keep launch policy layered correctly.** Compile-time launch shape, grid-lifetime parameters, and host-side launch permission solve different problems.
3. **Remember the host owns the exception.** Programmatic stream serialization is opt-in behavior, not the default interpretation of same-stream work.
4. **Separate boot overlap from read safety.** Early scheduler admission and correct memory visibility are related, but they are not the same event.
5. **Tune for net throughput, not for dramatic overlap screenshots.** Concurrency that damages cache residency or queue pressure can easily lose to a slightly later, cleaner launch point.

#### Glossary

| Term | Definition |
| --- | --- |
| Programmatic stream serialization | A host-authorized relaxation of default same-stream ordering that allows dependent grids to participate in the launch/wait protocol. |
| Launch dependents | The producer-side signal that tells the scheduler a dependent grid may begin booting. |
| Dependent wait | The consumer-side fence that stalls warps before dependency-sensitive reads become legal. |
| Overlap window | The timing gap between early dependent admission and the moment the unresolved dependency actually becomes safe to consume. |
| Grid constant | A read-only kernel parameter guaranteed to stay stable for the lifetime of one launched grid. |

### Continue the course

This supplement finishes the single-node kernel handoff story. The next lesson leaves kernel boundaries behind and moves into the fabric itself: NVLink, NVSwitch, rails, and the system topology that governs how many H100s become one training machine.

## Full Slide Deck Text

Extracted from `H100-Course/slides/8.2 Kernel Launch.pdf` with `pdftotext -layout`. Total slides: 10.

### Slide 1: Kernel Launch

Prateek Shukla

### Slide 2: Some important constraints and functions

__cluster_dims__(x,y,z) for compile-time thread-block cluster shape.

__launch_bounds__(maxThreads[,minBlocksPerSM[,maxBlocksPerCluster]]). The 3rd
argument is the cluster-specific one that matters more on SM90.

__maxnreg__(N) to cap registers per thread.

__grid_constant__ for read-only grid-lifetime kernel params. This is especially
common for CUtensorMap / TMA descriptors.

__forceinline__ Forces nvcc to inline the function within a single translation unit.

__restrict__ tells nvcc that, for the lifetime of that pointer in the current scope, the
pointed-to memory is not aliased by any other pointer used to access the same data

### Slide 3: Need for multi kernel setups

The epilogue has a hard ceiling. Pushing past it destroys your performance. Fused
operations compete with the mainloop's accumulator tiles which might cause
register pressure.
Operations demanding cross-tile dependencies (LayerNorm, Softmax) are
architecturally incompatible with the independent-tile processing of a GEMM
epilogue. The math strictly requires a kernel boundary to reduce properly.
You must accept kernel boundaries and intermediate DRAM writes. The engineering
challenge isn't forcing a monolithic kernel it's making the handoff between kernels
virtually free
This is done by deploying dependent-launch protocols, L2 prefetch strategies, and
cross grid mbarriers

### Slide 4: The Stream Serialization Problem

Generally one grid must fully retire before the next grid on the same stream can begin
issuing work. This strict completion ordering creates a utilization gap as the first grid enters
its low-occupancy tail.
The hardware scheduler treats same-stream grids as a single ordered queue. It does not
hand out thread-block slots for Kernel B until Kernel A reports global completion, even if
some SMs are already underutilized.
As Kernel A nears completion, only a shrinking subset of SMs still has active work. The
remaining SMs sit idle because no next-grid blocks are eligible under default stream rules.
The handoff is a full-grid retirement event, not a tile-by-tile transfer. That means all
dependency-safe overlap opportunities are ignored unless explicit dependent-launch
control is enabled.
The critical cost is tail latency amplification: silicon is present but not issuing useful
instructions during the final phase of Kernel A.

### Slide 5: Hardware Signaling (GDC Instructions)

GDC instructions control over grid handoff timing inside the GPU execution timeline. Its
physical purpose is to separate "scheduler may start dependents" from "dependents may
read dependency-sensitive memory."
griddepcontrol.launch_dependents is a scheduler-facing signal emitted by running
warps. It tells the dispatch logic that the dependent grid is now eligible to boot, even before
the active grid has globally retired.
griddepcontrol.wait is a hard execution fence in the dependent grid. Warps reaching this
instruction are held until the launch dependency condition is satisfied, preventing
premature reads of unresolved data.
Together they form a two-part protocol: early boot permission plus memory-safety gating.
The scheduler can overlap startup work while the fence preserves correctness for
dependency-sensitive loads. These are physical control points in the instruction stream, so
placement directly changes overlap timing and hardware contention.

### Slide 6: Host Launch Authority

Only the CPU launch packet can authorize relaxed stream behavior for dependent
overlap. The physical purpose is to keep default stream semantics intact unless host
software explicitly opts in.
At launch time, the CPU writes attributes into the command descriptor consumed by the
GPU command processor. One attribute grants permission for dependent-grid scheduling
before full predecessor retirement.
If that permission bit is present, scheduler state machines honor launch_dependents and
wait as dependency control instructions. If it is absent, the scheduler enforces normal
serialization and the instructions have no enabling effect.
The handoff authority is therefore host-originated and hardware-enforced. Device-side
signaling cannot override stream policy unless launch metadata authorizes it.
This protects correctness across mixed workloads, where some streams need strict
ordering and others need controlled overlap.

### Slide 7: The Dependent's Wall (Wait)

This phase defines the early-boot stall point for producer warps in the dependent grid. Its physical
purpose is to let the dependent kernel reserve execution context early while blocking unsafe
memory traffic.

The dependent grid can be admitted and begin executing setup instructions on available SMs.
Producer warps run until they reach griddepcontrol.wait, where they are parked by hardware.

While parked, these warps cannot issue dependency-sensitive global reads such as activation
tensors produced by the active kernel. This prevents cache fill and memory ordering violations
from reading not-yet-final data.

Once the dependency signal is satisfied, the fence releases and those same warps resume issuing
loads. The result is immediate forward progress without paying full cold-start delay after producer
completion.

The tradeoff is reserved resources: parked warps still consume scheduling slots and can reduce
instantaneous headroom for other work.

### Slide 8: The Producer's Green Light (Launch)

This phase is where the active kernel emits the exact signal that opens dependent
scheduling. Its physical purpose is to trigger overlap at the last correctness-safe moment
that still exposes startup latency hiding.
Producer warps in the active grid execute griddepcontrol.launch_dependents when their
remaining writes are sufficiently advanced for dependent boot. This signal targets
scheduler eligibility, not immediate memory-read permission.
In persistent kernels, correct placement is tied to the final scheduled work tile's lifecycle.
The signal should be aligned with true pipeline completion, not merely the lexical end of a
code region.
After the signal, dependent blocks can be dispatched while the active grid drains residual
work. The dependent grid's wait fence still guards dependency-sensitive reads until the
required condition is met.
Signaling too late wastes overlap; signaling too early increases concurrent pressure and
can reduce net throughput.

### Slide 9: The L2 Prefetch Trick (Split DMA)

This phase implements overlap by dividing roles across warps in the dependent kernel. Its physical
purpose is to keep memory fabric busy with dependency-free transfers while dependency-bound
producers remain fenced.

One producer warp reaches the dependency barrier and pauses before activation reads. A
different prefetch warp continues running and issues DMA-style requests for static weights that
are independent of producer completion.

Those requests are targeted to populate L2 ahead of compute demand, reducing later miss
penalties when the dependency fence opens. The memory system can therefore warm useful
cache lines during otherwise stalled dependency time.

The handoff is asymmetric but safe: activation traffic waits behind the fence, weight prefetch does
not. When the fence releases, producers see improved effective latency because part of the
working set is already resident.

This strategy only works if prefetched data has high reuse and survives in L2 until consumption.

### Slide 10: Tuning the Overlap Window

The overlap ratio is set by the timing gap between dependent-launch signal emission
and true dependency readiness. A larger gap increases potential hiding, but only if
resources remain uncongested.

If signaling is too early, both kernels compete for SM issue slots, register file capacity,
shared memory allocation, and memory ports. This can throttle each kernel enough
that total time increases instead of decreases. If prefetch is too aggressive, L2 lines
churn before use and DRAM traffic spikes from refills. Bandwidth is then spent
moving evicted data back in, erasing the gain from prefetch overlap.

Practical tuning uses hardware counters: launch at the latest safe point, prefetch
only stable /reused tensors, and watch L2 hit rate plus memory-queue pressure as
the primary guardrails.

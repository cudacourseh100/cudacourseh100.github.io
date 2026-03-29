---
title: "Lesson 6 - WGMMA Part 1"
lesson_number: "6"
lesson_slug: "wgmma-part-1"
instructor: "Prateek Shukla"
course: "CUDA Programming for NVIDIA H100s"
source_page: "pages/lesson-6.html"
source_slide_pdf: "H100-Course/slides/6. WGMMA-1.pdf"
published_lesson_page: "/pages/lesson-6.html"
published_markdown_path: "/markdown/lessons/lesson-06-wgmma-part-1.md"
topics:
  - "Warpgroup"
  - "wgmma.mma_async"
  - "ldmatrix"
  - "Shared Descriptors"
  - "Tensor Cores"
code_refs:
  - "sm90_mma_tma_gmma_ss_warpspecialized.hpp"
generated_from:
  - "pages/lesson-6.html"
  - "H100-Course/slides/6. WGMMA-1.pdf"
---

# Lesson 6 - WGMMA Part 1

This file combines the published lesson page with full slide-deck text so agents can fetch and search it directly.

## Sources

- Lesson page: `pages/lesson-6.html`
- Slide deck: `H100-Course/slides/6. WGMMA-1.pdf`
- Published lesson URL: `https://cudacourseh100.github.io/pages/lesson-6.html`
- Published markdown URL: `https://cudacourseh100.github.io/markdown/lessons/lesson-06-wgmma-part-1.md`

## Lesson Summary

Lesson 6 is where Hopper tensor-core programming stops looking like older warp-level MMA. WGMMA turns four warps into one computational unit, launches tensor-core work asynchronously, consumes operand tiles from shared memory or registers, and depends on a very specific set of descriptors, register layouts, and fence semantics.

## Why This Lesson Matters

**The unit of tensor-core execution shifts from one warp to one warpgroup.**

Hopper WGMMA is not just a bigger MMA instruction. It changes who issues the work, where operands can reside, how register and shared-memory layouts are interpreted, and how asynchronous math launch fits into a pipelined kernel.

## Topics

- Warpgroup
- wgmma.mma_async
- ldmatrix
- Shared Descriptors
- Tensor Cores

## Lesson Facts

- **Course Position:** Lesson 06 of 10
- **Main Shift:** Tensor-core issue becomes a 128-thread coordination problem.
- **Companion Deck:** `6. WGMMA-1.pdf`
- **Code Anchor:** `sm90_mma_tma_gmma_ss_warpspecialized.hpp`

## Key Takeaways

- WGMMA is asynchronous tensor-core issue launched by a full warpgroup, not one warp.
- Operand placement matters: A can be registers or shared memory, B is shared memory, accumulators stay in registers.
- `ldmatrix` and the WGMMA descriptor define the operand layouts the hardware expects.

## Web Lesson Content

### Why WGMMA matters

The lesson frames four key Hopper innovations around tensor math: tensor-core work is now fully asynchronous, execution expands from warp to warpgroup, operands can come directly from shared memory or registers without the old register-only flow, and Hopper adds stronger support for FP8 and sparsity.

#### Asynchronous math launch

`wgmma.mma_async` launches tensor-core work and lets the issuing threads continue while the math is in flight.

#### Larger tiles, cleaner scheduling

WGMMA avoids chaining a forest of smaller warp-level MMA calls by letting four warps cooperate on a larger logical operation.

This is the compute-side analogue of the earlier async movement lessons. Hopper wants math and movement to stay in flight together, and WGMMA is the tensor-core issue model that makes that practical.

### Warpgroup execution is the paradigm shift

A warpgroup is four warps, or 128 threads, acting as one computational unit for tensor-core issue. All four warps reach the WGMMA instruction together, the scheduler checks convergence, fuses them for the operation, dispatches the command to tensor cores, and then the threads continue onward while the math proceeds asynchronously.

> **Why Hopper does this:** the older single-warp MMA model makes very large tiles and deep pipelining harder to schedule cleanly. The warpgroup model lets Hopper keep tensor cores busy for longer with less launch overhead.

#### The fixed `m64`

The notes explain why the M dimension is fixed at 64. The hardware maps operand fragments into a static tensor-core input pattern and avoids the need for a much more expensive dynamic routing structure. N is more flexible, while K is determined by input datatype.

| Dimension | Rule | Why it matters |
| --- | --- | --- |
| M | Fixed at 64. | Matches the static hardware mapping for warpgroup tensor-core issue. |
| N | Flexible, typically multiples of 8 or 16 up to 256. | Controls how many columns of B and C are processed. |
| K | Driven by input precision. | Determines the inner-product depth the tensor core consumes per operation. |

### The instruction shape tells you what the hardware expects

The core operation is `wgmma.mma_async`. Operand A can come from registers or shared memory. Operand B comes from shared memory. Accumulators and outputs live in registers.

```text
wgmma.mma_async.sync.aligned.shape.dtypeD.dtypeA.dtypeB
d, a_operand, b_desc, scale_d, imm_scale_a, imm_scale_b, imm_trans_a, imm_trans_b;
```

#### `.sync`

Requires the participating warpgroup threads to reach the instruction together before issue proceeds.

#### `.aligned`

Asserts warpgroup convergence at the instruction's program counter. Divergence here breaks the contract.

#### `wgmma.fence.sync.aligned`

The lesson is precise here: this is a register-ordering barrier for later `wgmma.mma_async` use, not a completion barrier. It orders relevant register accesses before a later async WGMMA reuses those registers as accumulators or A fragments. Shared-memory ordering still needs the appropriate async proxy fence.

### Register-sourced A and `ldmatrix`

Hopper gives A two possible locations. If A is reused heavily, keeping it in registers can reduce shared-memory pressure. If it is not reused enough to justify the extra instruction and register cost, shared-memory sourcing may be better.

#### A in registers

Good when reuse is high. Loading A from registers avoids doubling shared-memory bank pressure for A and B together.

#### A in shared memory

Simpler when the kernel does not benefit enough from register reuse to justify the extra address and register overhead.

#### `ldmatrix` is the main register-loading path

`ldmatrix` does not behave like a plain linear shared-memory load. It moves matrix fragments into opaque register patterns that are already aligned with tensor-core input expectations. The threads providing addresses are not always the threads that hold the resulting data.

- `.m8n8` describes the geometric shape of the loaded matrix fragment.
- `.x1`, `.x2`, and `.x4` control how many core 8x8 matrices are loaded per instruction.
- `.trans` can transpose during load, avoiding manual shuffling.
- BF16 data is packed into 32-bit registers because NVIDIA registers are 32-bit containers.

> **Important hardware fact:** WGMMA expects the exact fragment layout that `ldmatrix` produces. The register arrangement is not arbitrary.

### Shared-memory descriptors and swizzle complete the operand contract

When A or B is shared-memory sourced, WGMMA uses a compact 64-bit descriptor rather than a normal pointer. The lesson describes five packed fields: base address, leading byte offset, stride byte offset, matrix base offset, and swizzle mode.

| Field | Meaning |
| --- | --- |
| Base address | The shared-memory address of the tile, 16-byte aligned and packed into the descriptor. |
| LBO | Leading byte offset used along the operand's leading direction. |
| SBO | Stride byte offset in the other core-matrix direction. |
| Matrix base offset | Offsets the descriptor into the repeating swizzle region when the tile does not start at the pattern boundary. |
| Swizzle mode bits | Tell hardware whether the operand is unswizzled or uses 128B, 64B, or 32B swizzle. |

Shared memory is swizzled so that the physical address mapping aligns with bank-friendly access patterns. The descriptor has to encode the correct swizzle mode. If it does not, WGMMA will read scrambled data as if it were laid out linearly.

```text
// Representative pattern from the repo
uint64_t desc_a = make_smem_desc(&sA[0]);
uint64_t desc_b = make_smem_desc(&sB[0]);

asm volatile(
  "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 ..."
);
```

### Practical notes

1. **Treat warpgroup convergence as part of correctness.** WGMMA is not tolerant of casual divergence.
2. **Choose A's location deliberately.** Registers help when reuse is high; otherwise they can become pure pressure.
3. **Respect the descriptor and swizzle contract.** Tensor cores only see what the descriptor tells them to see.
4. **Remember what `wgmma.fence` is and is not.** It orders register use, but it is not a completion barrier for shared-memory visibility.
5. **Watch accumulator pressure.** Large tiles and async math are powerful, but they can explode register usage quickly.

#### Glossary

| Term | Definition |
| --- | --- |
| Warpgroup | Four warps acting together as one tensor-core issue unit. |
| `wgmma.mma_async` | Hopper's asynchronous warpgroup matrix multiply-accumulate instruction. |
| `ldmatrix` | The main mechanism for loading matrix fragments from shared memory into register layouts tensor cores expect. |
| Descriptor | A packed 64-bit shared-memory operand description used by WGMMA for operand layout and stride interpretation. |
| Scale / transpose immediates | Instruction flags that control accumulator behavior, sign scaling, and transpose behavior. |

### Continue the course

Lesson 6 introduces the basic WGMMA execution model and operand setup. The next lesson goes deeper into commit and wait groups, FP8 behavior, `stmatrix`, packing constraints, and the rest of the WGMMA lifecycle details.

## Full Slide Deck Text

Extracted from `H100-Course/slides/6. WGMMA-1.pdf` with `pdftotext -layout`. Total slides: 44.

### Slide 1: Introduction to WGMMA

Prateek Shukla

### Slide 2: 4 important innovations in mma in hopper

Apart from having faster tensor cores these are other important innovations:

1. The tensor core operations are now fully asynchronous. The math instruction
is now non blocking. This enables multiple in flight ops which means that
tensor cores would be active for more time
2. We have now transitioned from using single warps for tensor core operations
to using warpgroups which allow the use of much larger tiles
3. The ability to fetch the data directly from shared memory/registers for tiles
using asynchronous hardware. In ampere threads could not launch mma.sync
operation until the data arrived in the registers. In wgmma we can skip loading
registers and directly launch math operations
4. Specialized hardware support for fp8 and sparsity

### Slide 3: The paradigm shift: warp vs warp group

In Hopper we have transitioned from having a decade long emphasis on warp as
the unit of execution, we changed that to warp group
When you call wgmma you are fusing 4 warps to make a single computational
entity which operates on tensor cores. This maximizes the utilization of tensor
cores and avoids the hassle of pipelining multiple mma.sync operations.
All the 4 warps launch the wgmma instruction together. The warp scheduler
checks if all the warps converge at that instruction in PC and then scheduler fuses
them and then dispatches the math command to the tensor cores
Once the command is sent to the tensor core the threads go on to work on next
instruction.

### Slide 4: WGMMA pipeline

Load the data from tiles asynchronously, pre-load the next buffer if nessecery

If loading A tiles into registers then use ldmatrix and a bunch of complicated
addressing, use the wgmma descriptor for loading b tiles

If not just create a descriptor for A and B tiles.

Launch the wgmma operation.

Collect the results from registers. Launch wgmma.fence.sync.align to make the
writes to registers visible do some complicated address calculations to move the
data to shared memory.

### Slide 5: wgmma.fence.sync.aligned

wgmma.fence is a register-ordering barrier for wgmma.mma_async, not a
completion barrier. Its job is to make prior warpgroup register accesses visible in
the right order before a later wgmma.mma_async uses those same registers.

You need it in two main cases: before the first wgmma.mma_async in a warpgroup,
and any time a thread has accessed registers that a later wgmma.mma_async will
reuse as accumulators or A-fragment inputs

What it does not do is order shared-memory writes for the matrix
descriptors/data consumed by wgmma.mma_async. For that, the PTX docs require
an async proxy fence such as fence.proxy.async to order prior shared-memory
writes against later wgmma.mma_async reads

### Slide 6: wgmma.mma_async

The core async instruction which tells the tensor cores to do

A is read directly from shared memory/registers

B is read directly from shared memory

C, D reside in registers

The whole operation looks like this
wgmma.mma_async.sync.aligned.shape.dtype.bf16.bf16 d, a-desc, b-desc, scale-d,
imm-scale-a, imm-scale-b, imm-trans-a, imm-trans-b;

### Slide 7: .sync and .aligned

.sync - this is the SM wide barrier. It mandates that all participating threads in the
warpgroup (typically 128 threads, or 4 warps) must reach this instruction before
any of them can proceed.

.aligned - This qualifier asserts that all threads in the warpgroup are converged
(executing in lockstep) at the instruction's memory address (Program Counter)
and there is no thread divergence between the warp group

### Slide 8: The instruction qualifiers

Shape - the shape of your tile, we represent them using m64, n{}, k{}

dtypeD, dtypeA, dtypeB - the datatypes of D, A, B tensors

### Slide 9: m64nXkY

For all wgmma instructions, the M dimension is fixed at 64.

The K dimension (inner product dimension) is determined strictly by the precision
of the input matrices A and B.

The N dimension is the most flexible. It determines how many columns of matrix B
(and C) are processed.

N must be a multiple of the underlying memory allocation block size (typically
multiples of 8 or 16 depending on the type).

Valid N values range from 8 up to 256.

### Slide 10: The fixed m

The hardware is designed to hold 4 registers per thread. This means that each
warpgroup can hold 128 * 4 = 512 registers

This means that if we use the smallest tile m = 64, n = 8 we get 64 * 8 = 512
elements which is exactly the capacity of a warpgroup

The hardware physically maps the data from A to tensor core input B. At M = 64,
The hardware has a perfect, static map. This requires zero decision logic at
runtime.

If we have m configurable then you need a massive Crossbar Switch (a complex
multiplexer) between the registers and the math units to dynamically reroute the
data based on your requested which is expensive

### Slide 11: Operand location

Matrix A - registers/shared memory

Matrix B - shared memory

Matrix C/D - Registers only

### Slide 12: scale_d

Just specifies if we should have the accumulate(D = A x B + C) or

overwrite(D = A x B) operation

Scale_d = 1 - accumulate

Scale_d = 0 - overwrite

### Slide 13: scale_a/b

These are the scaling factors to scale the elements of A/B by 1 or -1

### Slide 14: The register operand d

These are the registers in which the outputs are gonna be stored

In the PTX instruction string, operand d is the first operand. It must be declared as a
vector (tuple) of registers enclosed in braces {}.

Number of output operands can be calculated by -

Total elements in the output / total number of threads

### Slide 15: The location of A

The tiles of A can either reside in shared memory or registers. And thus there are
two ways to tell wgmma where A tiles resides

a. Registers: If the tiles reside in the registers then we pass in the registers to
the wgmma instructions
b. Shared memory: We pass in the wgmma descriptor

### Slide 16: A in registers

This is a good approach when you are reusing the data because loading A and B
from shared memory doubles the pressure on the banks, multiple writes into shared
memory would be costly therefore better to load from registers.

Moving data from shared memory is expensive. Moving data from registers is
cheap. Loading from registers keeps the reusable data where it can be accessed
faster.

The data for registers comes from shared memory, the instruction ldmatrix takes
the unswizzled addresses and loads the data into registers

If you are not reusing the data then loading A from registers does not make sense
because there is instruction overhead and register pressure

### Slide 17: Mechanism for loading data from registers

Every thread provides the pointer for loading the data into registers and the
ldmatrix instruction populate the registers of the target threads with the data

Only for blackwell -

### Slide 18: ldmatrix

It is the primary mechanism for populating registers with Matrix A data in the
WGMMA pipeline.

Unlike a standard ld.shared (which loads linear data), ldmatrix loads data in opaque
register patterns designed to align physically with the Tensor Core's input lanes

The threads which point to the data are sometimes not the threads which end up
with the data.

.m8n8 is the geometric shape of the matrix tile being loaded from Shared
Memory into the Warp's registers.

The qualifiers .x1, .x2, and .x4 dictate the vectorization width and the number of
matrix fragments loaded per instruction.

### Slide 19: The x{1, 2, 4}

.x{1, 2, 4} tells us what number of core matrices of size 8 x 8 we can move to
registers for the computation

.x4 - This means loading 4 registers per thread or loading 4 core matrices of 8x8
dimension Most common because it saturates the register bandwidth by loading
16 * 16 * 4 = 1024 bytes of data from shared memory to registers

.x2 - Loading 512 bytes of data i.e loading 2 registers per thread. Generally use it
for loading smaller tiles

.x1 - 1 register per thread, mostly for boundary handling and other works.

### Slide 20: How a 16x16 tile is loaded into registers from shared memory

Whole operation works in two phases:

The Address Phase: Who provides the pointer? (The "Gather")

The Register Phase: Who holds the result? (The "Destination")

With x4 we are working with 4 section of 8x8 tiles

### Slide 21: The "Split-Warp" Strategy

We don't calculate addresses linearly. We split the warp into two "Vertical Strips"
(Left/Right) because ldmatrix loads 8-column chunks.

Left Half (Cols 0-7): This is controlled by the first 16 threads (Lanes 0-15).

Right Half (Cols 8-15): This is controlled by the next 16 threads (Lanes 16-31).

### Slide 22: Phase 1: Address Responsibility (which thread points)

In this phase, the hardware looks at the address register in each thread to decide
where to read from Shared Memory. The warp is split into 4 groups of 8 threads.

### Slide 23: * @param tile_row_offset Global row offset of this tile in the larger matrix

* @param tile_col_offset Global col offset of this tile in the larger matrix

### Slide 24: Phase 2: Register Responsibility (who holds the data)

Once the data is fetched, ldmatrix distributes it across the warp so that it is ready
for Tensor Core math. The distribution pattern is Row-Striped across Groups of 4
Threads.
For ANY 8x8 matrix (M0, M1, M2, or M3), the rows are distributed as follows:
Row 0 goes to Threads 0-3
Row 1 goes to Threads 4-7
Row 2 goes to Threads 8-11
...
Row 7 goes to Threads 28-31

### Slide 25: Layout of a single core matrix

### Slide 26: The address calculation

The tile lives in shared memory in a swizzled layout (chosen to avoid bank
conflicts), so the bytes for a "logical row" are not stored contiguously in physical
address order.
Each thread starts from its logical coordinate (e.g., "I need row r, column block c").
It then computes the physical shared-memory address for that fragment using
the swizzle mapping (often an XOR-based remap)
smem_addr = base + (row_offset ^ swizzle_mask) + col_offset
That computed address is the pointer the thread provides to ldmatrix.
Because the pointers are swizzle-mapped, the accesses land in different banks,
so the warp hits minimal/zero bank conflicts.
ldmatrix loads 16 bytes per thread and reorders it in fragmented layout

### Slide 27: Address bits breakdown

To understand how swizzle formula works we need to understand the address layout

Bits [0-1]: Byte offset within a 4-byte word. Irrelevant for bank selection.

Bits [2-6]: The Bank Index. These 5 bits determine which of the 32 banks (0-31) the
data lands in.

Specifically, bits [4-6] control the upper 3 bits of the bank index.

Bits [7-9]: The Row Index. Since the pitch is 128 bytes (2^7), bit 7 is the first bit that
changes when you move to the next row.

We want the Bank (controlled by bits 4-6) to change based on the Row (controlled by
bits 7-9).

### Slide 28: Swizzled address calculation

Physical_Address = (Base_Address + Linear_Offset) ^ Swizzle_Mask;

Where

./ 128 bytes = 2^7. We take bits [7,8,9], and shift them to positions [4,5,6].

uint32_t mask = ((linear_offset .> 7) & 0x7) .< 4;

./ Final Pointer

uint32_t smem_ptr = (base_ptr + linear_offset) ^ mask;

### Slide 29: Packing

The NVIDIA GPU architecture does not have 16-bit registers; it uses 32-bit
registers. Therefore, bf16 data must always be packed in pairs when residing in
registers.

Container: One .b32 (32-bit) register holds two bf16 elements.

Layout:

Bits 0-15 (LO): Element N (Even index)

Bits 16-31 (HI): Element N+1 (Odd index)

When moving this data, you often use .b32 type instructions. When computing,
you use .bf16x2.

### Slide 30: ldmatrix packing

If you use ldmatrix.sync.aligned.m8n8.x1.b16, each thread receives one
32-bit register.

This register contains two bf16 elements packed as described above.

If you use .x2 or .x4, you get 2 or 4 registers, each containing a packed bf16 pair.

The address in shared memory must be 16-byte (128-bit) aligned.

Using .trans allows you to load Row-Major data as Column-Major (or vice-versa)
directly into registers without manual shuffling.

### Slide 31: .sync.aligned and .trans

.sync: It acts as a mini-barrier. The hardware needs to guarantee that T16 is ready
to have its register overwritten by data requested by T0. If T16 was busy doing
something else, the hardware writer would corrupt T16's state

.aligned: All threads must execute it together so the hardware knows the Warp is
ready.

.trans: this tells whether to transpose the matrix while loading the data.

Because the registers of a thread is not visible to others, this whole instruction is
carried out by hardware internally.

### Slide 32: How they plug into wgmma instructions

When wgmma reads the registers {ra0, ra1, ra2, ra3} from a specific thread
(say, Thread 0 of Warp 0), it expects them to contain specific fragments of the
matrix.

Crucially, wgmma is hardwired to expect the EXACT layout that ldmatrix produces.

The Tensor Core hardware unwires these registers and re-maps them to the
physical dot-product units.

Instruction: wgmma... {acc}, {ra0...ra3}, desc_b ...

Hardware grabs {ra0...ra3} from Warp 0. It knows this is Rows 0-15 and others
from the warpgroups and executes wgmma with data from B.

### Slide 33: The wgmma descriptor

This is the 64 bit descriptor which contains information about the data stored,
strides and swizzle layouts

The descriptor packs 5 distinct fields. Consisting of the address, LBO, SBO, Matrix
base offset and the swizzle layout

### Slide 34: The blueprint

A single 64-bit register encodes the base address, K stride, M/N stride, matrix base
offset and swizzle mode. This compact structure allows efficient memory access
patterns for tensor operations.

Every stride is pre-divided by 16 to fit into the 14-bit fields(last 4 bits are useless
because the shared memory pointer is 16 byte aligned). The hardware multiplies by
16 at runtime, ensuring 16-byte alignment and reducing address latency, crucial for
high-performance tensor-core operations.

The base address must be 16-byte aligned. This alignment ensures that the
preprocessed strides can be correctly interpreted by the hardware, maintaining
efficient memory access and avoiding performance penalties.

### Slide 35: The base shared memory address

The base address is the first 14 bits of the descriptor. This is the shared memory
address of the tile which we are gonna load.

We extract the 14 bits from the address and then use them as the 14 bits of the
descriptor

### Slide 36: Leading byte offset (LBO)

the byte distance in shared memory between two core-matrix columns that are
adjacent along the operand's "leading" direction
LBO occupies bits 16-29. To encode the K stride, divide the desired byte stride by
16, mask to 14 bits, then shift left by 16
LBO for K major swizzled layouts is not used. Hardware assume its 1
For MN Major, LBO = offset from the first (swizzle-byte-size/16) rows to the next
(swizzle-byte-size/16) rows
For 128B swizzle, swizzle-byte-size/16 = 128/16 = 8, so it's "jump by 8 rows" in that
normalized layout.

### Slide 37: Stride Byte Offset (SBO)

It's the other stride WGMMA uses (along with LBO) to navigate a matrix operand
in shared memory. SBO is the byte distance in SMEM to jump "one core-matrix
block" in the other (non-leading) direction.

SBO occupies bits 32-45. Encode the M/N stride similarly:

For K major swizzled layout, SBO is offset from the first 8 rows to the next 8 rows

For MN Major swizzled layout, SBO is offset from the first 8 columns to the next 8
columns

### Slide 38: Matrix base offset

When SMEM is swizzled (e.g., 128B swizzle), the mapping from a logical address
to the physical SMEM bank/line uses a repeating permutation pattern. That pattern
repeats every 128B for a 128B swizzle pattern

If your matrix start address is exactly at the "pattern start" boundary for that
swizzle, then base_offset = 0. Otherwise, you must tell hardware which 128B
chunk within the repeating region you're starting in. That's what the base offset
encodes.

### Slide 39: Swizzle mode bits

If the data is the shared memory is swizzled then we need to apply the swizzled
layouts to the tensors to read the data correctly

If we don't set the correct swizzle layout correctly then wgmma units read the
scrambled data linearly

We have 4 swizzle modes, bits 61-63 control the swizzle mode. Use 00 for no
swizzle 01 for 128B, 10 for 64B, 11 for 32B

### Slide 40: The whole descriptor function

### Slide 41: What does the wgmma.mma_aync operation does

It asynchronously moves the data from your shared memory/registers to the tensor
cores for the operation. All the threads in the consumer warp group are calling
wgmma.mma_async

The tensor cores start computing on the data. Then result is going to be written
back into the registers of consumer warpgroup based on the warpId, laneId. Its low
overhead the hardware is dumping the data to the closes memory

The hardware wants to write results to the threads that are physically closest to the
math unit responsible for that chunk. So the threads are interleaved into small
blocks where each thread holds non contiguous data in memory

### Slide 42: A thread is getting all those registers in output

If we are working with wgmma with m64n256k16 then each thread has 128
elements in its accumulators. Every single WGMMA instruction is launched by
every single thread in the warpgroup!

For 128 threads in warpgroup you need, 16 stmatrix instructions with .x4 if
accumulators are 16 bit and 32 stmatrix if 32 bit accumulators

In the case of kernels like flashattention 3 one should be careful about the
register pressure by the softmax gemm pipeline if using registers for loading the
A tiles.

### Slide 43: imm-scale-{a,b} - option to "negate" A, this means A becomes -A

This can have two values 1 or -1, no other values because this transformation just
requires switching a bit instead of multiple bits
Imm-trans-{a,b} - this is option to transpose the tensor
This can take values 0 or 1
These are stored in bits because aren't enough bits left in the instruction string to
store a full 16-bit floating point number

### Slide 44: bye

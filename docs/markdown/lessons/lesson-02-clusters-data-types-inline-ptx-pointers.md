---
title: "Lesson 2 - Clusters, Data Types, Inline PTX, Pointers"
lesson_number: "2"
lesson_slug: "clusters-data-types-inline-ptx-pointers"
instructor: "Prateek Shukla"
course: "CUDA Programming for NVIDIA H100s"
source_page: "pages/lesson-2.html"
source_slide_pdf: "H100-Course/slides/2. Clusters, Data types, inline PTX, State Spaces.pdf"
published_lesson_page: "/pages/lesson-2.html"
published_markdown_path: "/markdown/lessons/lesson-02-clusters-data-types-inline-ptx-pointers.md"
topics:
  - "Clusters"
  - "DSMEM"
  - "Inline PTX"
  - "State Spaces"
  - "cvta / mapa"
code_refs:
  - "fast.cu/examples/matmul/matmul_12.cu"
generated_from:
  - "pages/lesson-2.html"
  - "H100-Course/slides/2. Clusters, Data types, inline PTX, State Spaces.pdf"
---

# Lesson 2 - Clusters, Data Types, Inline PTX, Pointers

This file combines the published lesson page with full slide-deck text so agents can fetch and search it directly.

## Sources

- Lesson page: `pages/lesson-2.html`
- Slide deck: `H100-Course/slides/2. Clusters, Data types, inline PTX, State Spaces.pdf`
- Published lesson URL: `https://cudacourseh100.github.io/pages/lesson-2.html`
- Published markdown URL: `https://cudacourseh100.github.io/markdown/lessons/lesson-02-clusters-data-types-inline-ptx-pointers.md`

## Lesson Summary

Lesson 2 moves from architecture into mechanism. The focus shifts to cluster residency, distributed shared memory, PTX state spaces, inline PTX constraints, packed datatypes, and the address conversions Hopper code relies on when memory movement stops being generic.

## Why This Lesson Matters

**High-performance Hopper code stops being generic pointer arithmetic.**

This lesson is where the course starts sounding unmistakably Hopper-specific: cluster-level cooperation, remote shared-memory access, instruction-level control through PTX, and the conversion steps needed to map abstract CUDA pointers onto concrete address spaces.

## Topics

- Clusters
- DSMEM
- Inline PTX
- State Spaces
- cvta / mapa

## Lesson Facts

- **Course Position:** Lesson 02 of 10
- **Main Shift:** Memory placement and instruction form become explicit optimization surfaces.
- **Companion Deck:** `2. Clusters, Data types, inline PTX, State Spaces.pdf`
- **Code Anchor:** `fast.cu/examples/matmul/matmul_12.cuh`

## Key Takeaways

- Clusters expand cooperation without splitting one CTA across many SMs.
- Inline PTX matters when operand type, state space, or synchronization semantics need exact control.
- `cvta` and `mapa` are the bridge from generic pointers to Hopper-specific memory behavior.

## Web Lesson Content

### Overview

Modern CUDA performance work often requires thinking across scheduling, memory topology, compiler behavior, and instruction-level control at the same time. Lesson 2 is where those layers begin to collapse into one picture. Thread block clusters change how blocks cooperate. Distributed shared memory changes what can be shared without bouncing through global memory. PTX and inline PTX expose instruction forms that CUDA C++ does not always surface directly. Pointer conversion instructions tie all of that back to the actual hardware address spaces.

> **Main idea:** high-performance GPU code is not only about doing less work. It is also about placing data in the right memory region, using the right instruction form, and minimizing unnecessary routing, synchronization, and bank conflicts.

#### Clusters

Guaranteed co-scheduled CTAs on nearby SMs create a new cooperation surface above the block level.

#### DSMEM

Remote shared-memory access inside a cluster reduces how often blocks need to fall back to global memory.

#### Inline PTX

Instruction-level control matters when the exact state space, datatype, or synchronization form is the point.

#### Pointer conversion

`cvta` and `mapa` connect high-level pointers to concrete memory and cluster semantics.

### Thread block clusters

A thread block cluster is a group of thread blocks guaranteed to be co-scheduled concurrently on adjacent SMs within a GPC. The important point is that this extends the hierarchy from **threads -> thread blocks -> clusters -> grid** without pretending one block is now spread across many SMs.

#### Why clusters matter

When one thread block cannot hold enough useful shared state on a single SM, clusters provide a way for nearby CTAs to cooperate more directly.

#### What does not change

A CTA still executes on exactly one SM. Clusters coordinate whole blocks; they do not split one CTA across the machine.

#### Key properties

- Clusters are explicit software constructs, not an automatic scheduling side effect.
- All CTAs in a cluster are guaranteed concurrent residency on nearby SMs.
- They are useful when cross-block data sharing is frequent enough to make global memory too expensive.
- They trade some coordination overhead for a larger effective working set.

#### Cluster sizing tradeoffs

| Cluster size | Upside | Downside |
| --- | --- | --- |
| 2 | Low coordination overhead and often a practical default. | Limited shared-memory reach across the cluster. |
| 8 | More cooperating CTAs with a useful balance of scale and usability. | More synchronization cost and more constraints on grid shape. |
| 16 | Maximum neighborhood described in the lesson notes. | Highest synchronization overhead and a larger risk of idle hardware. |

#### Declaring a cluster

```text
__global__ void __cluster_dims__(2, 1, 1)
cluster_kernel(float* input, float* output) {
  // kernel body
}

// launch shape must remain compatible with cluster dimensions
```

#### Cluster special registers

| Register | Meaning |
| --- | --- |
| `%cluster_ctaid` | CTA identifier inside the cluster. |
| `%cluster_nctaid` | Cluster dimensions. |
| `%cluster_ctarank` | Linearized rank of the CTA inside the cluster. |
| `%cluster_nctarank` | Total CTA count inside the cluster. |
| `%is_explicit_cluster` | Whether the launch used an explicit cluster rather than the implicit `1x1x1` case. |

### Distributed shared memory

Distributed shared memory is the cluster-era extension of shared memory. It allows direct memory access between SMs inside the same thread block cluster. Ownership still stays per block, but the programmer can address a neighboring CTA's shared-memory allocation without routing everything through global memory.

#### What it enables

Remote shared-memory loads, stores, and atomics between CTAs in the same cluster.

#### Why it is fast

It uses an SM-to-SM communication path instead of forcing all communication through L2 and HBM.

#### What it is not

It is not one merged shared-memory pool. Each CTA still owns its own local allocation.

The practical win is a larger effective working set at a much cheaper memory tier than global memory. That is especially useful for top-k, matrix pipelines, reductions, or tile exchange patterns where neighboring CTAs need each other's intermediate state.

> **TMA and multicast:** DSMEM becomes much more powerful when paired with TMA-backed movement and multicast. One producer can seed multiple consumers without repeating the full data movement path for every destination.

### PTX, inline PTX, operands, and state spaces

PTX is NVIDIA's virtual ISA inside the CUDA toolchain. CUDA C++ lowers into PTX, which is then lowered again into machine-specific code. Inline PTX matters when you want instruction-level control without rewriting an entire kernel in PTX.

#### Why use inline PTX

- To access hardware features that CUDA C++ does not expose cleanly.
- To hand-optimize small performance-critical fragments.
- To request exact operand types, state spaces, or synchronization forms.
- To inject constants or instruction parameters through templates.

```text
asm volatile(
  "ptx instruction here;"
  : /* outputs */
  : /* inputs */
  : /* clobbers */
);
```

#### Constraint modifiers and common operand classes

| Constraint | Meaning |
| --- | --- |
| `=` | Write-only output operand. |
| `+` | Read-write operand. |
| `r` | General integer register. |
| `l` | 64-bit integer register, commonly used for addresses. |
| `f` | 32-bit floating-point register. |
| `d` | 64-bit floating-point register. |

A `memory` clobber is often required when the assembly changes memory in ways the compiler cannot infer from the operand list alone. Without it, the compiler may legally reorder surrounding code in a way that breaks the intended semantics.

#### PTX state spaces

#### `.global`

Large device memory visible across the grid, high capacity and high latency.

#### `.shared`

Low-latency memory local to a CTA, or reachable through cluster mechanisms in DSMEM workflows.

#### `.local`

Per-thread storage, often backed by memory when registers spill.

#### `.const` / `.param`

Read-only constant space and kernel argument space with their own access semantics.

#### Data types that show up in Hopper code

- Standard integer and floating-point scalar families still matter for control flow and addressing.
- BF16, FP16, TF32, and FP8 matter because they change throughput, packing density, and tensor-core issue shape.
- Packed types matter because multiple low-precision values are often carried inside one 32-bit register.

### Addressing, banking, and swizzling

Shared memory is still banked, and the bank structure still matters even on Hopper. Poor access patterns can serialize otherwise parallel traffic, which is why low-level layout choices continue to be part of performance engineering instead of a side detail.

#### Shared-memory banks

- Threads in a warp ideally spread their requests across banks instead of piling onto the same bank.
- Bank conflicts increase replay and reduce effective bandwidth.
- The more structured the access pattern, the easier it is to reason about conflict behavior.

#### 128-byte transaction granularity

Many high-performance paths reason in 128-byte chunks because that aligns well with the granularity of efficient movement and layout transformations. This is one reason tile shapes and swizzles show up so often in Hopper code.

> **Why swizzling exists:** it is a layout trick for spreading accesses more evenly across banks and transactions so structured movement stays fast even when tensor tiles are large and reused aggressively.

### Generic pointers, state-space pointers, `cvta`, and `mapa`

CUDA C++ usually gives you generic pointers, but PTX instructions often need a pointer that is already understood in the correct state space. That is why conversion steps matter. The compiler cannot always assume which physical memory region a generic pointer should target inside low-level assembly.

#### Generic pointers

Flexible at the CUDA C++ level, but ambiguous for instructions that care about a specific state space.

#### State-space pointers

Explicitly identify shared, global, local, or constant semantics so the instruction can target the correct path.

#### `cvta` and `mapa`

```text
// Convert a generic pointer into a specific address space
asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(smem_addr) : "l"(ptr));

// Map a shared-memory address into another CTA's shared-memory region
asm volatile("mapa.shared::cluster.u64 %0, %1, %2;" : "=l"(remote_addr) : "l"(addr), "r"(rank));
```

`cvta` turns a generic address into the appropriate state-space address. `mapa` takes that one step further for cluster programming by mapping a shared-memory address into the address space of another CTA inside the cluster.

> **Important distinction:** cluster rank is not the same thing as a global block ID. A block ID tells you where a CTA sits in the full grid. Cluster rank tells you where it sits inside its local cluster, and that is the value mapping operations care about.

### Integration takeaways

1. **Do not flatten clusters into "shared memory, but bigger."** Residency guarantees and ownership still matter.
2. **Keep state-space terminology precise.** `.global`, `.shared`, and generic pointers are not interchangeable.
3. **Treat inline PTX as a surgical tool.** Use it when exact semantics are the point, not as a stylistic preference.
4. **Respect bank behavior and 128-byte layout assumptions.** Swizzles and packed layouts exist for a reason.
5. **Be exact about rank and mapping.** Many cluster bugs are really address-space or rank-mapping bugs.

#### Glossary

| Term | Definition |
| --- | --- |
| Cluster | A guaranteed co-scheduled group of CTAs on nearby SMs inside a GPC. |
| DSMEM | Distributed shared memory, remote shared-memory access across CTAs in one cluster. |
| PTX | NVIDIA's virtual ISA used as the lower-level target inside the CUDA toolchain. |
| `cvta` | Convert to address; turns a generic pointer into the correct state-space address. |
| `mapa` | Map address; remaps shared-memory addresses across CTAs inside a cluster. |

### Continue the course

Lesson 2 builds the address-space and instruction-level vocabulary Hopper kernels need. The next big step is synchronization under overlap: barriers, wait patterns, and correctness when loads and compute are intentionally in flight at the same time.

## Full Slide Deck Text

Extracted from `H100-Course/slides/2. Clusters, Data types, inline PTX, State Spaces.pdf` with `pdftotext -layout`. Total slides: 44.

### Slide 1: Clusters, Data types, inline

PTX, pointers
Prateek Shukla

### Slide 2: What are thread block clusters

Cluster is a collective of up to 16 thread blocks that are guaranteed to be
co-scheduled concurrently on adjacent SMs in a GPC

This extends the traditional CUDA programming hierarchy from three levels to
four explicit levels: threads -> thread blocks -> thread block clusters -> grid

All blocks in a cluster run concurrently on SMs that are physically close together,
enabling efficient cross-SM cooperation

The reason of using a cluster is for most of the time using distributed shared
memory for different purpose.

### Slide 3: Each thread block is constrained by the limited shared memory and

computational resources available on a single SM.

If the thread wants to access more data it needs to go to global memory to fetch
it which is expensive

For operations like top-k or matrix multiplication that require accessing data from
thread blocks executing on different SMs, thread block clusters provide a
solution.

They enable algorithms to trade modest memory bandwidth for significantly
expanded data accessibility across multiple SMs

### Slide 4: All that we do is take all the thread blocks and pool some of their memory. Now

threads from different blocks can access the data which they need without
needing to access the global memory

This pool is not pool in the sense that all the threads just pour all their shared
memory allocated into a single thing. Each block still owns its piece of shared
memory, the difference is that now the threads can go down and access other
block's memory

Generally this pool is small and constrained inside a GPC

### Slide 5: A few important points

A threadblock still runs on a single SM - the one-to-one relationship hasn't
changed

SMs can run multiple threadblocks simultaneously - this was always possible and
remains true

Clusters are a software concept, GPC is a hardware concept

No automatic assignment: You must explicitly define clusters in code; the
scheduler won't group blocks automatically

### Slide 6: On cluster sizes

With a cluster size of 8 thread blocks, the hardware can fit two clusters per GPC,
utilizing roughly 16 SMs per GPC efficiently

Mention that using cluster size > 8 requires explicitly setting
cudaFuncAttributeNonPortableClusterSizeAllowed at kernel launch. For size 8 or
smaller, the code remains portable

With a cluster size of 16 thread blocks, only one cluster fits per GPC, leaving
approximately 1-2 SMs idle per GPC or roughly 18 SMs unused across the entire GPU

Cluster size 2 have been found a lot more optimal as there is a hidden synchronization
cost to larger clusters (87 cycles for size 4+ and 150 cycles for size 16)

### Slide 7: What is Distributed Shared Memory

Feature in H100 which allows a direct memory accesses between SMs within a
thread block cluster

H100 implements a dedicated SM-to-SM network for clusters that provides fast,
low-latency access to remote shared memory which enables SMs to perform
load, store and Atomic operations across the shared memory of other SMs

DSMEM can be used simultaneously with L2 cache accesses. This allows
applications to utilize the combined bandwidth of both pathways when
communicating data between SMs

### Slide 8: Multicast and TMA with DSMEM

We can use TMA for asynchronous copy operations with DSMEM

One of the most important features of DSMEM are multicast operations that
deliver data to multiple SMs' shared memory simultaneously

TMA multicast bypasses the SM-to-SM network bottleneck we discussed earlier.
Instead of threads explicitly writing and reading across DSMEM (causing
congestion and synchronization overhead), a single thread per cluster can issue a
TMA multicast operation to distribute data to all SMs at once

### Slide 9: Creating thread block clusters

You can define cluster dimensions at compile time using the __cluster_dims__
attribute in your kernel declaration

Once defined you can launch the kernel normally but the grid dimensions must
be a multiple of cluster size

### Slide 10: Cluster handlers

#include <cooperative_groups.h> (cooperative groups help manage
clusters)

namespace cg = cooperative_groups;

cg.:cluster_group cluster = cg.:this_cluster();

creates a handle to the thread block cluster that the currently executing thread
belongs to
int* remote_smem = cluster.map_shared_rank(smem, target_block_rank);

remote_smem[idx] = value;

unsigned int cluster_size = cluster.num_blocks();

unsigned int cluster_rank = cluster.block_rank();

### Slide 11: But we are not gonna use it a lot.

We will generally use mapa to
convert the shared memory
address to address in clusters

### Slide 12: PTX special registers

These provide information about thread blocks in the cluster

%cluster_ctaid: CTA ID within the cluster

%cluster_nctaid: the dimensions of the cluster

%cluster_ctarank: linearized rank of the CTA in the cluster

%cluster_nctarank: Total number of CTAs in the cluster

%is_explicit_cluster: Differentiates explicit vs implicit (1x1x1) cluster launch

### Slide 13: Parallel Thread Execution

PTX is the assembly language of Nvidia GPUs and serves as an Instruction set
architecture in the CUDA ecosystem

When CUDA C++ files are compiled they translate into PTX which then jit compiles
into SASS which is then assembled into binaries

We are learning PTX because it enables direct access to low-level GPU features
that are not directly exposed in CUDA C/C++ which will help us to hand optimize
performance critical sections

Its widely used in a lot of famous repositories and learning PTX allows you to get
to know all kinds of optimizations in highly optimized libraries

### Slide 14: Inline PTX

Inline PTX is PTX assembly code that is embedded directly within CUDA C/C++
using asm keyword

We use inline PTX because it reduces a lot of heavy lifting in hand writing
everything from scratch in PTX and allows writing PTX instructions inside c++
code

Templates enable to embed parameterized compile-time constants directly into
GPU instructions which is why often times we would use templates with inline
instructions

### Slide 15: The format of PTX instructions

- asm() inserts the ptx code into your cuda program
- Adding volatile keyword after asm prevents the compiler from deleting or
moving your ptx instruction
- If you don't want compiler to see the way you access memory use memory
keyword in the clobbers section

### Slide 16: Input and output operands

Input and output operands are the bridge between your C++ variables and the
assembly instructions. They use a constraint-based syntax to tell the compiler
how to map variables to registers or memory

Output operands are declared first after the instruction string, then input
operands and then clobbers

Operands are numbered sequentially across outputs and then inputs

If you don't have output you can leave fields empty like asm("string".: inputs)

### Slide 17: Constraint modifiers

`=` - Write only

`+` - read and write only

`&` - Early clobber - prevents the compiler from using the same register for an
output and a subsequent input. Critical when output is written before all inputs
are consumed

### Slide 18: Constraints

`h` - 16 bit unsigned integers

`r` - 32 bit unsigned integer used for 32 bit addresses and unsigned integers

`l` - 64 bit unsigned integer and 64 bit addresses

`f` - 32 bit floating point number

`d` - 64 bit floating point numbers

`n` - immediate integer(the compile time constant)

### Slide 19: Some more points about operands

We represent the operands in text order like %0, %1, %2

So %0 would represent the first variable after the instruction string

We represent the variables using the constraint modifier and constraints("+r",
"=f", "+l" for output and "r", "l", "f" for inputs)

Curly braces are used for local register scoping {}

### Slide 20: The instruction string

- This is the string containing actual instructions which we want to run on the gpu

- Instructions can be written in a single line or multiple lines using newline

- You can use placeholders like %0, %1 etc to represent the operands which will be
replaced by the actual operands used in the assembly which we after :

asm volatile("wgmma.mma_async {%0, %1, %2, ...}")

### Slide 21: Output operands

They come after the instruction string

Use special modifiers and constraints to represent the output

Use comma separated outputs to represent multiple outputs

If there is no output operand leave the field empty (like ::)

### Slide 22: Input operands

Represented after output operands separated by colon

You can have either read only input operands ("n", "r", "f", "l") or read and
write operands ("+n", "+r", "+f") but not write only operands

You can use "+n" constraint to get the compile time operands

You can use multiple input operands by separating them with commas and
putting them after the output operands

### Slide 23: Clobbers

They tell the compiler what resources might get modified apart from the output
operands

This prevents the compiler from making wrong assumption and modify the code
incorrectly.

### Slide 24: What are state spaces in CUDA

State spaces in CUDA refer to different memory regions that are accessible by
GPU threads, each optimized for specific purposes

When writing code in PTX we need to specify for which state spaces we are
launching instructions

They're not just a low-level PTX detail. They're the reason we can write high
performance code by making memory placement a first class decision, not an
afterthought

### Slide 25: Register State Space (.reg)

The register memory

you need explicit .reg declarations when you require temporary registers that
aren't covered by the input/output constraint system.

Spilling mechanism: When register usage exceeds hardware limits, variables
automatically spill to local memory, which can significantly impact performance.

### Slide 26: Global State Space (.global)

Represents the global memory

For different operations which use data from global memory

### Slide 27: Local State Space (.local)

Per-thread private memory: Each thread has its own private local memory space
for storing data that doesn't fit in registers.

Rarely used

### Slide 28: Parameter State Space (.param)

Dual-purpose usage: Serves as kernel input parameters (read-only, per-grid) and
device function parameters (read/write, per-thread).

Argument passing mechanism: Used to pass arguments to kernels and functions
without using the general register file.

Again not used a lot

### Slide 29: Shared State Space (.shared)

Represents the shared memory

Cluster wide accessibility: With sub-qualifiers (::cta or ::cluster), can be accessed
by threads in other CTAs within the same cluster for advanced communication

### Slide 30: The data types

Unsigned integer types - .u8, .u16, .u32, .u64

Signed integer types - .s8, .s16, .s32, .s64

Floating point types - .f16, .f32, .f64

Raw bit patterns - .b8, .b16, .b32, .b64, .b128

Predicate type - store TRUE or FALSE values

### Slide 31: Some important datatypes in deep learning

bf16(stored as .b16): 8-bit exponent, 7-bit mantissa (16 bits total)

e4m3, e5m2(stored as .b8): 8-bit Float Formats (4-bit exponent + 3-bit mantissa,
or 5-bit exponent + 2-bit mantissa)

tf32(stored as .b32) - 32-bit format with same range as .f32 but reduced precision
(>=10 bits mantissa)

4-bit Float (e2m1)

Ultra-compact: 2-bit exponent, 1-bit mantissa (4 bits total)

### Slide 32: Packed data types

These pack multiple values into a single register for parallel operations:

.u16x2, .s16x2 - Holds two 16-bit integers in a 32-bit register (.b32)

.f16x2 - Holds two .f16s

.bf16x2 - holds two BF16

.e4m3x2 - holds two e4m3

.e5m2x2 - holds two e5m2

### Slide 33: What are memory addresses really

The entire memory of the GPU as one infinitely long ruler. The smallest unit of
memory you can have an address for is a Byte (8 bits) this is also called an atom.

The address is the distance in bytes from the start of the ruler. Byte Address is the
raw integer index on the ruler

In CUDA, you rarely work with raw bytes. You work with types (float, int, bf16).
When you write ptr + 1, the compiler does not move 1 byte. It moves 1 element.

### Slide 34: The shared memory and the banks

To allow 32 threads (a warp) to access memory simultaneously, shared memory is
divided into 32 discrete memory banks. The bank indexes from 0 to 31 and each
bank is 4 bytes wide.
The GPU determines which bank an address belongs to based on the 4-byte word
index.

(Bytes 0-3): Stored in Bank 0
(Bytes 4-7): Stored in Bank 1
(Bytes 8-11): Stored in Bank 2
(Bytes 12-15): Stored in Bank 3 and so on

### Slide 35: Bank Conflicts

The hardware can only serve one unique address per bank per clock cycle. If 2
threads conflict, the access is serialized (takes 2x as long). If 32 threads conflict
(worst case), it takes 32x as long.

If multiple threads read the exact same address, there is no conflict. The hardware
performs a multicast/broadcast, serving all threads in 1 cycle. A Bank Conflict
occurs when multiple threads in the same warp try to access different addresses
that map to the same bank.

The H100 memory controller processes requests in 128-byte transactions

If your warp requests more than 128 bytes total (e.g., every thread loads a 16-byte
float4), the request is split into multiple transactions (waves).

### Slide 36: Origin of swizzling

Because the H100 relies on Tensor Cores (which read 64x64 tiles of data), standard
linear addressing often creates massive bank conflicts (strided access).
If you have a 64x64 tile of bf16 numbers.
Row size: 64 elementsx2 bytes=128 bytes.
Bank capacity per row: 32 banks x 4 bytes = 128 bytes.
Every column in your matrix lines up perfectly in the same bank((128/4) mod(32))
We want to make sure that Matrix[0][0] and Matrix[1][0] land in different banks,
even though their stride is a perfect multiple of the bank width

### Slide 37: Pointers and state spaces

The Unified Virtual Addressing (UVA) system maps each state space to a
non-overlapping region:

0x0000000000000000 -> 0x0000FFFFFFFFFFFF (Reserved)

0x0001000000000000 -> 0x0001FFFFFFFFFFFF (Global memory)

0x0002000000000000 -> 0x0002FFFFFFFFFFFF (Local memory per context)

0x0003000000000000 -> 0x0003FFFFFFFFFFFF (Shared memory per block)

0x0004000000000000 -> 0x0004FFFFFFFFFFFF (Constant memory)

### Slide 38: Generic pointers

In modern CUDA (Compute Capability >= 3.5), all pointers are generic by default -
they're 64-bit addresses in a single unified virtual address space spanning all state
spaces

There is no runtime checking to determine what state space we're in

Can do all the normal arithmetic like the pointers in CPU

When a generic pointer is dereferenced, the hardware checks the high bits to
route the request to the correct memory subsystem, this results in a few lost
cycles in runtime space detection

### Slide 39: State Space pointers

When you write inline PTX or compiler generates it, pointers can be qualified with
their state space

NOT a pointer just raw bits meaningful to specific hardware instructions

Cannot be dereferenced in C++ (*(float*)shared_bits would crash or give garbage)

No runtime space detection hardware knows it's shared memory immediately

Enables specialized instructions that require explicit space qualification

### Slide 40: cvta convert to address

cvta (Convert To Address) is a PTX (Parallel Thread Execution) instruction that
converts pointers between generic addresses and specific memory space
addresses.

### Slide 41: The cuda c++ API for this

__cvta_generic_to_global

__cvta_generic_to_shared

__cvta_generic_to_local

__cvta_generic_to_constant

### Slide 42: Where do we need these

Tensor Cores require explicit shared memory addresses for ldmatrix

TMA needs explicit space qualifications

Thread Block Clusters need precise space control for cross-SM data sharing, we
use cluster specific addresses to work with them

In Multi-Instance GPU (MIG) configurations , explicit address space conversions
help ensure proper memory isolation between GPU instances.

### Slide 43: mapa and thread block clusters

Used for memory address conversion from shared memory in cta to distributed
shared memory

mapa converts a shared memory address from the current CTA's shared memory
space into the corresponding address in another CTA's shared memory within the
same cluster

Crucial point: It takes a rank, not a block ID!

### Slide 44: Rank vs Block Id

Block ID (%ctaid): Global position in the grid (0 to N-1 across all clusters)

Rank (%cluster_ctarank): Position within the cluster (0 to cluster_size-1)

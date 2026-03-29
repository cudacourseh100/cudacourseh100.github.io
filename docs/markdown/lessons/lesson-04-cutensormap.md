---
title: "Lesson 4 - cuTensorMap"
lesson_number: "4"
lesson_slug: "cutensormap"
instructor: "Prateek Shukla"
course: "CUDA Programming for NVIDIA H100s"
source_page: "pages/lesson-4.html"
source_slide_pdf: "H100-Course/slides/4. cuTensorMap.pdf"
published_lesson_page: "/pages/lesson-4.html"
published_markdown_path: "/markdown/lessons/lesson-04-cutensormap.md"
topics:
  - "CUtensorMap"
  - "TMA"
  - "Swizzle"
  - "Interleave"
  - "L2 Promotion"
code_refs:
  - "fast.cu/examples/matmul/matmul_12.cu"
generated_from:
  - "pages/lesson-4.html"
  - "H100-Course/slides/4. cuTensorMap.pdf"
---

# Lesson 4 - cuTensorMap

This file combines the published lesson page with full slide-deck text so agents can fetch and search it directly.

## Sources

- Lesson page: `pages/lesson-4.html`
- Slide deck: `H100-Course/slides/4. cuTensorMap.pdf`
- Published lesson URL: `https://cudacourseh100.github.io/pages/lesson-4.html`
- Published markdown URL: `https://cudacourseh100.github.io/markdown/lessons/lesson-04-cutensormap.md`

## Lesson Summary

Lesson 4 is the descriptor lesson. Hopper TMA only becomes truly asynchronous because the transfer state is encoded up front: base pointer, dimensions, strides, element traversal, swizzle, interleave, L2 fetch policy, and out-of-bounds behavior all become one reusable object the hardware can execute against.

## Why This Lesson Matters

**Address arithmetic stops living in the instruction stream and moves into metadata.**

This lesson explains why Hopper TMA is not just "a faster copy." It is a descriptor-driven movement path whose performance and correctness depend on encoding tensor structure, layout policy, and fetch behavior before the async operation begins.

## Topics

- CUtensorMap
- TMA
- Swizzle
- Interleave
- L2 Promotion

## Lesson Facts

- **Course Position:** Lesson 04 of 10
- **Main Shift:** Movement becomes a first-class object with an explicit contract.
- **Companion Deck:** `4. cuTensorMap.pdf`
- **Code Anchor:** `fast.cu/examples/matmul/matmul_12.cuh`

## Key Takeaways

- Descriptors are what let Hopper launch one async transfer and let hardware do the rest.
- Dimensions, element strides, byte strides, and box dimensions are not interchangeable units.
- Swizzle, interleave, L2 promotion, and OOB fill are performance and correctness knobs, not decoration.

## Web Lesson Content

### What TMA changes

TMA is Hopper's hardware path for large asynchronous tensor copies between global memory and shared memory. Instead of tying movement to per-thread address generation, Hopper lets one launching thread describe the transfer and then continue executing while hardware handles the operation in the background.

#### Before descriptor-driven movement

Threads participate heavily in address generation, copies fragment into smaller pieces, and the SM stays tethered to the mechanics of movement.

#### With Hopper TMA

One instruction plus a descriptor can describe a whole tile transfer while the hardware handles the movement path mostly on its own.

That is why the lesson keeps returning to descriptors. Hopper's asynchronous machine works because layout knowledge is encoded ahead of time rather than recomputed by the issuing threads every time.

> **Core idea:** `CUtensorMap` shifts transfer state out of the instruction stream and into a reusable 128-byte metadata object that the TMA hardware can execute against.

### What the descriptor contains

A `CUtensorMap` is created on the host, encoded with the CUDA Driver API, and then used by the runtime path that issues TMA instructions. Conceptually it answers three questions: where the tensor lives, how it is laid out, and how the hardware should move it.

#### Addressing metadata

Base pointer, rank, global dimensions, and byte strides define the source tensor in memory.

#### Traversal metadata

Box dimensions and element strides define the tile size and per-dimension movement behavior.

#### Layout and policy metadata

Data type, swizzle, interleave, L2 promotion, and OOB fill shape how the transfer behaves.

#### Encoding workflow

1. Create a `CUtensorMap` object in host memory.
2. Encode it with the CUDA Driver API, typically with `cuTensorMapEncodeTiled(...)`.
3. Pass the encoded descriptor to the path that will launch TMA-backed movement.

```text
CUtensorMap tma_desc;
// host-side descriptor object, 128 bytes, 128B aligned

// encode with CUDA Driver API
// cuTensorMapEncodeTiled(&tma_desc, ...);
```

### Field-by-field rules that matter

The easiest way to break a tensor map is to blur units and invariants. The lesson material is very clear that some fields are expressed in elements, some in bytes, and some carry alignment constraints that depend on datatype or interleave mode.

| Field | Meaning | Unit / constraint |
| --- | --- | --- |
| Data type | Defines how the source tensor is interpreted and what alignment / size rules apply. | Type enum such as FP16, BF16, FP32, TF32, integers. |
| Rank | Number of tensor dimensions, not matrix rank. | Notes frame supported movement across 1D to 5D tensor copies. |
| Global address | Base device pointer in HBM. | 16B aligned baseline, stricter when interleaving is enabled. |
| `globalDim` | Extent of each tensor dimension. | Measured in elements. |
| `globalStrides` | Distance to move across higher dimensions in source memory. | Measured in bytes. |
| `elementStrides` | Logical step size for traversal during the async copy. | Measured in elements, array size equals rank. |
| `boxDim` | Tile size for one TMA transfer. | Measured in elements, must fit downstream layout assumptions. |

> **Keep this straight:** global dimensions and box dimensions are in elements, while global strides are in bytes. Losing that distinction is one of the fastest ways to encode the wrong descriptor.

#### Datatype and rank

The notes list common types including unsigned integers, signed integers, FP16, BF16, FP32, FP64, and TensorFloat-32 variants. Rank here means tensor dimensionality, not matrix rank from linear algebra.

#### Alignment and interleave constraints

Interleave modes tighten the address and stride rules. The notes call out 16-byte alignment for 16B interleave, 32-byte alignment for 32B interleave, stride granularity to match, and a dimensionality requirement of at least 3 when interleaving is used.

### Swizzle and interleave are how layout becomes hardware-friendly

Swizzle is the shared-memory-side layout transform. Interleave is the source-layout decode in global memory. They are related because Hopper often has to decode a packed source layout and then place the result into a shared-memory arrangement that downstream consumers can read efficiently.

#### Why swizzling exists

Shared memory has 32 banks. Regular strided layouts can repeatedly hit the same banks and serialize what should have been parallel access. Hopper uses hardware-accelerated address permutation during transfer so the physical placement in shared memory better matches the access pattern of the consuming warps.

| Mode | What changes | Typical use |
| --- | --- | --- |
| 32B | Smallest swizzle span, lower cache-line utilization, narrower tiles. | Very small inner dimensions. |
| 64B | Better bank spread than 32B but still not the strongest layout protection. | Mid-width tiles. |
| 128B | Largest common swizzle span, aligned with the main shared-memory / tensor consumption pattern. | Common FP16 / BF16 tensor-core pipelines. |

The notes also distinguish atom size and span. The atom is the indivisible chunk the swizzler moves. The span is the repeat window of the swizzle pattern. That is why the notes highlight the bounding-box constraint: if the inner dimension in bytes exceeds the swizzle span, the pattern repeats and can recreate bank collisions.

#### Interleave coupling

Some source layouts are physically interleaved, for example NC/8HWC8 or NC/16HWC16 style packing. The interleave mode tells TMA how to decode those chunks. The lesson notes explicitly call out that 32B interleave must pair with 32B swizzle because those hardware paths operate in lockstep.

### L2 promotion and out-of-bounds fill finish the movement contract

The descriptor does not stop at layout. It also controls how aggressively the system fetches into L2 and what value the destination should receive when a tile extends beyond the valid tensor boundary.

#### L2 promotion

| Mode | Meaning | Best fit |
| --- | --- | --- |
| NONE | Use the default smaller fetch behavior. | Sparse or irregular access where over-fetching would waste bandwidth. |
| L2_64B | Promote to 64B fetch width. | Moderately dense cases with narrow contiguous width. |
| L2_128B | Fetch a full cache line in one shot. | Common dense GEMM or convolution-style pipelines. |
| L2_256B | Fetch two full cache lines logically together. | Highly dense workloads with immediate reuse and enough locality to justify cache space. |

The lesson's rule of thumb is simple: dense GEMM and convolution often benefit from 128B or 256B promotion, while sparse lookups usually do better with no promotion to avoid fetching neighbors that will never be touched.

#### Out-of-bounds fill

Edge tiles still need to move. OOB fill lets the transfer continue without manual per-element guards by synthesizing destination values for out-of-range positions while leaving the source tensor in HBM unchanged.

- Zero fill is useful for padding and clean boundary math.
- NaN-related fill modes are useful for debugging or specialized behavior.
- This simplifies tiled kernels that naturally extend beyond tensor edges.

### Practical guidance

1. **Get the descriptor math right first.** Rank, dimensions, and byte strides must describe the real tensor.
2. **Respect alignment rules.** Interleave mode and datatype change what the hardware expects.
3. **Choose tile geometry around the consumer.** TMA should feed the compute pattern you actually use downstream.
4. **Use swizzle deliberately.** It only helps when it matches the access pattern of the reader.
5. **Match L2 promotion to contiguous width.** Dense predictable traffic and sparse lookups want very different fetch behavior.

#### Glossary

| Term | Definition |
| --- | --- |
| TMA | Tensor Memory Accelerator, Hopper's asynchronous tensor copy engine. |
| `CUtensorMap` | The host-encoded descriptor that teaches TMA how to interpret and move a tensor tile. |
| Swizzle | Shared-memory address remapping used to reduce bank conflicts and align with consumer access patterns. |
| Interleave | Source-layout decoding for packed global-memory formats. |
| L2 promotion | A fetch-width hint that trades bandwidth efficiency against possible cache pollution. |
| OOB fill | Destination fill policy used when a TMA tile extends beyond the valid tensor boundary. |

### Continue the course

Lesson 4 gives TMA its descriptor and layout vocabulary. The next lesson, `cp.async.bulk`, moves into the instruction family that uses this machinery directly for structured bulk movement, grouping, multicast, and barrier-linked completion.

## Full Slide Deck Text

Extracted from `H100-Course/slides/4. cuTensorMap.pdf` with `pdftotext -layout`. Total slides: 39.

### Slide 1: cuTensorMap

Prateek Shukla

### Slide 2: What is TMA

TMA is the new unit which enables fully asynchronous data movements with
minimal thread involvement

A single thread launches TMA instructions and immediately continues execution,
the whole operation is handled by the hardware in the background

Instead of address calculation you get descriptors

TMA can transfer data to shared memory of multiple SMs simultaneously

Can handle 1d-5d tensors

### Slide 3: How does H100 gets the perfect asynchronicity

We need all the units to be busy all the time

H100 gets it right by having multiple buffers loading using TMA and having tensor
cores do the operations simultaneously

The more important point is that you don't need a lot of complex engineering.
Descriptors handle everything from data to layout to optimizations, leaving a lot of
the heavy lifting to the hardware

### Slide 4: Why do we need Descriptors

Descriptors are the piece of the puzzle which enable the asynchronous nature of
H100.
In earlier times the whole fetching of data would happen using indexing where
threads would need to compute the address of the data which they wanted to
fetch and then data was fetched.
This was fixed a bit in ampere where the copy instruction was non blocking but
threads still had to compute the addresses for every 16 bytes. SMs were tethered
to the copy engine.
Now with H100s The descriptor encapsulates the entire transfer state. Because all
the information required to complete the job is contained in that 128-byte object in
memory, the hardware can do the job in the background without SM's involvement

### Slide 5: How to use TMA for copies

- Use CUDA API to create a cuTensorMap
- Encode it with information required
- Launch the operation

### Slide 6: cuTensorMap

It is a descriptor which stores the information about -

Memory base pointer (device address)
Tensor shape (number of elements per dimension)
Strides (in bytes)
Data type
Alignment and swizzling
Memory space (device, host, or unified)
Order and rank (up to 32 dimensions)
Optionally: tiling or "interleaved" layouts

### Slide 7: Creating a tensormap object

CUtensorMap tma_desc;

Created in host memory as a local variable, it has a size of 128 bytes and It must
be 128B aligned. It is not a regular pointer but a specific data structure created by
nvidia

cuTensorMapEncodeTiled() function is used to encode the values in the
tensormap.

### Slide 8: CUtensormap.

This is the tensormap object which we created using

CUtensorMap tma_desc which is gonna be filled with the encoded descriptor.

### Slide 9: Determines the datatype of the actual data which you are gonna copy from HBM

TMA engine uses this to automatically get memory alignment and transfer size

### Slide 10: Datatypes

CU_TENSOR_MAP_DATA_TYPE_UINT8 - Unsigned 8-bit integer
CU_TENSOR_MAP_DATA_TYPE_UINT16 - Unsigned 16-bit integer
CU_TENSOR_MAP_DATA_TYPE_UINT32 - Unsigned 32-bit integer
CU_TENSOR_MAP_DATA_TYPE_INT32 - Signed 32-bit integer
CU_TENSOR_MAP_DATA_TYPE_UINT64 - Unsigned 64-bit integer
CU_TENSOR_MAP_DATA_TYPE_INT64 - Signed 64-bit integer
CU_TENSOR_MAP_DATA_TYPE_FLOAT16 - 16-bit floating-point (half precision)
CU_TENSOR_MAP_DATA_TYPE_FLOAT32 - 32-bit floating-point (single precision)
CU_TENSOR_MAP_DATA_TYPE_FLOAT64 - 64-bit floating-point (double precision)
CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 - 16-bit brain floating-point
CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ - 32-bit float with flush-to-zero mode
CU_TENSOR_MAP_DATA_TYPE_TFLOAT32 - TensorFloat-32 format
CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ - TensorFloat-32 with flush-to-zero mode

### Slide 11: Flush-to-zero (FTZ) mode is a floating-point arithmetic optimization that handles

denormalized (subnormal) numbers by replacing them with zero instead of
computing them normally.

### Slide 12: Tensor Rank and Global Memory address

- Tensor Rank is number of dimensions of the tensor, not the matrix rank from
linear algebra
- Global Memory address is the memory address of the tensor in the HBM

The global address must be 16 byte aligned for efficient memory access patterns
in Hardware

### Slide 13: Global Dimension (globalDim)

Array that specifies the size of each dimension of the tensor in terms of number of
elements (not bytes).

For a r dimension array we arrange data in following way

globalDim[0] = innermost dimension

globalDim[r] = outermost dimension

Note that every element of this array ranges from 0 to 2 ^ 32 (around 4 billion)

### Slide 14: These are byte strides between elements in each dimension

how many bytes hardware must skip to move from one coordinate to the next
along specific dimensions.

The stride for the innermost dimension is implicit based on element size and the
size of this array is (rank - 1)

globalStrides[0] is the byte stride for globalDim[1] (second-innermost dim).

globalStrides[rank-2] is the byte stride for globalDim[rank-1] (outermost
dim).

### Slide 15: It is the number of elements you want to skip along each dimension when you are

doing the async copy.

It is also an array of size rank and have specific requirements -

It must contain non zero values

Stride value should be less than or equal to 8

Array size = rank

Also it is calculated in number of elements and not bytes

### Slide 16: It is the tile size. It is an array of size rank with one entry per tensor dimension

matching the tensor rank.

Specifies the size (in elements) of the traversal box the chunk of data transferred
per TMA operation from global memory to shared memory

Measured in elements (not bytes), corresponding to the data type being transferred

Inner dimension must be <= swizzle size

### Slide 17: Shared memory

Shared memory is the on-chip scratchpad attached to an SM. Your thread block
gets a region of it, and every thread in that block can read or write any address in
that region. Programmatically it looks like one flat address space.

What the hardware does underneath is split that address space into 32 banks. A
bank is just one independently serviceable lane of the shared-memory array. The
reason there are 32 is simple: a warp has 32 threads, so the ideal case is one
thread per bank.

Important point: banks are not 32 separate arrays you index manually. You still just
compute an address. The bank is chosen from the address.

### Slide 18: Swizzling and bank conflicts

All 32 banks can be accessed simultaneously in one cycle. a warp issues one logical
load/store instruction, but the SMEM subsystem may execute it in several rounds:

round 1: one subset of requests from each bank

round 2: the remaining conflicting requests and so on

However, if multiple threads access the same bank, the accesses serialize causing
bank conflicts. The normal way to access data in shared memory are strided access
patterns this causes in serialization.

1 request -> no conflict, 2 requests -> 2-way conflict, 4 requests -> 4-way conflict,   8
requests -> 8-way conflict, 32 requests -> worst-case conflict

### Slide 19: How swizzling works

It's an address-mapping function. It takes the logical address (the GmemAddress
your program thinks it's writing to) and "shuffles" its bits to create a physical
address (the SmemAddress where the data actually lands).

We need to make the sequential access patterns and spread them across all the
banks to avoid conflicts

To do this we have 3 modes of swizzling 32B, 64B, 128B

The 128B, 64B, and 32B layouts define the span or chunk size of the memory
swizzling and, consequently, the size of the memory transactions. Choosing the
right layout is a critical performance tuning step based on your application's
memory access patterns.

### Slide 20: The bank equation

where a' is the actual shared-memory byte address used by hardware. If swizzle is
enabled, a' is the swizzled address, not the logical unswizzled one. This 32-bank,
4-byte-step model is the standard way to reason about conflicts and matches the
address-bit view you were using earlier

That means bank ID comes from address bits a'[6:2]:

then it repeats every 32 * 4 = 128 bytes.

### Slide 21: Swizzle formula

a'= a^((a & Y_mask) >> 3)

where a is the byte address, Y_mask is 1<< 7 and the low 4 bits a[3:0] are never
touched.
32B: a[4]' = a[4] xor a[7]
64B: a[5:4]' = a[5:4] xor a[8:7]
128B: a[6:4]' = a[6:4] xor a[9:7]
the bits that are XORed are the low bits of the swizzle-pattern row index in shared
memory, not inherently the row of your mathematical matrix. They become your
matrix-row bits only if your layout maps matrix rows onto those shared-memory
pattern rows.

### Slide 22: Swizzle Span and atom

For SM90, the full swizzle pattern repeats every:

32B swizzle: 256 bytes

64B swizzle: 512 bytes

128B swizzle: 1024 bytes

Each row is 128 byte, each row is split into 8 cells of 16 bytes each. The swizzle
permutes those 8 cells from row to row. So repeat_bytes = 128B * distinct_rows

A swizzle row is not your matrix row. It is a 128 byte shared memory line. The swizzle
only permutes those 16B chunks. Inside the swizzle atom, elements are not
randomly shuffled the hardware/layout keeps small contiguous groups intact, it only
permutes those groups as units

### Slide 23: 128B swizzling

Span = 128 Bytes, Atom = 16 Bytes
The hardware thinks of the destination shared-memory region as:
Rows of 128 bytes, each row split into 8 cells of 16 bytes each, cell index inside a
row: x = 0..7 row index: y = 0, 1, 2, ...
If the base shared-memory address is aligned to the swizzle's full pattern boundary,
the physical destination for logical cell (y, x) is: phys_addr = base + 128*y + 16*x
We apply the swizzle formula like this:
a[6:4]' = a[6:4] ^ a[9:7]
the whole 128B row is treated as 8 swizzlable 16B cells. Row id modulo 8 selects the
permutation, the pattern repeats every 8 rows. Total repeat size = 8 * 128B = 1024B

### Slide 24: 64B swizzling

64B swizzle operates on groups of 4 16B cells.
cell index inside a row: x = 0..3
row index: y = 0, 1, 2, ...
the physical destination for logical cell (y, x) is:
phys_addr = base + 128*y + 16*x'
x' = x ^ (y & 3)
a[5:4]' = a[5:4] ^ a[8:7]
The left 64B half of the row is permuted by the row id mod 4, the right 64B half is
permuted the same way. The pattern repeats every 4 rows, total repeat size = 4 *
128B = 512B

### Slide 25: 32B swizzling

32B swizzle operates on pairs of 16B cells.
cell index inside a row: x = 0-1
row index: y = 0, 1, 2, ...
the physical destination for logical cell (y, x) is:
phys_addr = base + 128*y + 16*x'
x' = x ^ (y & 1)
a4' = a[4] ^ a[7]
inside each 32B region, the two 16B cells swap on every other 128B row. The full
pattern repeats every 2 rows. since each row is 128B, the repeat period is 256B

### Slide 26: Setting up swizzling

### Slide 27: Interleaving

Data in Global Memory is not always linear. Libraries like cuDNN primarily use
primarily uses the NCHW memory layout for optimal GPU performance in
convolutions and other operations.

The CUtensorMapInterleave parameter informs the TMA how to decode the
address space. It maps the physical, interleaved arrangement in HBM to a logical,
linear reconstruction in Shared Memory

The Hopper TMA supports specific interleaving patterns designed for NC/xHWCx
layouts:
CU_TENSOR_MAP_INTERLEAVE_16B

CU_TENSOR_MAP_INTERLEAVE_32B

### Slide 28: The two major modes

Layout: NC/8HWC8 (Vector of 8 channels).

Math: 8 channelsx2 bytes (FP16)=16 bytes.

Behavior: Data is accessed in 16-byte interleaved chunks

Layout: NC/16HWC16 (Vector of 16 channels).

Math: 16 channelsx2 bytes=32 bytes.

Behavior: Data is accessed in 32-byte interleaved chunks.

Note: If the channel count isn't a multiple of the slice, the last slice must be
zero-padded to maintain the interleave granularity.

### Slide 29: Interleaving and swizzling

Coupling: Interleaving and Swizzling are not independent.

The Requirement: The documentation states:

"When interleave is CU_TENSOR_MAP_INTERLEAVE_32B, the swizzle parameter
must be set to CU_TENSOR_MAP_SWIZZLE_32B."

Reasoning: The hardware path for de-interleaving 32B chunks from Global
Memory feeds directly into the swizzling logic for 32B atoms in Shared Memory.
They operate in lockstep to map the complex global layout to a bank-conflict-free
shared layout

### Slide 30: A few important points

If using 32B interleaving then your global address should be 32B aligned and if
using 16B interleaving then global address should be 16B aligned

Also your global strides should be a multiple of 16 if using 16B interleaving and
global strides should be multiple of 32 if using 32B interleaving

Also if you are using interleaving the dimensionality must be greater than or equal
to 3

### Slide 31: Usage

### Slide 32: L2 promotion

Fetching data from HBM is slow compared to fetching data from the L2 cache.

Prefetching anticipates future data needs by issuing non-blocking transfers from
HBM to L2 cache

Without promotion (CU_TENSOR_MAP_L2_PROMOTION_NONE), the memory controller
uses a standard cache line fetch size (typically 32 bytes). This is inefficient
because the L2 cache manages data in 128-byte lines.

By promoting the fetch size to 128B or 256B, a single TMA request ensures that a
larger contiguous block of data is brought into the L2 cache in one go.

This maximizes bandwidth efficiency for predictable patterns like GEMMs,
ensuring data is resident in L2 exactly when the compute units need it.

### Slide 33: L2 Promotion Enumeration, Hardware Implications

### Slide 34: CU_TENSOR_MAP_L2_PROMOTION_L2_64B and CU_TENSOR_MAP_L2_PROMOTION_NONE

CU_TENSOR_MAP_L2_PROMOTION_NONE: Uses the default 32-byte sector fetch. If
your tensor is very sparse or the stride is massive, fetching neighbors is wasteful.
Use this to avoid "over-fetching" (wasting bandwidth on useless data).

CU_TENSOR_MAP_L2_PROMOTION_L2_64B: Fetches two adjacent 32B sectors (64
bytes total).

useful for specific half (FP16) tensor shapes where the inner dimension is exactly
64 bytes (32 elements)

### Slide 35: CU_TENSOR_MAP_L2_PROMOTION_L2_128B

Fetches the full 128-byte cache line in one shot.

Reduces DRAM command overhead by 75% (1 command instead of 4).

Alignment: Matches the standard vector load size of CUDA (ld.global.v4 = 16B x
32 threads = 512B, often split into 128B chunks).

Recommendation: This is the standard default for most dense FP16/BF16 GEMM
operations.

### Slide 36: CU_TENSOR_MAP_L2_PROMOTION_L2_256B

Fetches two full cache lines (256 bytes) in a single logical transaction.

The H100 memory controller is robust enough to handle multi-line prefetching

Requires high data density. Your application must consume this data immediately
to justify the cache space.

Highest risk of cache pollution if the data isn't used.

### Slide 37: A small but important note

Rule of Thumb:

Dense GEMM/Conv: Always use L2_128B or L2_256B. Maximize bus
efficiency.

Embedding Lookups / Sparse: Use NONE (32B). Don't fetch what you won't
touch.

Tuning Variable: The choice depends on Tensor_Inner_Dim_Bytes.

If Inner Dim < 64B, L2_128B might be wasteful. Match the promotion size to
your contiguous data width.

### Slide 38: OOBfill

oobFill helps handle out-of-bounds conditions during tensor copy operations on
H100 GPUs.

Automatically fills out-of-bound regions with zeros or NaNs in the destination, not
in HBM.

Source data in HBM remains unchanged throughout the process.

Eliminates manual boundary checks in kernels, improving code clarity and
performance.

Useful for tiling operations, especially when tiles extend beyond tensor edges.

Choice of fill depends on application needs; zero for padding, NaN for debugging.

### Slide 39: usage

The second statement uses Nan but during FMA operation its treated as zero

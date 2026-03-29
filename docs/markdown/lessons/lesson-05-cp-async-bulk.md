---
title: "Lesson 5 - `cp.async.bulk`"
lesson_number: "5"
lesson_slug: "cp-async-bulk"
instructor: "Prateek Shukla"
course: "CUDA Programming for NVIDIA H100s"
source_page: "pages/lesson-5.html"
source_slide_pdf: "H100-Course/slides/5. cp.async.bulk.pdf"
published_lesson_page: "/pages/lesson-5.html"
published_markdown_path: "/markdown/lessons/lesson-05-cp-async-bulk.md"
topics:
  - "cp.async.bulk"
  - "cp.async.bulk.tensor"
  - "Multicast"
  - "bulk_group"
  - "Cache Policy"
code_refs:
  - "fast.cu/examples/matmul/matmul_12.cu"
generated_from:
  - "pages/lesson-5.html"
  - "H100-Course/slides/5. cp.async.bulk.pdf"
---

# Lesson 5 - `cp.async.bulk`

This file combines the published lesson page with full slide-deck text so agents can fetch and search it directly.

## Sources

- Lesson page: `pages/lesson-5.html`
- Slide deck: `H100-Course/slides/5. cp.async.bulk.pdf`
- Published lesson URL: `https://cudacourseh100.github.io/pages/lesson-5.html`
- Published markdown URL: `https://cudacourseh100.github.io/markdown/lessons/lesson-05-cp-async-bulk.md`

## Lesson Summary

Lesson 5 turns the TMA descriptor into an actual instruction family. Hopper bulk copies move large regions asynchronously with much lower issue overhead than Ampere-era copy loops, and they expand the design space to include tensor-aware movement, multicast, prefetch, cache policy, bulk groups, and barrier-linked completion.

## Why This Lesson Matters

**The copy path becomes a real instruction family with its own modes, contracts, and topology.**

This lesson is the movement counterpart to the barrier and descriptor lessons. It explains how Hopper uses TMA-backed bulk instructions to move linear regions, structured tensor tiles, clustered payloads, and multicast fanout while compute keeps going.

## Topics

- cp.async.bulk
- cp.async.bulk.tensor
- Multicast
- bulk_group
- Cache Policy

## Lesson Facts

- **Course Position:** Lesson 05 of 10
- **Main Shift:** One issuing thread can launch movement for a whole block or cluster tile.
- **Companion Deck:** `5. cp.async.bulk.pdf`
- **Code Anchor:** `fast.cu/examples/matmul/matmul_12.cuh`

## Key Takeaways

- Hopper bulk copies offload address generation and large transfer work to TMA hardware.
- `cp.async.bulk` and `cp.async.bulk.tensor` solve different movement problems.
- Completion style, cache policy, and multicast topology meaningfully change kernel behavior.

## Web Lesson Content

### Why bulk copies matter

`cp.async.bulk` is Hopper's hardware-accelerated asynchronous bulk transfer family. The big idea is not just that the instruction is non-blocking. It is that the TMA hardware can own the address calculation, loop expansion, and movement of a much larger region while the SM goes back to compute.

#### Ampere `cp.async`

Uses the LSU path. The warp is still responsible for issuing many 16-byte operations and carrying the address-generation burden itself.

#### Hopper `cp.async.bulk`

Uses TMA. One thread can launch a whole tile transfer while the hardware tracks progress and updates the relevant completion object.

That matters because movement is often the long pole. Reducing issue overhead, register pressure, and instruction traffic gives the rest of the kernel more space to stay compute-focused.

### Raw and tensor-aware bulk copies solve different problems

Hopper exposes two broad shapes. The raw form is a bulk memcpy-like instruction. The tensor form is descriptor-aware and uses the tensor map plus coordinates to locate and move a structured tile.

#### Raw / linear form

```text
cp.async.bulk.dst.src.barrier_type{.cache_hint}
dst_addr, src_addr, size, mbarrier_addr;
```

Use this when memory is just a contiguous byte range. The hardware does not interpret the source as a tensor with dimensions and bounds.

#### Tensor-aware form

```text
cp.async.bulk.tensor.ndim.dst.src.barrier_type{.cache_hint}{.multicast}
dst_addr, tensor_map, coordinate_array, mbarrier_addr;
```

Adding `.tensor` makes the instruction descriptor-aware. The dimensional suffix tells the hardware how many coordinates to read, and the tensor map carries shape, stride, bounds, swizzle, and related layout information.

#### State spaces

The source and destination state spaces are explicit in the instruction shape, including `.global`, `.shared::cta`, and `.shared::cluster`.

#### Load modes

`.tile` fetches a dense multidimensional tile. `.im2col` applies a convolution-friendly transform during fetch instead of requiring a separate rearrangement kernel.

### Completion mechanisms define how consumers know the transfer is safe

The lesson makes the completion model explicit because the transfer itself is intentionally asynchronous. Hopper offers two main completion styles.

#### `mbarrier::complete_tx::bytes`

The richer coordination path. Hardware updates the barrier's transaction count as bytes arrive and flips phase when the expected work completes.

#### `bulk_group`

The lighter-weight batching model. You launch operations, commit the group, and later wait until only a limited number of recent groups remain pending.

`mbarrier` is the right model when you want explicit byte-tracked coordination between producers and consumers. `bulk_group` is simpler when coarse-grained grouped waiting is enough.

> **Important distinction:** `bulk_group` is about batching async operations. `mbarrier` is about tracking work and phase completion with a reusable synchronization object.

### Cache policy and prefetch shape how bulk movement interacts with L2

Bulk copies can flood L2 with one-use traffic if the kernel does not communicate reuse intent. Hopper lets the instruction attach L2 cache hints through a cache policy descriptor.

| Policy | Meaning | Typical fit |
| --- | --- | --- |
| `evict_first` | Mark lines as early eviction candidates. | Streaming or one-pass data, and final-output stores. |
| `evict_last` | Mark lines as persistent for longer retention. | Reused weights or tiles likely to be consumed repeatedly. |
| `evict_normal` | Default behavior. | When reuse is unclear or mixed. |

```text
createpolicy.fractional.L2::evict_last.b64 policy_reg, 1.0;
createpolicy.fractional.L2::evict_first.b64 policy_reg, 1.0;
```

`cp.async.bulk.prefetch` and its tensor-aware variant can also prefetch data into L2 as a latency hint, but the lesson is clear that prefetch is not a correctness primitive. If the later main transfer arrives before the data is cached, hardware will fetch from HBM anyway.

### Multicast turns one HBM fetch into cluster-wide delivery

A recurring Hopper use case is that multiple CTAs in a cluster need the same operand tile. In GEMM-like workloads, many blocks may need the same slice of matrix A while each consumes a different slice of matrix B. Multicast exists to avoid paying for that same HBM fetch repeatedly.

1. A leader thread issues the multicast TMA instruction.
2. The data is fetched once from global memory into L2.
3. The cache/controller fabric broadcasts it to the shared-memory destinations of the CTAs selected by the mask.
4. The relevant barrier state is updated for each participating CTA.
5. Consumers in each CTA wait on their local barrier and resume when the tile is valid.

The notes emphasize that the barrier object is replicated at the same relative shared-memory offset in each participating CTA. Only CTAs included in the mask should behave like receivers for that transfer.

> **Practical consequence:** multicast is a topology feature, not just a copy qualifier. The mask, barrier placement, and cluster participation all have to agree or the synchronization contract breaks.

#### TMA stores and reductions

The lesson also points out that structured stores can use `bulk_group` completion, and Hopper extends the offload idea further with `cp.reduce.async`, where the TMA path can own bulk reduction-style accumulation work too.

### Practical guidance

1. **Use raw bulk copies when memory is just bytes.** Use tensor copies when shape, stride, and bounds are part of the problem.
2. **Choose the completion model intentionally.** `mbarrier` and `bulk_group` are different coordination tools.
3. **Pick cache policy based on reuse.** Streaming traffic and persistent weights want opposite hints.
4. **Treat multicast as a cluster contract.** Mask membership, barrier offsets, and receiver logic have to line up.
5. **Remember that prefetch is only a hint.** It can reduce latency, but it does not guarantee residency.

#### Glossary

| Term | Definition |
| --- | --- |
| `cp.async.bulk` | Hopper's asynchronous bulk copy instruction family for large transfers. |
| `cp.async.bulk.tensor` | The descriptor-aware structured tensor-copy form. |
| `bulk_group` | A batching and grouped-wait mechanism for bulk async operations. |
| Multicast | Cluster-wide fanout of one fetched payload to multiple CTA destinations. |
| Cache policy descriptor | A 64-bit object created with `createpolicy` to carry eviction hints. |
| `cp.async.bulk.prefetch` | An L2 prefetch hint for future bulk transfers. |

### Continue the course

Lesson 5 defines the main async movement family. Lesson 6 pivots to the compute side of the same pipeline: WGMMA, warpgroups, descriptor-backed tensor-core issue, and the register or shared-memory operand paths Hopper uses for large asynchronous matrix math.

## Full Slide Deck Text

Extracted from `H100-Course/slides/5. cp.async.bulk.pdf` with `pdftotext -layout`. Total slides: 37.

### Slide 1: cp.async.bulk

Prateek Shukla

### Slide 2: cp.async.bulk operations

cp.async.bulk is a family of PTX instructions for hardware-accelerated, asynchronous
bulk memory transfers on Hopper H100 GPUs.

These operations are offloaded to dedicated hardware and execute independently of
the SM's compute pipeline, allowing computation to proceed while data transfers
happen in the background

cp.async.bulk can handle large, multi-dimensional tensor transfers efficiently from 1D
to 5D tensors containing hundreds or thousands of bytes.

cp.async.bulk operations require barrier objects for coordination, ensuring proper
ordering between asynchronous transfers and compute operations

### Slide 3: cp.async.bulk vs cp.async(ampere)

cp.async in ampere
Uses the Load/Store Unit (LSU)
It is "asynchronous" because the thread issues the instruction and moves on, but
the thread is still responsible for calculating the address and issuing the command
for every 16 bytes of data.
cp.async.bulk in hopper
Uses the Tensor Memory Accelerator (TMA)
A single thread issues one instruction to copy an entire tile and the TMA handles
all the address calculations, loop unrolling, and movement in the background

### Slide 4: Some more key differences

cp.async (ampere)
To copy a 4KB tile of data, every thread in the warp has to loop and issue multiple
cp.async instructions. This burns register cycles and instruction cache.
Uses cp.async.commit_group and wait_group
cp.async.bulk (hopper)
One single thread can initiate the transfer for the whole block. The other 31
threads in the warp (or 127 in the block) effectively do nothing related to memory
copy initiation.
With mbarrier The TMA hardware updates the barrier's "transaction count"
automatically as bytes arrive

### Slide 5: Layouts of cp.async.bulk

1.   The tensor layout

This is the layout used when you have a TMA descriptor (Tensor Map) and want to
copy a specific multidimensional tile.
cp.async.bulk.tensor.ndim.dst.src.barrier_type{.cache_hint}{.multicast}

dst_addr, tensor_map, coordinate_array, mbarrier_addr;

2.   The Raw Layout (Linear)

This is the layout used for simple, contiguous byte copies
cp.async.bulk.dst.src.barrier_type{.cache_hint}

dst_addr, src_addr, size, mbarrier_addr;

### Slide 6: cp.async.tensor

Adding .tensor in the instruction specifies that the operation is tensor-aware,
meaning the instruction works on multidimensional tensor data rather than just flat
arrays

This opens up a bunch of other really important options which we can set in the
instruction. This is also what enables the use of cuTensorMap

### Slide 7: cp.async.bulk.tensor.{1d,2d,3d,4d,5d}

1d/2d/3d tells the hardware how many numbers (indices) it needs to read from you
to locate the tile.

So basically once we get the n number of indices which are required to locate the
tile in the tensor then we can grab the tile as per the information given in
cuTensorMap

Nd in here means the tensor have N dimensions

We can have tensors with up to 5 dimensions

### Slide 8: The state space of source and destination

In cp.async.bulk.tensor.{dim}.{space1}.{space2}

space represents the state space of the source tensor

space2 represents the state space of the destination tensor

These can be anything in ones which we discussed, global, shared::cta,
shared::cluster etc

### Slide 9: The load mode {.tile, .im2col, .im2col::w ....:}

This modifier is critical because it tells the TMA hardware how to interpret the
coordinates you provide and how to transform the data on the fly

.tile - The TMA uses the coordinates provided in tensorCoords to calculate a
base address and follows the strides defined in the tensorMap to fetch a dense,
contiguous multi-dimensional box (a tile) of data

### Slide 10: im2col

It performs a hardware-accelerated im2col transformation during the fetch. In past
you had to write a kernel to copy pixels from an Image layout (N,C,H,W) into a
Column layout (Matrix). This wasted memory bandwidth and register space

Now you can just give the coordinates of the top-left corner of a convolution
window and the TMA grab the pixels required for the filter, unroll them, and place
them in L2 as if they were a column of a matrix, this speeds up the operation.

### Slide 11: The completion mechanism

cp.async.bulk.tensor.{}.{}.{}.completion_mechanism

We have two completion mechanisms
- mbarrier::complete_tx::bytes
- .bulk_group

### Slide 12: mbarrier*:complete_tx*:bytes

The cp.async.bulk call takes the mbarrier handle (likely in [mbar] parameter) and
the completion mechanism specification. The hardware tracks this operation and
automatically updates the barrier's tx-count as data moves

When tx-count hits zero, the barrier phase flips and waiting threads are released

You need a pointer to the mbarrier and the size of the copy operation as the
operands for this operation

### Slide 13: bulk_group

bulk_group is a much more simpler and lightweight alternative to mbarrier.

Instead of tracking the tx_count, you issue a bunch of copy operations using
bulk_group and then then batch them together using commit_group then wait on it
using wait_group until only N or less of the most recent bulk async-groups are
pending.

### Slide 14: L2 cache hinting

cp.async.bulk.tensor.1d.shared*:cta.global.mbarrier*:complete_tx*:bytes.L2*:cache_hint

During our async copies if we let the data flood the L2 cache it will kick out other
important cached data which you might be using repeatedly. To prevent one use
data to take useful space from L2 cache we use L2 hinting
There are different ways to implement it for loads and stores
evict_first is a way to tell the hardware to throw the data immediately to save the
cache space
evict_last is a way to tell the hardware to mark the data persistent and keep it as
long as possible
evict_normal is the default behavior

### Slide 15: L2 hinting loads(global -> shared/dsmem)

During the global -> shared transfers data is loaded and cached into L2 according
to write back cache policy.

For Reused Weights/Data (Best): Use evict_last. This marks the data as
"persistent," telling the L2 cache to keep it as long as possible.

For Streaming/One-Pass Data: Use evict_first. Because essentially you are using
the data once and then throwing it away

### Slide 16: L2 hinting stores (shared -> global)

When writing results back to global memory, you typically don't need to read them
again immediately.

For Final Output (Best): Use evict_first. You are writing data to HBM and likely
won't read it back on this SM.

This minimizes cache pollution. The data hits L2 (to serve the write) but is
immediately marked as the first candidate for eviction, keeping the cache clean for
your inputs

### Slide 17: Creating a cache policy descriptor

The createpolicy instruction in PTX creates a 64-bit cache policy descriptor that
encodes eviction priorities for specific memory access patterns. Here's how it
works:

A cache policy descriptor is 64 bit object which determines the cache policy
createpolicy.fractional.L2*:evict_last.b64 policy_reg, 1.0;

createpolicy.fractional.L2*:evict_first.b64 policy_reg, 1.0;

dest is the 64 bit register which will hold the cache policy descriptor

fraction(1.0) determines to what fraction of data policy is applied

### Slide 18: Copying unstructured data vs structured (tensors) data

cp.async.bulk (Unstructured) is a hardware-accelerated memcpy. It treats memory
as a linear stream of bytes. It does not "know" that your data is a matrix, a 3D
volume, or a tiled tensor.

We use cp.async.bulk.shared/cp.async.bulk.global to copy unstructured
data

cp.async.bulk.tensor (Structured) is a smart, descriptor-based copy. It relies on a
CUtensorMap object (opaque handle) created on the host. The hardware
"understands" the dimensions, strides, and boundaries of your data.

We use cp.async.bulk.tensor to copy structured tensors, cuTensorMap comes
to play here

### Slide 19: Unstructured copies

### Slide 20: Operands:

dstmem - the destination address in the shared memory

srcMem - the source in shared memory

size - size of the data transferred in bytes

mbar - the pointer to the mbarrier object which you are using

cache_policy - 64 bit pointer to the policy descriptor

### Slide 21: Operands:

Dst: the destination address in the global

Src: The source from where the copy is done

Size: size of transaction

Cache_policy: the descriptor for the cache policy

Mask: for masked writes, specifies which bytes to write for the destination

### Slide 22: dst - the destination address in the dsmem

src - the source address in the shared memory
size - copy size
completion_mechanism - mbarriers
The point here is that you are not transferring data to the shared memory allocated
to some other block but you are moving data to a location in distributed shared
memory which is pooled by all the blocks

### Slide 23: Origins of multicast

Computing data is quicker than moving data. Especially moving data from HBM is
really time and energy intensive. Computation can be done a lot more quicker

A lot of deep learning applications use gemms and in gemm many different blocks
need to read the same chunk of Matrix A to multiply it against their specific chunk of
Matrix B. If SM0, SM1, SM2, and SM3 all need "Tile A0".

In the previous generations if you had 4 SMs that all needed the same piece of data
(a weight matrix for a neural network layer), all 4 SMs had to individually request
that data from the global memory (which was really expensive).

Why not fetch the data once from HBM and copy it to all 4 SMs simultaneously as it
flows through the wire?

### Slide 24: Breakdown of the operation

The TMA reads size bytes from Global Memory (srcMem) into the L2 cache. The cache hint
tells L2 to persist this line, optimizing bandwidth for subsequent accesses by other waves.

The L2 cache controller reads the data once and broadcasts it over the cluster crossbar,
simultaneously targeting the SMEM banks of every block defined in the multicast mask.

The L2 cache controller reads the data once and broadcasts it over the cluster crossbar,
simultaneously targeting the SMEM banks of every block defined in the multicast mask.

Upon completion, the TMA uses the mbar pointer (also multicast-encoded) to atomically add
size bytes to the transaction count of the mbarrier in every participating block simultaneously

single leader thread issues this non-blocking instruction. The TMA hardware manages the
entire fetch-broadcast-signal pipeline independently, allowing threads in all blocks to compute
or sleep while waiting

### Slide 25: Mbarriers and multicast

Its really easy to shoot yourself in the foot here because there are things to
understand about using mbarriers for specific thread blocks in the clusters.

the mbarrier object is replicated at the same relative memory offset within the
Shared Memory of every participating CTA. When the producer issues the TMA
instruction, the hardware broadcasts data to all CTAs in the ctaMask and
automatically signals the mbarrier at that specific address in each destination CTA

The hardware knows the group members immediately from the instruction's mask.
Receiver CTAs do not need to 'arrive' to form the group; they simply wait on their
local mbarrier instance to know when valid data has landed

### Slide 26: The first thread (or any single thread) of all the blocks in the cluster call expect_tx

with the amount of data they are expecting and they all are pointing to the mbarrier in
their own shared memory. The way cp.async tracks the barrier is by using offset to
get to the mbarrier for each block

A single thread in the whole block calls cp.async.bulk with multicast and the mask
for all the blocks in that clusters, it points to its own barrier

If you want the data to be transferred to blocks 0 and 3 and not in block 1 and 2 then
you must not call mbarrier from block 1 and 2 and put them in mask

All consumer warps in all the blocks spin on their local barrier mbarrier.try_wait

The arrival counts are managed by the TMA itself for the masked blocks. TMA
automatically signals "remote arrival" to the mbarrier at the specified offset

### Slide 27: A small note

You need to manually manage the shared memory layout when using dynamic
shared memory

Instead of doing this - extern *_shared*_ char smem[]; and the adding mbarrier

Use this - uint64_t* bar_ptr = reinterpret_cast<uint64_t*>(smem);
int tma_alignment = 128;

int data_offset = (sizeof(uint64_t)+ tma_alignment - 1) & ~(tma_alignment - 1);

half* tile_ptr = reinterpret_cast<half*>(smem + data_offset);

### Slide 28: All participating CTAs (say, 16 CTAs running on 16 different SMs within the same cluster) must first establish themselves as a multicast

group. Each CTA allocates the same mbarrier object in shared memory and calls mbarrier.init.shared.b64 with an arrival count, then executes
arrive() on that mbarrier. This arrival isn't just synchronization-it's hardware registration. The memory subsystem now knows these 16 CTAs
form a logical group that will receive identical data.

Every CTA in the multicast group creates an identical CUtensorMap descriptor. On the host, you call
make_tma_copy(SM90_TMA_LOAD_MULTICAST{}, gmem_tensor, smem_layout, cluster_size) which encodes the tensor geometry, data
type, swizzling pattern, and critically, the cluster dimensions. This descriptor is passed to the kernel (marked __grid_constant__) and tells
TMA what global memory region to fetch and where in each CTA's shared memory to place it. All 16 CTAs must use this exact same
descriptor-any deviation breaks the contract that they want the same data.

Each CTA (typically from a single elected thread per CTA) issues the TMA multicast instruction:
cp.async.bulk.tensor.shared.cluster.global.mbarrier.multicast. Notice the .cluster scope and .multicast qualifier-these signal hardware intent.
The instruction takes the shared memory destination address, the tensorMap pointer, tensor coordinates, the mbarrier pointer, and crucially, a
ctaMask parameter. The ctaMask is a 16-bit bitmask (for a cluster size of 16) where bit i indicates whether CTA i participates-for example,
0xFFFF means all 16 CTAs receive the data.

Here's where the magic happens at the L2 cache level. The L2 cache controller receives 16 seemingly independent requests for the same
tensor tile, all tagged with the same mbarrier group ID. The hardware recognizes this pattern and promotes one request as the leader. That
leader request triggers a single read from HBM (let's say 1MB of weight data). As data streams into the L2 cache, the cache controller doesn't
just send it to one SM-it multicasts (simultaneously forwards) the data to all 16 participating SMs' L1 caches and directly into their shared
memory regions. You pay 1MB of HBM bandwidth but deliver 16MB worth of data across the SMs.

The TMA hardware also automatically decrements the transaction byte count (tx-count) on each CTA's mbarrier as data arrives, tracking
completion progress.

After issuing the TMA multicast, each CTA executes wait_barrier(tma_load_mbar, phase) or the equivalent mbarrier.try_wait in PTX. This
blocks until the mbarrier's transaction count reaches zero-meaning all expected bytes have been delivered to that CTA's shared memory.
Once all CTAs pass this barrier, they're guaranteed their shared memory is populated with the data, and compute can begin

### Slide 29: Dst - the destination address

Src - the global memory pointer
Size - the size of the operation
Mbar - the pointer to mbarrier
Ctamask - the 16 bit mask for multicasting
Cache-policy - the cache policy

### Slide 30: Structured copies

### Slide 31: Copy data from global to cta

Operands -

dstMem - pointer to shared memory location

tensorMap, tensorCoord - address to tensorMap, array of box coordinates

srcMem - pointer to the global memory address

cache-policy - pointer to the cache descriptor

### Slide 32: Operands -

dstMem - pointer to dsmem
tensorMap, tensorCoord - address to tensorMap, 1d vector telling the coordinates
mbar - the pointer to mbarrier object
ctaMask - Mask for selecting the blocks in which we have to copy
cache-policy - pointer to the cache descriptor

### Slide 33: TMA stores using bulk_group

tensorMap, tensorCoords - 64 bit pointer to the cuTensorMap, array of box
coordinates

srcMem - the memory location from which data is coming from

Cache-policy - pointer to the cache policy descriptor

### Slide 34: cp.async.bulk.prefetch

We can prefetch the data to L2 cache for lower latency

We use cp.async.bulk.prefetch.tensor to do that

### Slide 35: A few important points

cp.async.bulk.prefetch is a performance hint, if you launch cp.async.bulk.prefetch
and then immediately launch cp.async.bulk.tensor and the data isn't cached then
the device will fetch the data from HBM anyways.

Even though you use the same tensorMap for both the main cp.async op and the
prefetch operation there is no effect of the L2 promotion or the swizzling to the
way to data is cached in L2

### Slide 36: cp.reduce.async

It offloads the atomic accumulation of an entire data tile from the SM to the TMA

This is literally the work which TMA does with that instruction. Following are two
versions

cp.reduce.async.bulk.dst.src.completion_mechanism{.level::cache_hint}.redOp.typ
e [dstMem], [srcMem], size{, cache-policy}

cp.reduce.async.bulk.tensor.dim.dst.src.redOp{.load_mode}.completion_mechanis
m{.level::cache_hint} [tensorMap, tensorCoords], [srcMem] {,cache-policy}

### Slide 37: The redOps and the data types allowed

.add                         .f16

.min/.max                    .bf16

b32
.inc/.dec
u32
.and
s32
.or
b64
.xor
u64

s64

f32

f64

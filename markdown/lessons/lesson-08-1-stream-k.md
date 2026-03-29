---
title: "Lesson 8.1 - Stream-K"
lesson_number: "8.1"
lesson_slug: "stream-k"
instructor: "Prateek Shukla"
course: "CUDA Programming for NVIDIA H100s"
source_page: "pages/lesson-8.1.html"
source_slide_pdf: "H100-Course/slides/8.1 Stream-K.pdf"
published_lesson_page: "/pages/lesson-8.1.html"
published_markdown_path: "/markdown/lessons/lesson-08-1-stream-k.md"
topics:
  - "Stream-K"
  - "Fixup"
  - "Persistent Scheduling"
  - "Groups"
  - "L2 Locality"
code_refs:
  - "sm90_tile_scheduler_stream_k.hpp"
  - "sm90_tile_scheduler.hpp"
  - "sm90_tile_scheduler_group.hpp"
generated_from:
  - "pages/lesson-8.1.html"
  - "H100-Course/slides/8.1 Stream-K.pdf"
---

# Lesson 8.1 - Stream-K

This file combines the published lesson page with full slide-deck text so agents can fetch and search it directly.

## Sources

- Lesson page: `pages/lesson-8.1.html`
- Slide deck: `H100-Course/slides/8.1 Stream-K.pdf`
- Published lesson URL: `https://cudacourseh100.github.io/pages/lesson-8.1.html`
- Published markdown URL: `https://cudacourseh100.github.io/markdown/lessons/lesson-08-1-stream-k.md`

## Lesson Summary

Lesson 8.1 is the scheduler addendum that makes persistent kernels feel complete. Instead of assigning one whole output tile to one CTA and accepting a weak tail wave, Stream-K slices the remaining K work into a balanced tape, tracks split ownership explicitly, and pays the reduction cost only where the tail actually needs it.

## Why This Lesson Matters

**The unit of scheduling shifts from output tiles to a controlled budget of K work.**

Kernel design already introduced persistent scheduling. This addendum isolates the tricky part: how to eliminate the half-empty tail wave without turning the entire problem into expensive split-K fixup.

## Topics

- Stream-K
- Fixup
- Persistent Scheduling
- Groups
- L2 Locality

## Lesson Facts

- **Course Position:** Lesson 08.1, between Kernel Design and Multi GPU
- **Main Shift:** Some CTAs stop owning complete tiles and start owning math ranges inside a tile.
- **Companion Deck:** `8.1 Stream-K.pdf`
- **Code Anchors:** `sm90_tile_scheduler_stream_k.hpp`, `sm90_tile_scheduler.hpp`, `sm90_tile_scheduler_group.hpp`

## Key Takeaways

- Stream-K is a tail-balancing scheduler, not the default answer for every tile in the problem.
- Each split tracks where it starts in K, how much K it owns, and whether it is responsible for the final epilogue.
- Groups recover L2 locality after the 1D K-work tape would otherwise scatter cooperating CTAs across the output space.

## Web Lesson Content

### Why Stream-K exists

The problem is not correctness. Plain data-parallel persistent scheduling is correct. The problem is utilization. If the output tile count is not a clean multiple of available SMs, the last wave leaves a large fraction of the machine idle while only a handful of CTAs finish the tail.

#### Tile-parallel view

One CTA owns one output tile. That is clean and cheap until the last wave is sparse and many SMs are waiting for a few final tiles to retire.

#### Stream-K view

The remaining work is treated as a tape of math iterations along K. The tail tile can be split so every SM finishes closer to the same time.

The notes are careful that this is usually a hybrid strategy. Early waves often stay purely data-parallel because that has the lowest overhead. Stream-K is most valuable when it targets the tail wave where the decomposition mismatch is doing visible damage.

> **Useful heuristic from the notes:** if the tail is still mostly full, it can be better to keep the normal scheduler and avoid unnecessary split-K reduction overhead. Stream-K is most attractive when the tail is meaningfully under-filled.

### Work units and splits make the scheduler explicit

For one output tile, GEMM is the sum of many K tiles. If a CTA computes all of them, no reduction is needed. If multiple CTAs each compute a fraction of those K tiles, each CTA owns a split of that output tile and the result needs fixup.

| Field | Meaning | Why it matters |
| --- | --- | --- |
| `M_idx`, `N_idx`, `L_idx` | The output tile coordinates. | Tell the CTA which C tile it contributes to. |
| `K_idx` | Where this split begins inside the output tile's K dimension. | Distinguishes the first split from middle or final splits. |
| `k_tile_count` | How many K tiles this split computes. | Tells you how much math this CTA actually owns for that output tile. |
| `k_tile_remaining` | How much of the work unit is still left to process. | Matters because one CTA can span multiple splits as it walks the work tape. |

A split is simply one CTA's contribution to one output tile. One CTA can end a tile and begin the next one inside the same assigned range of math work, which is why the scheduler tracks both tile coordinates and the exact K start within the tile.

```text
// Conceptual split state
tile: (M_idx, N_idx, L_idx)
K_idx: where this CTA starts in the tile's K dimension
k_tile_count: how much K this CTA computes

is_final_split = (K_idx + k_tile_count) == k_tiles_per_output_tile
```

### The first, middle, and final split roles fall directly out of that state

Once a CTA knows its `K_idx` and `k_tile_count`, its role is no longer mysterious. The slides describe three cases, and the Stream-K scheduler header in this repo mirrors them with helpers like `is_final_split(...)` and `compute_epilogue(...)`.

#### First split

`K_idx == 0`. Nobody has written workspace for this tile yet, so the first split initializes the partial result.

#### Middle split

Owns a K range strictly inside the tile. It reduces its partials into existing workspace and does not own the final epilogue.

#### Final split

Covers the end of the K dimension. It waits for prior splits, loads their partials, adds its own, and then owns the final epilogue.

This is the real conceptual shift. A CTA can still be persistent and still follow a deterministic work loop, but it is no longer guaranteed to own a complete output tile from start to finish.

> **Backward iteration helps the final split:** the notes emphasize that workers often iterate through shared tiles in reverse K order so the ending split reaches the fixup point later, reducing how long it has to wait for the earlier split to finish.

### The workspace and lock protocol are the cost of balancing the tail

Split ownership only works because partial accumulators can be staged in global workspace and progress can be tracked with a lock. The notes describe that lock as an integer per output tile that monotonically encodes how much K work has already been completed and published.

| Mechanism | Purpose | Effect on execution |
| --- | --- | --- |
| Reduction workspace | Stores partial accumulators for tiles split across CTAs. | Creates the place where later splits can load and reduce prior work. |
| Lock / progress counter | Records how much K work has already been completed for the tile. | Lets later splits know whether they can proceed with deterministic or opportunistic reduction. |
| Separate reduction units | Allow reduction and epilogue work to be modeled explicitly for some scheduler modes. | Decouples math ownership from final fixup ownership when the scheduler decides that is beneficial. |

The Stream-K header in this repo exposes both deterministic and non-deterministic reduction modes. The deterministic path waits for the exact cumulative K progress it expects. The non-deterministic path is looser and cares mainly that the workspace has been initialized before middle splits race to reduce into it.

```text
// Conceptual fixup flow
first split   -> store partials, publish progress
middle split  -> wait until workspace is valid, reduce partials, publish progress
final split   -> wait for prior progress, load-add all required partials, run epilogue
```

### Groups recover locality after the 1D work tape scatters CTAs

A naive Stream-K decomposition balances work but destroys the nice spatial wave pattern that helps L2. Groups are the locality repair mechanism. They partition Stream-K units into sub-groups so cooperating workers stay closer to the same region of output space and reuse more useful cache state.

#### Base persistent scheduler

Keeps the standard tile-parallel persistent loop, swizzle, and raster order. It is the cheap baseline when split-K fixup is not needed.

#### Grouped persistent scheduler

Extends persistence across multiple grouped GEMM problems, treating them as one long linear tile space while preserving per-group swizzle and locality metadata.

#### Stream-K scheduler

Adds split tracking, reduction ownership, and locality groups so the tail can be balanced without giving up the reuse story completely.

| File | Role in the story |
| --- | --- |
| `sm90_tile_scheduler.hpp` | The base static persistent scheduler and swizzle machinery. |
| `sm90_tile_scheduler_group.hpp` | The grouped persistent extension for multiple GEMM problems in one launch. |
| `sm90_tile_scheduler_stream_k.hpp` | The split-K, fixup, reduction, and grouped-locality scheduler used for the Stream-K path. |

The notes also mention HyTiS as an alternative way to fight wave quantization. Its point is useful even if you do not adopt it: Stream-K is not the only answer. It is one answer that trades extra reduction machinery for better tail utilization when K is the natural dimension to slice.

### Practical guidance

1. **Use Stream-K for the mismatch, not for its own sake.** The goal is to fix the tail wave, not to split every tile when the normal persistent scheduler is already efficient.
2. **Track split ownership explicitly.** `K_idx`, `k_tile_count`, and final-split status are the mechanism that decides whether a CTA stores, reduces, or runs the epilogue.
3. **Remember that fixup is a real cost.** Workspace traffic, locks, and cross-CTA reduction are why hybrid policies often outperform applying Stream-K everywhere.
4. **Protect locality after balancing.** Groups matter because a pure 1D work tape can solve utilization while quietly destroying L2 reuse.
5. **Read the scheduler files as a family.** The base persistent scheduler, grouped scheduler, and Stream-K scheduler are three connected answers to the same utilization problem.

#### Glossary

| Term | Definition |
| --- | --- |
| Split | One CTA's contribution to one output tile when the tile is divided along K. |
| Fixup | The reduction of partial accumulators from multiple CTAs into one final output tile. |
| Final split | The split whose K range reaches the end of the tile and therefore owns the final epilogue. |
| Separate reduction | A scheduler mode where reduction and epilogue work can be modeled as distinct work units. |
| Group | A locality-preserving subset of Stream-K units that cooperates on its own portion of the work tape. |

### Continue the course

The scheduler addendum finishes the single-node kernel story. The next lesson moves out of one GPU and into the node and cluster fabrics that make large H100 training systems possible in the first place.

## Full Slide Deck Text

Extracted from `H100-Course/slides/8.1 Stream-K.pdf` with `pdftotext -layout`. Total slides: 10.

### Slide 1: Stream-K

Prateek Shukla

### Slide 2: The first principles

For one output tile C_tile, GEMM computes:

C_tile = sum over K-tiles of (A_tile_k * B_tile_k)

If one CTA computes all K-tiles for that output tile, no inter-CTA reduction is
needed.

If multiple CTAs each compute only a subset of K-tiles, each CTA produces a
partial sum and those partial sums must be merged.

In this scheduler, that merge step is called fixup.

### Slide 3: The unit of work

each work unit includes:

- tile coordinates (m_idx, n_idx, l_idx)

- k_idx: start k-tile within the output tile

- k_tile_count: number of k-tiles this work unit computed for this output tile

reduction need is decided by:

- full tile in k: k_tile_count == k_tiles_per_output_tile -> no reduction needed

- partial tile in k: k_tile_count != k_tiles_per_output_tile -> reduction needed

that is the key predicate behind requires_fixup(...).

### Slide 4: Split

A single CTA's assigned range of k-tile iterations may land in the middle of an output
tile. For example, with 3 output tiles of 90 k-tiles each and 4 CTA units:

A "split" is a CTA's contribution to a single output tile. Unit 1 above has two splits one
for the tail of tile 0 and one for the head of tile 1. The code processes these one at a
time (that's the k_tile_remaining loop in advance_to_next_work).

### Slide 5: Tracking the triplet

For a single split (one CTA's work on one output tile):

K_idx: Where this split starts within the output tile's K dimension.

K_idx = tile_iter_start - output_tile_iter_start.

For Unit 0's work on tile 0, K_idx = 0. For Unit 1's work on tile 0, K_idx = 67.

k_tile_count: How many k-tiles this split processes. For Unit 0 on tile 0, that's 67.
For Unit 1 on tile 0, that's 23 (= 90 - 67).

is_final_split(): (K_idx + k_tile_count) == k_tiles_per_output_tile. True
when this split covers the end of the K dimension. Unit 1's split on tile 0 is a final split
(67 + 23 = 90).

### Slide 6: The three roles follow directly

Given that an output tile might have 2-4 CTAs each computing a portion of K:

K_idx == 0 -> you're the first split. You computed k-tiles [0, N). Nobody wrote to
workspace before you. Just store.

is_final_split() == true and not separate reduction -> you're the final split and
epilogue owner. You computed k-tiles [X, 90). Wait for everyone before you, load
their accumulated result, add yours, run the epilogue.

Everything else -> you're a middle split. You computed k-tiles [A, B) where 0 < A < B
< 90. You need to reduce your partials into what's already in workspace.

### Slide 7: The lock: what it is physically

There's a contiguous array of ints in global memory, one per output tile (times
num_barriers for multi-warp-group kernels).
The pointer to this array sits right after the reduction data buffer in the same
allocation. Every lock starts at 0 when kernel launches.
The lock is a single integer that encodes how much K-dimension work has been
completed and written to workspace for a given output tile. In normal (non separate
reduction) mode, it counts cumulative k-tiles processed. It only ever increases.
The lock encodes progress in K-tile space. For deterministic mode, each split waits
for the exact cumulative K-tile count that matches its starting position, enforcing a
strict left-to-right reduction order. For non-deterministic mode, middle splits just
need to know the workspace has been initialized (lock >= 1), and they race to
atomically reduce into it.

### Slide 8: Groups

Groups are an L2 cache locality optimization. They partition the stream-K units into
G independent sub-groups, where each group collaborates only on its own subset
of stream-K tiles. This is an optimization specific to stream-K as stream-K breaks
the nice wave-based rasterization pattern because a single CTA might span tiles
from different regions of the output grid, destroying that locality.
Without groups (G=1), all stream-K units share one big pool of K tiles across all
stream-K output tiles. Unit 0 might work on tile 0 and tile 1, while unit 7 works on tile
5 and tile 6 completely different spatial positions. Their data in L2 cache doesn't
overlap at all.
With groups, units 0 in each group will compute identical K extents of tiles that
would be assigned in the same wave according to the rasterization order of the
data-parallel formulation

### Slide 9: Group hierarchy

Groups (up to 8, for L2 locality)

Each group contains multiple cluster-tiles

Each cluster contains multiple CTAs (thread blocks)

Each CTA processes K-tiles

The grouping is determined along the rasterization dimension. For example, if you
rasterize along M and have problem_blocks_m / cluster_m = 4, you'd get 4
groups. Groups are interleaved in the output space. Final output_tile_id is
computed as:
output_tile_id = (output_tile_id_in_group * num_groups) + group_idx

### Slide 10: HyTiS

HyTiS solves wave quantization by making the partial wave use finer-grained tiles so
more SMs stay busy. It's a purely spatial decomposition (MxN) with heterogeneous
tile sizes, vs Stream-K's K-dimension decomposition with homogeneous tile sizes.
Instead of using stream-k, HyTiS uses two different tile sizes in one kernel launch:
- Large tiles (e.g., 128x256) for full waves -> max throughput
- Small tiles (e.g., 64x64) for the partial wave -> min latency
No reduction, no workspace, no barriers, no fixup
The tradeoff is that HyTiS can't help when the problem is small in M and N but large
in K

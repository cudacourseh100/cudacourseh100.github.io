---
title: "Lesson 3 - Asynchronicity and Barriers"
lesson_number: "3"
lesson_slug: "asynchronicity-and-barriers"
instructor: "Prateek Shukla"
course: "CUDA Programming for NVIDIA H100s"
source_page: "pages/lesson-3.html"
source_slide_pdf: "H100-Course/slides/3. Asynchronicity and barriers.pdf"
published_lesson_page: "/pages/lesson-3.html"
published_markdown_path: "/markdown/lessons/lesson-03-asynchronicity-and-barriers.md"
topics:
  - "Latency Hiding"
  - "mbarrier"
  - "fence.proxy.async"
  - "wait_group"
  - "barrier.cluster"
code_refs:
  - "sm90_gemm_tma_warpspecialized_pingpong.hpp"
generated_from:
  - "pages/lesson-3.html"
  - "H100-Course/slides/3. Asynchronicity and barriers.pdf"
---

# Lesson 3 - Asynchronicity and Barriers

This file combines the published lesson page with full slide-deck text so agents can fetch and search it directly.

## Sources

- Lesson page: `pages/lesson-3.html`
- Slide deck: `H100-Course/slides/3. Asynchronicity and barriers.pdf`
- Published lesson URL: `https://cudacourseh100.github.io/pages/lesson-3.html`
- Published markdown URL: `https://cudacourseh100.github.io/markdown/lessons/lesson-03-asynchronicity-and-barriers.md`

## Lesson Summary

Hopper only works as the asynchronous machine if overlap stays correct. Lesson 3 is the synchronization lesson: proxy separation, RAW and WAR hazards, release and acquire ordering, `mbarrier`, cluster-wide coordination, and the wait rules that keep tensor data valid instead of merely in flight.

## Why This Lesson Matters

**Completion is no longer enough. Visibility and ordering become part of the design.**

The lesson moves from "launch async work" to "make async work safe." That means reasoning about who is touching memory, when data becomes visible across proxies, and how shared-memory ownership flips across producer and consumer stages without corrupting the tile pipeline.

## Topics

- Latency Hiding
- mbarrier
- fence.proxy.async
- wait_group
- barrier.cluster

## Lesson Facts

- **Course Position:** Lesson 03 of 10
- **Main Shift:** Correct overlap requires explicit synchronization contracts.
- **Companion Deck:** `3. Asynchronicity and barriers.pdf`
- **Code Anchor:** `sm90_gemm_tma_warpspecialized_pingpong.hpp`

## Key Takeaways

- The whole point of asynchronicity is latency hiding, not "doing things later" in the abstract.
- Proxy separation means normal thread-local ordering is not enough for async copies and tensor work.
- `mbarrier` tracks work and ownership across reusable phases, not just thread arrival.

## Web Lesson Content

### Why asynchronicity matters

Systems alternate between **doing** and **fetching**. In a synchronous world those happen one after another. In an asynchronous world, the request for data is decoupled from the moment its result is consumed. That gap is where latency hiding lives.

#### Blocking / synchronous

The launching thread pauses until the operation finishes. Control does not return, so no useful work overlaps the wait.

#### Non-blocking / asynchronous

The launching thread regains control immediately while some other part of the system tracks the work in the background.

On GPUs math is fast and memory movement is slow. Hopper leans into this: TMA and Tensor Cores are designed so compute continues while data movement and completion tracking happen elsewhere.

> **Mental model:** the goal is not to maximize async instructions. The goal is to keep expensive units busy while slower operations complete in parallel.

### Proxies, the async lifecycle, and why hazards appear

The async lifecycle is three stages: initialize async work, let another part of the system track it while unrelated work continues, then synchronize where the result is needed. Hopper's twist: the part that issues work and the part that performs it may live in different **proxies**.

#### Generic proxy

Ordinary thread-issued loads and stores. Ordering within this path is what you expect from sequential instructions inside one thread.

#### Async proxy

Hardware paths like `cp.async.bulk` and `wgmma`. Once launched, they run independently of the issuing thread's current instruction stream.

#### RAW and WAR hazards

Generic proxy writes data, async proxy immediately reads the same location — the async path can observe stale state. That is a RAW hazard. The async proxy or tensor consumer is still reading a tile and the producer overwrites the same shared-memory region too early — that is a WAR hazard.

- RAW matters when launch is much faster than memory visibility.
- WAR matters when double-buffered shared tiles are reused aggressively.
- Neither problem is solved by "the instruction finished" alone.

### Fences

A fence constrains how memory effects become observable. In Hopper workflows the important distinction is between ordinary ordering and cross-proxy ordering. Per-thread sequencing alone does not make generic and async paths agree on memory state.

#### Release

Producer-side ordering. Earlier writes must become visible before the signal or arrival that follows them.

#### Acquire

Consumer-side ordering. Later reads are not allowed to slip before the synchronization point that tells the consumer the data is ready.

#### Cross-proxy fence

Required when generic and async paths touch the same location. `fence.proxy.async` is the bridge.

```text
// Conceptual producer / consumer ordering
generic proxy writes shared state
cross-proxy or release fence
producer signals barrier
consumer waits with acquire semantics
consumer reads the tile safely
```

> **Important limitation:** a cross-proxy fence is still per-thread ordering. It does not replace the CTA- or cluster-level coordination needed to make sure all writers finished setup before an elected thread launches the async operation.

### The `mbarrier` pipeline

`mbarrier` is a shared-memory hardware synchronization primitive that tracks asynchronous work. A classic barrier asks "have the threads arrived?" An `mbarrier` asks "has the tracked work completed, and is this phase safe to consume?"

1. Initialize the barrier in shared memory with an expected arrival count.
2. Attach expected transaction work for the async operation.
3. Launch the async copy or producer stage.
4. Do independent work while consumers test or wait.
5. Flip phase and reuse the barrier for the next tile.

#### Initialization and reuse

Initialization bugs ruin everything else. The barrier lives in shared memory, the address must be converted correctly, and the object must be 64-bit aligned. Reuse is phase-based — each pipeline round is a separate generation, not one endless counter.

#### Expected transaction count

For async data movement, the barrier tracks both thread arrival and outstanding work (transaction bytes). Completion requires both the arrival count and the transaction count to reach the values the barrier expects.

#### Parity and wait semantics

Parity-based waits check one specific phase. `mbarrier.try_wait.parity` and `mbarrier.test_wait.parity` tell the consumer whether that phase has completed. The `.acquire` form adds the visibility guarantee that makes subsequent reads safe.

```text
// Conceptual consumer pattern
while (!phase_complete) {
  // do unrelated work, test again, or yield
}
// acquire visibility of the produced tile
consume shared-memory data or descriptor-backed work safely
```

> **Practical pattern from the notes:** keep a local phase bit in registers and toggle it at the end of each loop iteration instead of carrying heavier synchronization tokens around.

### Cluster barriers, async groups, and named barriers

Hopper needs multiple synchronization primitives because the ownership question changes depending on scope. Coordinating producer and consumer inside one block is different from ensuring multiple CTAs in a cluster are physically present before remote shared-memory access, which is different from batching several async bulk copies into one committed group.

#### `barrier.cluster.arrive`

Signals cluster arrival without immediately stopping independent work.

#### `barrier.cluster.wait`

Blocks until all participants arrived and cluster-visible writes are safe to consume.

#### `cp.async.bulk.commit_group`

Batches launched bulk async operations into a committed group that later waits can reason about.

#### `cp.async.bulk.wait_group<N>`

Waits until only *N* committed groups remain pending, not until *N* groups have finished.

Named barriers let subsets of warps synchronize without a whole-block rendezvous, which is what makes warp-specialized producer-consumer pipelines possible.

> **Why cluster barriers exist:** standard `__syncthreads()` cannot coordinate block-to-block handoff inside a cluster, which matters for DSMEM access and TMA multicast patterns.

### Design rules

1. **Separate progress from visibility.** "Done" and "safe to read" are not interchangeable.
2. **Use the right barrier for the ownership question.** Thread rendezvous, async work tracking, and cluster presence are different contracts.
3. **Keep scopes honest.** CTA scope, cluster scope, GPU scope, and system scope are not cosmetic suffixes.
4. **Make shared-memory reuse explicit.** If ownership is changing, encode it with arrival, wait, and phase management.
5. **Use asynchronicity as a scheduling tool.** The payoff is clean overlap that keeps the machine productive.

#### Glossary

| Term | Definition |
| --- | --- |
| Proxy | A memory-operation path or agent with its own visibility and ordering rules. |
| RAW hazard | A consumer reads before the producer's newer write is visible on the path it uses. |
| WAR hazard | A producer overwrites data before the consumer has finished reading the previous tile. |
| `mbarrier` | A shared-memory hardware barrier that tracks asynchronous work across reusable phases. |
| Phase / parity | The generation state of a reusable barrier, often tracked with a single bit. |
| `wait_group` | A bulk async group wait based on how many committed groups are still pending. |

### Continue the course

Lesson 3 establishes the correctness layer of the asynchronous machine. Lesson 4 moves back into the movement path itself: how Hopper TMA uses a host-encoded descriptor to describe tiles, strides, swizzles, interleaving, and fetch policy before the transfer even begins.

---

## PTX ISA Deep Dive: `mbarrier` Internals

*Reference: PTX ISA 9.2, `mbarrier` chapter and the async-copy / async-store / async-reduce sections that hook into `mbarrier::complete_tx::bytes`.*

### What an `mbarrier` actually is

In PTX, an `mbarrier` is an opaque `.b64`, 8-byte-aligned object in shared memory. It tracks four things:

1. The **current phase**.
2. The current phase's **pending-arrival count**.
3. The next phase's **expected-arrival count**.
4. The current phase's **`tx-count`** for outstanding asynchronous transactions.

A phase completes only when **both** pending arrivals and `tx-count` reach zero. Completion then atomically advances the phase and reloads pending arrivals from expected arrivals.

### The two-debt mental model

An `mbarrier` phase has **two debts** that must be paid before waiters can observe completion:

| Debt | Meaning | Increased by | Decreased by |
|------|---------|--------------|--------------|
| Arrival debt | How many arrivals are still missing? | `mbarrier.init` (sets initial count) | `mbarrier.arrive*` |
| Transaction debt | How many async transactions are still outstanding? | `mbarrier.expect_tx` / `mbarrier.arrive.expect_tx` | `mbarrier.complete_tx` (explicit or implicit) |

`mbarrier.arrive*` pays down the first debt. `mbarrier.expect_tx` adds to the second debt. `mbarrier.complete_tx` pays down the second debt. `mbarrier.arrive.expect_tx` does both in one shot.

### `mbarrier.init`

```
mbarrier.init{.shared{::cta}}.b64 [addr], count;
```

Creates a valid barrier object at `addr`. Sets:
- phase = 0
- expected-arrival count = `count`
- pending-arrival count = `count`
- `tx-count` = 0

`count` must be in `[1, 2^20 - 1]`. If you omit the state space, generic addressing is used, but the address must still resolve to `.shared::cta` or behavior is undefined. Calling `mbarrier.init` on memory that already holds a valid `mbarrier` is also undefined — call `mbarrier.inval` first. Requires `sm_80+`.

`init` means: "for this phase, I expect `count` arrivals, and I am not yet tracking any async transactions." Without any `expect_tx`, the phase completes when `N` arrive-on events have happened.

### `mbarrier.arrive`

```
mbarrier.arrive{.sem.scope}{.shared{::cta}}.b64 state, [addr]{, count};
```

Performs an **arrive-on** operation: decrements the barrier's pending-arrival count by `count` (default 1), then checks whether the phase can now complete. The phase completes only if pending arrivals are zero **and** `tx-count` is zero.

Details:
- Default `.release` semantics; `.relaxed` optional on newer targets.
- On `.shared::cta`, returns an opaque 64-bit `state` capturing the phase **before** the arrive-on. Feed this to `mbarrier.test_wait` or `mbarrier.try_wait` later.
- On `.shared::cluster` (not local CTA shared memory), cannot return a value — destination must be `_`.
- Basic `arrive`: PTX 7.0 / `sm_80+`. `.expect_tx`, `.cluster`, `.scope`, `.relaxed`: `sm_90+`.

`arrive` says "my participation in this phase is done" and decreases arrival debt. It does **not** touch async work. If `tx-count` is nonzero, `arrive` alone will not complete the phase — you also need `complete_tx`.

### `mbarrier.arrive.expect_tx`

```
mbarrier.arrive.expect_tx{.sem.scope}{...}.b64 state, [addr], txCount;
```

Fused form: first does `expect-tx` with `expectCount = txCount`, then does the normal arrive-on with arrive count fixed at 1. Adds `txCount` to the phase's transaction debt and contributes one arrival toward the arrival debt.

Use when a thread's arrival also means "this phase must not complete until `txCount` async transaction units are retired." Inherits normal `arrive` behavior for return state and memory semantics.

**Watch out:** this can make a phase *look* arrival-complete while still incomplete — it may have driven pending arrivals to zero but also increased `tx-count`. Waiters cannot observe completion until `complete_tx` drives `tx-count` back to zero.

### `mbarrier.expect_tx`

```
mbarrier.expect_tx{.sem.scope}{.space}.b64 [addr], txCount;
```

Performs **only** the expect-tx part: increases the current phase's `tx-count` by `txCount`. Does **not** touch pending arrivals and does not itself wait. Supports `.shared::cta` and `.shared::cluster`, uses generic addressing if no state space is given, and is available only with `.relaxed` semantics. Introduced in PTX 8.0, requires `sm_90+`.

`expect_tx` is purely "increase async debt." It is useful when the async work and the thread's arrival are logically separate, or when some other instruction will later retire the debt by generating `complete_tx`.

### `mbarrier.complete_tx`

```
mbarrier.complete_tx{.sem.scope}{.space}.b64 [addr], txCount;
```

The opposite accounting step: decrements `tx-count` by `txCount`, then checks whether the phase can complete. PTX is very explicit that the instruction itself does **not** perform an asynchronous memory operation; it merely simulates the completion of one and applies the corresponding barrier side effect. Like `expect_tx`, it supports `.shared::cta` and `.shared::cluster`, only exposes `.relaxed`, and requires `sm_90+`.

If pending arrivals were already zero, this may be the instruction that flips the barrier into the next phase. If arrivals are still pending, it just reduces the outstanding transaction debt and the barrier remains incomplete.

**Subtle but important:** the explicit `mbarrier.complete_tx` instruction is only `.relaxed`. But several async PTX instructions generate an **implicit** `complete-tx` when the async operation finishes. For example, `cp.async.bulk...mbarrier::complete_tx::bytes` performs a `complete-tx` whose `completeCount` equals the number of bytes copied, and PTX says that implicit `complete-tx` has `.release` semantics at `.cluster` scope. The same pattern appears for `cp.async.bulk.tensor`, `st.async.shared::cluster.mbarrier::complete_tx::bytes`, and `cp.reduce.async.bulk`.

### How they fit together in one phase

A typical phase using these instructions works like this:

1. **`mbarrier.init [bar], N`** — sets the barrier to expect `N` arrivals and zero async debt.
2. **Threads arrive** — either via `mbarrier.arrive` or `mbarrier.arrive.expect_tx`, depending on whether they also need the phase to track async work. `expect_tx` may also be done separately.
3. **Async operations retire debt** — either explicitly via `mbarrier.complete_tx` or implicitly through instructions like `cp.async.bulk...mbarrier::complete_tx::bytes`.
4. **Phase completes** — only when arrival debt and transaction debt are both zero. At that moment the phase advances atomically. Waiters observe this with `mbarrier.test_wait` / `mbarrier.try_wait`; with `.acquire`, those waits form the acquire side of synchronization. PTX also requires that for each phase, at least one wait must successfully observe completion before any arrival in the subsequent phase.

### The shortest intuition for each opcode

| Instruction | One-liner |
|-------------|-----------|
| `mbarrier.init` | Start a fresh reusable barrier phase machine with `count` expected arrivals. |
| `mbarrier.arrive` | I am done with this phase; subtract from pending arrivals. |
| `mbarrier.arrive.expect_tx` | I am done with this phase, and this phase must also wait for `txCount` async transaction units. |
| `mbarrier.expect_tx` | Add `txCount` async debt to this phase. |
| `mbarrier.complete_tx` | Retire `txCount` of that async debt. |

**Biggest practical pitfall:** `arrive` does not mean completion once `tx-count` is involved. If you used `expect_tx` anywhere in the phase, then the barrier will not complete until matching `complete_tx` activity has driven `tx-count` back to zero.

---

## PTX ISA Deep Dive: Fences and Proxy Fences

*Reference: PTX ISA 9.2, fence / membar sections and the CUDA Programming Guide proxy model.*

### The mental model

Ordinary loads/stores/atomics/reductions use the **generic proxy**. Some mechanisms use a different proxy, especially the **async proxy**. A proxy is an abstract label on a method of memory access — if two accesses use different proxies, a **proxy fence is required** to synchronize them.

The memory model requires that two operations use the **same proxy** to be "morally strong" with respect to each other. Generic on one side and async on the other means ordinary memory-order reasoning is insufficient; you need a cross-proxy fence.

`cp.async`, `cp.async.bulk`, `cp.reduce.async.bulk`, and `wgmma.mma_async` are not normal in-thread operations in plain program order. They have weaker ordering guarantees, and you must rely on each instruction family's documented completion and synchronization rules.

### PTX fence syntax reference

```
// Thread fence:
fence{.sem}.scope;

// Thread fence with sync restriction:
fence.acquire.sync_restrict::shared::cluster.cluster;
fence.release.sync_restrict::shared::cta.cluster;

// Operation fence:
fence.op_restrict.release.cluster;

// Proxy fence (bi-directional):
fence.proxy.proxykind;

// Proxy fence (uni-directional):
fence.proxy.to_proxykind::from_proxykind.release.scope;
fence.proxy.to_proxykind::from_proxykind.acquire.scope [addr], size;

// Async/generic restricted proxy fence forms:
fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;
fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;

// Old-style membar:
membar.level;
membar.proxy.proxykind;
```

### 1. Ordinary fences: `fence` and old-style `membar`

The main syntax is `fence{.sem}.scope`. Semantic qualifiers: `.sc`, `.acq_rel`, `.acquire`, `.release`. Scopes: `.cta`, `.cluster`, `.gpu`, `.sys`. The old `membar.level` form still exists with levels `.cta`, `.gl`, `.sys`. On `sm_70+`, `membar` is effectively `fence.sc`, and `membar.gl` maps to `fence.*.gpu`.

For scope, PTX defines:
- `.cta` — current thread block
- `.cluster` — current cluster
- `.gpu` — current device
- `.sys` — whole program including host threads and all devices

#### `fence.acq_rel` (the default)

Default if you omit the semantic qualifier. Sufficient for most synchronization patterns, but only synchronizes when combined with the right surrounding memory operations. `fence.acq_rel` by itself is not magic global synchronization — it participates in a pattern: "release fence; strong write" on one side, "strong read; acquire fence" on the other.

A release pattern is `fence.release` followed by a strong write. An acquire pattern is a strong read followed by `fence.acquire`. If the write from the release side is what the acquire side reads, the two synchronize and establish causality. The classic message-passing litmus test: producer writes data, fences, writes flag; consumer reads flag, fences, reads data.

#### `fence.sc` (the strong fence)

All morally strong `fence.sc` operations participate in a runtime partial order (Fence-SC order). If two `fence.sc` operations are ordered there, they synchronize. In the store-buffering litmus test, `fence.sc.sys` prevents the "both reads see 0" result; `fence.acq_rel` does **not**.

#### Fine-grained ordinary fence variants

`fence.acquire` and `fence.release` are newer qualifiers (PTX 8.6, `sm_90+`). Also:
- `fence.op_restrict.release.cluster`, where `.op_restrict = .mbarrier_init`, meaning the fence only orders prior `mbarrier.init` operations by the same thread.
- `.sync_restrict` forms for `.shared::cta` and `.shared::cluster`, which narrow the fence's effect to those state spaces and require `.cluster` scope.

### 2. Proxy fences in general: `fence.proxy.*`

Proxy fences bridge two sides of a communication that use different proxies. There are two flavors: **bi-directional** (orders both directions between generic and the named proxy) and **uni-directional** (orders "from proxy X" before "to proxy Y").

Bi-directional proxy kinds: `.alias`, `.async`, `.async.global`, `.async.shared::cta`, `.async.shared::cluster`. `fence.proxy.alias` handles virtual aliases of the same physical location. `fence.proxy.async.shared::cta` narrows the cross-proxy ordering to async-vs-generic on CTA shared memory only.

#### The alias case

The PTX litmus example shows a write through one virtual alias, then `fence.proxy.alias`, then a read through another alias. Virtual aliases of the same physical location require an alias proxy fence along the synchronization path.

#### Uni-directional proxy fences

Uni-directional proxy fences have `.release` / `.acquire` semantics. The **direction** comes from `to_proxykind::from_proxykind`; `.release` and `.acquire` control whether the proxy fence participates in a release or acquire synchronization pattern. A `.release` proxy fence forms a release sequence that synchronizes with an acquire sequence containing a `.acquire` proxy fence.

The address-range form of uni-directional proxy acquire: `addr` and `size` define the range over which cross-proxy ordering applies. Only supported `size` is `128`, and `addr` must be a generic address landing in `.global`.

### 3. `fence.proxy.async` specifically

Use when one side is in the **generic proxy** and the other is in the **async proxy**.

**Not every async-looking instruction uses the async proxy:**
- Plain `cp.async` is a weak memory operation performed in the **generic proxy**.
- `cp.async.bulk` and `cp.reduce.async.bulk` are performed in the **async proxy**.
- `wgmma.mma_async` is also performed in the **async proxy**.

Easy place to get tripped up. The rule:

| Traffic pattern | Fence needed |
|----------------|--------------|
| Ordinary generic-proxy traffic | Ordinary `fence` |
| Generic-vs-async traffic | `fence.proxy.async` |
| Aliasing through different virtual addresses | `fence.proxy.alias` |

#### What plain `fence.proxy.async` means

`fence.proxy.async` is the **bi-directional** form: generic-proxy and async-proxy accesses to the same location must respect each other's ordering. Narrower forms exist for when you only care about one state space: `fence.proxy.async.shared::cta`, `fence.proxy.async.shared::cluster`, `fence.proxy.async.global`.

#### What it is for

Data produced by normal stores and consumed by an async-proxy operation (or vice versa). If the same location is touched across generic and async proxies, you need this fence.

`wgmma` trap: **`wgmma.fence` is not a substitute for `fence.proxy.async`.** `wgmma.fence` orders **register** accesses relative to `wgmma.mma_async`. If you wrote matrices to **shared memory** generically and `wgmma.mma_async` reads them through the async proxy, you still need an **async proxy fence** for the shared-memory handoff.

#### What it does *not* do

`fence.proxy.async` does **not** mean "wait until the async operation is finished." It only provides cross-proxy ordering. Completion is separate:
- `cp.async.bulk` / `cp.reduce.async.bulk` completion must be observed using an async-group or `mbarrier`.
- `wgmma.mma_async` requires `wgmma.commit_group` / `wgmma.wait_group`.
- `clusterlaunchcontrol.try_cancel` can only use the documented `mbarrier` mechanism.

Important convenience: **completion** of `cp{.reduce}.async.bulk` and `wgmma.mma_async` is followed by an **implicit generic-async proxy fence**, so once completion is observed, the result is visible to the generic proxy. You still need both pieces: a completion wait to know the async op is done, and proxy-fence semantics so generic code can safely read the results.

### 4. The acquire/release flavored `fence.proxy.async::generic.*` forms

These show up in cluster and advanced async examples:

```
fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;
fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;
```

Not "just stronger versions" of plain `fence.proxy.async`. They are **directional** and participate in **release/acquire synchronization patterns** across threads while also establishing cross-proxy ordering. The `clusterlaunchcontrol.try_cancel` example uses them to order a generic-proxy weak read of a shared-memory handle against later async-proxy writes across cluster iterations.

`.sync_restrict` qualifiers:
- `.sync_restrict::shared::cta` may only be used with `.release`
- `.sync_restrict::shared::cluster` may only be used with `.acquire`
- In either case the scope must be `.cluster`

This is a very narrow acquire/release bridge for specific shared-memory communication patterns across CTAs in a cluster.

### 5. Version and architecture support

| Feature | Minimum PTX ISA | Minimum SM |
|---------|----------------|------------|
| `fence` | — | `sm_70` |
| `membar.proxy` | — | `sm_60` |
| `fence.proxy` | — | `sm_70` |
| `fence.proxy.async` | PTX 8.0 | `sm_90` |
| `.cluster` scope | — | `sm_90` |
| `.acquire` / `.release` qualifiers for `fence` | — | `sm_90` |

### Practical "when to use which fence" cheat sheet

| Situation | Fence to use |
|-----------|-------------|
| Normal message-passing on the generic proxy | `fence.acq_rel.scope` |
| Need SC-style behavior that acquire/release patterns do not guarantee | `fence.sc.scope` |
| Same physical bytes touched through different virtual aliases | `fence.proxy.alias` |
| One side is generic, other side is async proxy (`cp.async.bulk`, `cp.reduce.async.bulk`, `wgmma.mma_async`) | `fence.proxy.async[...]` |
| Need cross-proxy ordering AND release/acquire across CTAs in a cluster | `fence.proxy.async::generic.{release,acquire}.sync_restrict::shared::...` |

**Biggest gotcha:** `cp.async` is generic-proxy, but `cp.async.bulk` and `wgmma.mma_async` are async-proxy. That one distinction often determines whether you need an ordinary fence, a proxy fence, or both.

> **Do not confuse ordering with completion.** Proxy fences order visibility across proxies. `wait_group`, `commit_group`, `mbarrier`, or related mechanisms tell you the async work is actually done.

---

## PTX ISA Deep Dive: CTA Named Barriers (`bar` / `barrier`)

*Reference: PTX ISA 9.2, `bar` / `barrier` instruction pages.*

In PTX, **"named barriers"** refers to the CTA barrier resources used by `bar` / `barrier` instructions, where each CTA has **16 logical barriers numbered `0..15`**. PTX says each CTA instance has sixteen barriers, and operand `a` selects one of them.

### Full instruction forms

```ptx
// Preferred spellings
barrier{.cta}.sync{.aligned}          a{, b};
barrier{.cta}.arrive{.aligned}        a, b;
barrier{.cta}.red.popc{.aligned}.u32  d, a{, b}, {!}c;
barrier{.cta}.red.op{.aligned}.pred   p, a{, b}, {!}c;

// Older short spellings
bar{.cta}.sync                        a{, b};
bar{.cta}.arrive                      a, b;
bar{.cta}.red.popc.u32                d, a{, b}, {!}c;
bar{.cta}.red.op.pred                 p, a{, b}, {!}c;

.op = { .and, .or };
```

### What the operands mean

`a`, `b`, and `d` are `.u32`; `p` and `c` are predicates.

| Operand | Meaning |
|---------|---------|
| `a` | Barrier number, `0..15` (immediate or register) |
| `b` | Participating thread count, must be a **multiple of warp size**. If omitted, all threads in the CTA participate. |
| `d` | Reduction destination register for `popc` |
| `p` | Predicate destination register for `.and` / `.or` |
| `c` | Per-thread predicate input, optionally negated as `{!}c` |

### What each instruction does

#### `barrier.cta.sync a{, b}` / `bar.sync a{, b}`

The standard named barrier wait. The thread signals arrival at barrier `a` and waits until all participating warps have arrived. PTX says `barrier{.cta}.sync` waits for all non-exited threads from the executing thread's warp and, after that, for all other participating warps too.

```ptx
bar.sync 0;                  // all threads in CTA
bar.cta.sync 1, 64;          // only 64 threads
barrier.cta.sync 2;
barrier.cta.sync 3, 128;
```

#### `barrier.cta.arrive a, b` / `bar.arrive a, b`

Signals arrival at named barrier `a` but does **not** wait for other participating warps. PTX explicitly says `barrier{.cta}.arrive` does not cause the executing thread to wait for threads of other participating warps. This is what enables producer/consumer patterns.

```ptx
bar.arrive 0, 64;
barrier.cta.arrive 1, 128;
```

#### `barrier.cta.red.popc.u32 d, a{, b}, {!}c`

A named barrier plus a population-count reduction over the predicate contribution from participating threads. Counts how many threads satisfy a condition.

```ptx
bar.red.popc.u32 r3, 1, p;
barrier.cta.red.popc.u32 r4, 2, !p;
```

#### `barrier.cta.red.{and,or}.pred p, a{, b}, {!}c`

Named barrier reductions producing a predicate result using logical AND or OR over the participating threads' predicate inputs.

```ptx
bar.red.and.pred p0, 1, p1;
bar.red.or.pred  p2, 2, !p3;

barrier.cta.red.and.pred p0, 1, p1;
barrier.cta.red.or.pred  p2, 2, !p3;
```

### Memory-ordering semantics

PTX gives strong guarantees for CTA named barriers:

- `barrier{.cta}.sync`, `barrier{.cta}.red`, and `barrier{.cta}.arrive` guarantee that when the barrier completes, prior memory accesses requested by that thread are **performed** relative to all threads participating in the barrier.
- `barrier{.cta}.sync` and `barrier{.cta}.red` additionally guarantee that the thread does not issue new memory accesses before the barrier completes.

PTX defines "performed" precisely:
- A **read** is performed when the read value has been transmitted from memory and cannot be modified by another participating thread.
- A **write** is performed when the written value is visible to participating threads and the old value can no longer be read.

That is why named barriers are safe for shared-memory handoff patterns inside a CTA.

### Reuse and completion rules

When a CTA named barrier completes, the waiting threads restart without delay and the barrier is automatically reinitialized so it can be reused immediately.

Because barrier participation is tracked at warp granularity, the instruction first waits for all non-exited threads in the executing warp before the warp is considered arrived.

### Important restrictions

- For `barrier{.cta}` without a thread count, the instruction is equivalent to the `.aligned` variant and has the same restrictions.
- All threads in a warp, except exited threads, must execute `barrier{.cta}` in convergence.
- Thread count `b`, when present, must be a multiple of warp size.
- `barrier{.cta}.arrive` requires non-zero `b`.

The practical consequence: think in **whole-warps arriving at a barrier**, even when `b` is smaller than the whole CTA.

### The `.aligned` modifier

The full preferred syntax includes `{.aligned}` on the `barrier` spellings. PTX notes that `barrier{.cta}` without `.aligned` is equivalent to the `.aligned` variant and has the same restrictions. So for CTA named barriers, omitting `.aligned` does not relax the convergence rule.

### Short examples

#### Full-CTA synchronization

```ptx
st.shared.u32 [r0], r1;     // write to shared
bar.sync 0;                  // synchronize
ld.shared.u32 r2, [r3];     // read peers' shared data
```

The classic "write to shared, synchronize, then read peers' shared data" pattern.

#### Partial-CTA synchronization

```ptx
bar.cta.sync 1, 64;         // only 64 threads participate (must be multiple of warp size)
```

#### Producer/consumer with two named barriers

PTX gives a producer/consumer example using two named barriers:

```ptx
// Producer
st.shared   [r0], r1;       // produce data
bar.arrive  0, 64;          // signal "data ready"
ld.global   r1, [r2];       // do other work
bar.sync    1, 64;          // wait for "data consumed"

// Consumer
bar.sync    0, 64;          // wait for "data ready"
ld.shared   r1, [r0];       // consume data
bar.arrive  1, 64;          // signal "data consumed"
```

One barrier signals "data ready," the other signals "data consumed."

#### Reduction barrier

```ptx
setp.eq.u32 p, r1, r2;
bar.cta.red.popc.u32 r3, 1, p;    // count threads where p is true
bar.cta.red.and.pred p0, 1, p;     // true only if all participating threads have p=true
```

### Quick reference table

| Form | Blocks? | Reduction result | Participants | Memory guarantee |
|------|---------|-----------------|-------------|------------------|
| `bar.sync a` | Yes | — | All CTA threads | Full performed guarantee + no new accesses before completion |
| `bar.sync a, b` | Yes | — | `b` threads (warp-multiple) | Full performed guarantee + no new accesses before completion |
| `bar.arrive a, b` | **No** | — | `b` threads (warp-multiple, non-zero) | Performed guarantee on prior accesses |
| `bar.red.popc.u32 d, a, c` | Yes | `d` = popcount of `c` across participants | All CTA threads | Full performed guarantee + no new accesses before completion |
| `bar.red.and.pred p, a, c` | Yes | `p` = AND of `c` across participants | All CTA threads | Full performed guarantee + no new accesses before completion |
| `bar.red.or.pred p, a, c` | Yes | `p` = OR of `c` across participants | All CTA threads | Full performed guarantee + no new accesses before completion |

### PTX version and target support

- `bar.sync` without thread count: PTX ISA 1.0
- Register operands, thread count, and `bar.{arrive,red}`: PTX ISA 2.0
- `barrier` instruction spelling: PTX ISA 6.0

---

## Full Slide Deck Text

Extracted from `H100-Course/slides/3. Asynchronicity and barriers.pdf` with `pdftotext -layout`. Total slides: 50.

### Slide 1: Asynchronicity and barriers

Prateek Shukla

### Slide 2: The systems in synchronous world

In any system there are two major operations

Doing: processing information, solving an equation, cooking food

Fetching: retrieving the data, reading, comprehending the question, bringing the
ingredients

In a synchronous world, these happen sequentially. You are "blocked" until the
current task finishes.

The synchronous way of cooking food is where while you are cooking food you are
not multitasking on any other work. Once the food is cooked then you go to the next
task. You are "blocked" from doing anything else

### Slide 3: Asynchronicity decouples the Request from the Result.

While cooking food, you go out to work on something else and after some time at
some kind of signal you come back and use the ready food.

You accomplished two tasks in the time it took to do one. You have achieved
Latency Hiding.

If the cost of fetching a resource is higher than the time it takes to process it, you
must overlap the fetching of the next item with the processing of the current item.

The essence of latency hiding is not being "blocked" by an instruction and the
ability to put the current job in background and work on something else.

### Slide 4: Blocking and non blocking operation

A blocking or synchronous operation pauses the execution of the launching thread
(e.g., the CPU host thread) until that operation has fully completed. Control does
not return to the thread, so it cannot proceed to the next line of code. This creates
an implicit synchronization point, guaranteeing the task is finished before the host
continues

A non-blocking or asynchronous operation returns control immediately to the
launching thread, before the operation has actually finished.

### Slide 5: Asynchrony in GPUs

In modern computing, "Doing" (Math/Compute) is incredibly fast. "Fetching"
(Memory Access) is agonizingly slow in comparison.

The time it takes to fetch the data is much higher than the time it takes to compute
on that data. Thus the goal of high performance architecture is not just to make
the math faster; it is to ensure the math never stops.

The Nvidia H100 is often praised for its raw speed (FLOPS), but its true genius
lies in its architecture of asynchronicity. It is designed to ensure that its massive
Tensor Cores never have to wait for the data. This is powered by using non
blocking instructions for tensor cores and TMA

### Slide 6: Async operations follow this pattern

Stage 1: Initialization - A thread fires an asynchronous operation and immediately
goes off to the next instruction

Stage 2: Tracking and parallel execution - some part of system tracks the
operation (mbarrier for TMA, internal hardware scorecard for wgmma). While
these operations are working, other units work in parallel

Stage 3: Synchronization - Once the async operation is done operating in the
background

### Slide 7: The synchronization step in H100

Warp launches instructions really fast in the order of a few nanoseconds

Once a async instruction reaches the execution unit, it attempts to read the
memory immediately

The bottleneck is the memory bandwidth. Moving data from HBM to shared
memory/registers is a high-latency operation which take a few hundred ns
compared to the logic core.

### Slide 8: The two problems

Because the Issuer (~ns) is faster than the Memory Mover (~ hundreds of ns) tensor
cores would have to wait for a few hundred ns for the data to arrive if the instructions
for copy and compute are issued at the same time. Therefore there is a scope for
latency hiding

Also in a big kernel, where instruction queue is full of commands for data that hasn't
arrived yet, without some form of synchronization tensor cores might perform
operations before the data have even arrived.

We need a way to make sure that the right operations are done on right data and we
need to have a system where we can apply latency hiding

### Slide 9: What exactly is mbarrier/semaphore

A hardware-accelerated synchronization primitive resident in Shared Memory
designed to track the completion of asynchronous memory transactions.

Unlike traditional barriers that block execution until threads arrive, an mbarrier blocks
execution until data arrives. It decouples the "Producer" (who issues the copy) from
the "Consumer" (who waits for the data), enabling split-phase, fire-and-forget memory
pipelines.

Producer sets the expected data transfer size in the barrier and then the hardware
does its job in the background and once the transaction is completed the barrier is
opened

### Slide 10: Why is this a great solution

.The barrier forces the "fast" instruction launcher to respect the "slow" physical
hardware by

1. Making sure that the slower operations are launched independently from the
faster operations so that by the time the fast unit is launched we have the data

2. Have a way to prevent the fast execution unit from operating on the wrong data

And the way we do that is by using mbarrier.

### Slide 11: The Big Picture

### Slide 12: Proxies in CUDA

In the context of the NVIDIA H100 and the CUDA PTX memory model, Proxy is a
term used to distinguish who is performing memory operations.

If two memory operations happen in the same proxy (e.g., the Generic Proxy), the
hardware guarantees they are safe and ordered (mostly).

If operation A is in Proxy 1 and operation B is in Proxy 2, the hardware stops
checking. It assumes they are completely unrelated. It lets them run wild, out of
order, and in parallel

### Slide 13: Generic proxy vs Async proxy

When you write a CUDA kernel, your thread executes instructions sequentially.
When the thread reads or writes memory, it is acting as the Generic Proxy. The
hardware guarantees that these operations happen in the order you wrote them
(mostly) within that specific thread.

The Async Proxy refers to the hardware mechanism performing bulk async copies
or tensor core math. When you issue a command like cp.async.bulk or wgmma,
the Generic Proxy (thread) just kicks off the command and immediately moves to
the next line of code.

The Async Proxy runs completely independently of the Generic Proxy. It does not
know what the thread is currently doing, and the thread does not automatically
know when the proxy is finished.

### Slide 14: Read After Write

The Generic Proxy and Async Proxy have distinct paths to memory. The Generic
Proxy operates through the SM's L1 cache, while the Async Proxy bypasses the
L1 and interacts directly with the L2 cache or HBM.

Stores issued by the Generic Proxy are initially held in local store buffers or the L1
cache. They are not immediately visible to the rest of the system (including TMA).

If the Generic Proxy writes to a memory address and immediately triggers the
Async Proxy to read that same address, the Async Proxy will likely read stale data
from L2/DRAM because the new data is still stuck in the Generic Proxy's L1

This is called a read after write hazard

### Slide 15: Write after read

A thread (or CTA) does a Generic Proxy read of address X, pulling X into L1 (or
otherwise "anchoring" an old version locally).

Later, an Async Proxy write updates X via L2/HBM, without updating/invalidating
the Generic Proxy's L1 view. The Generic Proxy can keep operating on the stale
L1 line. Two common failure modes:

A later Generic Proxy store / writeback / eviction can clobber the newer value
written by the Async Proxy (stale line wins) or later Generic Proxy loads keep
returning the old value even though X was updated via Async Proxy.

This is called a write after read hazard or WAR

### Slide 16: fences

A fence is an explicit ordering/visibility point that constrains how memory effects
become observable. You use it when the hardware might otherwise let things
"pass" each other especially with asynchronous operations that are not
automatically ordered with normal loads/stores the way you might assume

NVIDIA explicitly calls out that a proxy fence is required to synchronize across
proxies for proper ordering

Also these fences are scoped (e.g., .cta, .cluster, .gpu, .sys) and the scope
determines who must see the ordering, tied to a point of coherency in the
hierarchy (e.g., L1 vs L2).

There are two types of fences Ordinary fence and cross proxy fence

### Slide 17: Release fence (producer-side)

A release fence enforces this one-way rule:

All memory operations (especially writes) that appear before the release in
program order become visible-before anything that appears after the release to
other threads that synchronize with it (at the chosen scope).

It prevents earlier writes from being delayed/reordered past the release point.

### Slide 18: Acquire fence (consumer-side)

An acquire fence enforces the opposite one-way rule:

No memory operation that appears after the acquire in program order (especially
reads) is allowed to be observed as happening before it.

After the acquire, the thread is guaranteed to observe the writes that were made
visible by a matching release (again, within the chosen scope/proxy).

### Slide 19: Cross proxy fence

If you access the same location across multiple proxies, you need a cross-proxy
fence. For the async proxy, use fence.proxy.async to synchronize memory
between generic and async proxy.

It drains/orders the generic-proxy shared-memory write visibility so the async
proxy doesn't read an older view.

it's not a "collective flush." It's per-thread ordering, so you still need a block sync
so all writers have done it before the elected thread launches TMA (the CUDA
example uses __syncthreads() around the setup/coordination).

### Slide 20: The whole mbarrier pipeline

Step 1: initialize the mbarrier in shared memory and use mbarrier.init to create a
barrier with expected thread arrival count. Thread arrival count make sure that a
specific number of threads have issued the copy instructions.
Step 2: Set the expected transaction count for the async op you are gonna launch,
using mbarrier.arrive. Everytime you do this you are logging an instruction thus
reducing the arrival count.
Step 3: launch the async operation
Step 4: launch other operations and wait (threads sleep till thread arrival and
expected_tx = 0)
Step 5: flip the phase, reset the arrival count, do the next operation

### Slide 21: Phase and expected transaction count

A barrier's phase refers to its current reusable state or cycle. It is a single bit which
flips whenever the barrier completes a cycle.
Transaction count represent the size of the asynchronous operations which you
are performing. Expected transaction count is the amount of 'work' which is yet to
be done with the async operation.
Because these are async ops, the hardware automatically decrements the
transaction count as the async operation progresses. We attach the barrier to the
async operation to make sure of this behavior
You can 'reuse' a barrier by flipping its phase at the end of operation and
increasing the transaction count. The thread arrival count is automatically resetted
as per the value set during initialization.

### Slide 22: Initializing the mbarrier

This is the only instruction where most bugs originate. Get this wrong, and nothing
else matters

addr is just the memory address of the mbarrier in the state space

Count is the expected thread arrival count in the barrier, this is the value to which
the barrier will reset when you reuse the barrier by flipping the phase.

### Slide 23: Some things to keep in mind

Generally the scope is shared::cta so that only the threads in the same thread
block can "see" the barrier

The address is shared memory pointer which is created using __cvta

You can use shared::cluster to make the barrier visible to all the threads in a
cluster

You must use the mapa ptx instruction to get the address in the cluster

Mbarrier objects must be 64 bit aligned, unaligned access may cause silent
corruption

### Slide 24: Setting expected transaction count

Decrementing the arrival count by 1

adding the expected transaction bytes with tx_count

The tx_count is added to the barrier's pending tx-count. If you call
arrive.expect_tx twice with 4096, the barrier expects 8192 bytes total

### Slide 25: The placeholder _

The _ in the instruction is a placeholder for the phase_out which is the way to get a
64 bit encoded token which contains information about the current phase and
transaction count of the barrier

It was there for some complicated synchronization issues

We discard phase_out because we are actually writing this information into the
producer threads. Register pressure is expensive. it is cheaper to toggle a 1-bit
integer than writing a 64 bit token in registers.

### Slide 26: Synchronization

So till now we have launched the async copy instruction but now how do we know
that the operation is completed.

The old way was barrier.sync which would block the thread until the operation
completes but that destroys asynchronicity

The way we do it in hopper is by having an instruction which can provide true or
false if the operation is completed. This allows Latency Hiding. A thread can check
the barrier, see it's not ready repeat until the operation ends and the phase flips.
This is precisely the job of mbarrier.try_wait.parity

### Slide 27: mbarrier.try_wait.parity

You pass a phaseParity bit (0 or 1) into the instruction representing the specific
execution phase you must verify is fully completed before proceeding

The hardware compares your input parity bit against the mbarrier object's current
internal parity state to determine if that specific phase is still active

If your input parity equals the barrier's current parity, the barrier is still processing
that phase, so the instruction returns false (keep waiting). If your input parity differs
from the barrier's current parity, the barrier has advanced to the next phase, so the
instruction returns true (proceed)

A successful return implies the barrier parity has "flipped," meaning your requested
parity now refers to the immediately preceding (and therefore completed) phase

### Slide 28: The instruction

try_wait: The operation name. It implies an attempt that might involve a temporary
suspension of the thread.

waitComplete: (Destination) The boolean result.

1 (True): The barrier has reached the target phase (work is done).

0 (False): The barrier is not done, and the thread has just woken up

suspendTimeHint: Optional immediate value (constant) that tells the GPU scheduler
how long to yield the thread if the condition isn't met.

phaseParity: An integer (0 or 1) representing the phase generation you are waiting for.

### Slide 29: mbarrier.test_wait.parity

test_wait.parity : The variant of the instruction. It tracks the barrier's progress using
a "phase parity" bit (0 or 1) rather than a raw integer count.

waitComplete: (Destination) A 1-bit predicate register (boolean). It receives 1 if the
barrier is done, 0 if it's still busy.

[addr]: (Source) The pointer/address to the mbarrier object in shared memory.

phaseParity: (Source) An integer (0 or 1) representing the phase generation you
are waiting for.

### Slide 30: mbarrier wait with .acquire

It does everything which is there in mbarrier.try_wait with a critical distinction.

If the current synchronization round is fully finished, the .acquire semantic creates a
strict memory fence, ensuring that all data writes from the async proxy or other
threads are guaranteed visible before proceeding.

If you are loading data from Shared Memory into Registers (to feed a Tensor Core
manually), the .acquire ensures those LD instructions don't fire until the data is valid.
If they fire early, your registers get garbage.

Even though wgmma reads directly from Shared Memory, the instruction itself
requires descriptors and memory state to be consistent. The .acquire ensures the
dependency chain is respected

### Slide 31: The complicated ways to implement it

### Slide 32: An important point

In the producer warp which is launching the copies only a single thread is running
all the instructions like mbarrier.init, mbarrier.arrive.expect_tx, cp.async.bulk and
all that. Other threads in the warp are just lying around

The consumer warps are the ones which launch the wait operation as they are
waiting for the tma copy to complete so that they can work on the data

### Slide 33: mbarrier reuse pattern

The phase is declared in registers
and not shared memory!
Each thread keeps its own int phase
phase ^= 1 as the last operation in loop

### Slide 34: Is mbarrier only for async bulk copies??

Till now we discussed about mbarriers and cp.async.bulk and it really feels that
mbarriers are built for async bulk copies.
Consider this:
Producer writes to Shared Memory
Consumer reads from Shared Memory (using WGMMA).
Producer wants to overwrite that same Shared Memory with the next tile.
If the Producer overwrites sA while the WGMMA is still reading it, your math is
garbage. This is a Write-After-Read (WAR) hazard.

### Slide 35: How do we prevent wars

We create a barrier with an expected thread arrival count

The Consumer is busy reading the data, producer is spinning on the barrier

The Consumer finishes its last read instruction

Consumer executes instruction to decrease the thread arrival count

The barrier state flips and now the producer can write the new data

The instruction to decrease the arrival count is ....

### Slide 36: mbarrier.arrive.scope

Just decreases the pending arrival thread count of the barrier by count

### Slide 37: Compilers and reordering

Compilers often reorder the operations in order to get some efficiency gains

a = 1;

b = 2;

c = a + b;

Compilers might do b = 2 before a = 1 because they think it does not affect the
whole c = a + b operation which uses them

### Slide 38: Why this is fatal in async gpu ops

In multi-threaded code, reordering breaks correctness because different threads
can observe operations in different orders.
Thread a -
d[0][0] = 42.0f;
flag = 1;
Thread b -
while (flag == 0);
float x = d[0][0];

### Slide 39: mbarrier.arrive.release.scope

The .release suffix provides release semantics. That means it acts as a one-way
hardware fence where memory writes above this line cannot reorder below it.

Use mbarrier.arrive.release after a thread finishes producing data that other
threads will consume:

### Slide 40: Destroying a barrier

It formally invalidates an mbarrier object (a 64-bit synchronization primitive),
effectively wiping the hardware's tracking of that barrier's state.

By invalidating the barrier, it frees up the specific shared memory address so it can
be safely overwritten or repurposed for other variables later in the kernel.

It specifically converts a generic pointer to a 32-bit shared memory offset to
ensure the GPU hardware addresses the correct local memory bank.

### Slide 41: The whole pipeline

1.   mbarrier.init: Threads create a barrier in shared memory set an expected
thread count and start in Phase 0
2.   mbarrier.expect_tx: producer threads tell the barrier to also expect a specific
number of bytes from an upcoming async operation
3.   cp.async.bulk : A thread launches cp.async (the copy) and then
mbarrier.try_wait.parity 0, spins until Phase 0 is done
4.   Flip: When all expected bytes (and threads) arrive, barrier auto-flips to phase 1
5.   Computation begins
6.   Once the computation ends we manually flip the phase and then start working on
the next k tile to reuse the barrier

### Slide 42: mbarrier.arrive_drop

It acts like a standard arrive operation, decrementing the pending arrival count for
the current phase.

But more importantly it also permanently decrements the expected arrival count
for the barrier. So if you are flipping the phase and trying to reuse the barrier the
"expected arrival count" will be the previous value minus the number of drops that
occurred

### Slide 43: .sem .expect_tx and .noComplete

arrive_drop.expect_tx: sets the expected transaction count and permanently and
temporarily decreases the arrival count by 1

arrive_drop.sem - sem can be .release or .relaxed. .relaxed Specifies that no
memory ordering is enforced and .release Ensures that all memory writes
performed by the thread before the arrival are visible to any thread that waits on
the barrier

.noComplete: instructs the hardware to perform the arrival (decrementing
pending/expected counts) without triggering the phase completion, even if the
conditions for completion (pending count reaching zero) are met.

### Slide 44: barrier.cluster

Standard barriers like bar.sync (__syncthreads) only synchronize threads within a
single block. They cannot coordinate producer/consumer blocks across a cluster.

barrier.cluster is required to synchronize across different thread blocks that are
co-scheduled on the same cluster. It also guarantees that any writes to DSMEM
made before the barrier are visible to all blocks in the cluster after the barrier

The key usecase is TMA Multicast. The way it works is that in the initial step, the
block 0 might be running but block 1 might not be running and if block 0 tries to write
into block 1 then the program will crash. By launching this you make sure that every
block is physically present.

### Slide 45: Two important instructions in barrier.cluster

barrier.cluster.arrive The thread signals it has reached the barrier. It does
not stop. It continues executing independent instructions (math, local register ops)
that don't depend on data from other blocks

barrier.cluster.wait this is a blocking instruction. The thread stalls here until
every other thread/block in the cluster has signaled arrive. Once this unblocks, you
are guaranteed that all data written by other blocks is now safe to read

On H100, Because Block A can write to Block B's memory, we need a barrier to
prevent Block B from reading before Block A has finished writing

### Slide 46: Asynchronous groups

When you launch a cp.async.bulk operation using bulk group it looks something
like this -

Bulk_group is attached to a cp.async.bulk instruction and this enable us to use
cp.async.bulk.commit_group and cp.async.bulk.wait_group

These are the instructions which we are gonna use for batching a bunch of
instructions and then using them

### Slide 47: cp.async.bulk.commit_group

The way pipeline works in here is that we launch a bunch of operations using
cp.async.bulk with bulk_group

Now there was no cp.async.bulk.commit_group operation which was done
before so these copies are uncommitted, we can commit i.e batch them into a
group, the point of batching them in a group is that we can later make other units
wait until the execution of N units is finished

Any instructions launched after this instructions belong to the next group or are
just instructions

We can have multiple groups with multiple cp.async.bulk operations

### Slide 48: cp.async.bulk.wait_group<N>

Once we have launched and committed a bunch of operations then we need
other units which are gonna use their outputs to wait

The wait_group<N> instruction waits until at most N committed groups remain
pending. For example, wait_group<0> waits for all groups to complete, while
wait_group<2> tells you to wait till only the two operations out of all the ones
you launched are still pending.

The count refers to groups still pending, not groups that have completed. So
wait_group<2> means "wait until only the 2 most recent committed groups
are still pending" all the older groups must been completed

### Slide 49: cp.async.bulk.wait_group.read

This stalls execution AND enforces an Acquire Fence.

This is really important for read-after-write scenarios because you risk reading the
older data

### Slide 50: namedbarriers

__syncthreads() is a whole-block rendezvous. Named barriers let you create
multiple independent synchronization points inside one block, so different subsets
of warps can coordinate without forcing unrelated warps to stop.

PTX explicitly allows different warps to use the same named barrier with different
operations, such as mixing .arrive and .sync to build producer/consumer pipelines.

a = barrier ID, b = participating thread count

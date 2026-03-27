# AGENTS.md

## Project identity

This project is the landing site and supporting content system for **CUDA Programming for NVIDIA H100s** by **Prateek Shukla**.

The course has already been created and shared with FreeCodeCamp. The current mission is to build a **distinctive, premium, technically serious website** around it. This website should not feel like a generic course page. It should feel like a high-signal artifact for serious GPU engineers, systems programmers, and performance-minded ML practitioners.

The role of any coding agent working in this repository is to:

1. understand the course accurately,
2. preserve the instructor’s tone and technical depth,
3. help build a world-class website and supporting assets around the course,
4. avoid flattening the material into beginner fluff or generic CUDA content.

---

## Primary objective for agents

Agents should treat this repository as a **technical course website project**, not just a frontend codebase.

That means every decision should support one or more of these outcomes:

* clearly communicate what makes this H100 course special,
* present the lessons in a compelling and elegant way,
* preserve the course’s advanced technical identity,
* make the code examples and lesson structure legible to ambitious learners,
* create a website that feels original, sharp, and high-status.

The site should feel closer to:

* a research-lab product launch,
* a systems engineering manifesto,
* a premium technical learning experience,

and **not** like:

* a commodity bootcamp landing page,
* a template-heavy SaaS page,
* a shallow “learn CUDA fast” marketing site.

---

## What this course is about

**CUDA Programming for NVIDIA H100s** is an advanced course focused on the architectural and programming model shift introduced by NVIDIA Hopper / H100.

The course is not merely about writing basic CUDA kernels. It is about helping learners build a correct mental model for:

* Hopper’s asynchronous execution model,
* thread block clusters and distributed shared memory,
* PTX and inline PTX,
* barriers and cross-proxy synchronization,
* Tensor Memory Accelerator (TMA),
* cuTensorMap,
* `cp.async.bulk`,
* Warp Group Matrix Multiply Accumulate (`wgmma`),
* kernel design for compute-bound workloads,
* and multi-GPU orchestration and communication on H100 systems.

This is a course about **how modern high-performance GPU systems actually work**, especially for AI workloads built around GEMMs, tensor cores, memory movement, synchronization, and distributed execution.

---

## Intended audience

Agents should assume the target audience is:

* engineers who already know C/C++,
* people with a functional understanding of older CUDA execution models,
* people familiar with matrix multiplication and memory layouts,
* ML engineers or systems engineers who want to understand why these kernels matter,
* ambitious learners willing to engage with PTX, hardware details, scheduling, and memory systems.

This is **not** a mass-market absolute beginner course.

The site should welcome motivated learners, but it must remain honest about the technical bar.

---

## Pedagogical philosophy

The course philosophy matters and should shape copy, UX, and content structure.

### Core teaching beliefs

* Learners need **mental models**, not just syntax.
* An AI assistant is part of the learning workflow.
* Students are not supposed to understand everything instantly.
* The course is a staircase, not a single moment of mastery.
* Code and architecture should reinforce each other.
* The point is not memorization. The point is seeing the machine correctly.

### Important tone cues

When generating copy, preserve these ideas:

* direct,
* high-agency,
* technically ambitious,
* anti-fluff,
* anti-handwavy,
* serious but energizing.

The instructor voice is not corporate, not sterile, and not fake-hype. It is confident, compressed, explanatory, and obsessed with mechanism.

---

## Course structure

The course contains 10 lessons. Agents should preserve this structure unless explicitly asked to reorganize presentation for UX.

### Lesson 1 — Introduction to H100s

Focus:

* H100 architecture,
* hardware components,
* memory and execution hierarchy,
* shift toward asynchronous execution.

### Lesson 2 — Clusters, Data Types, Inline PTX, Pointers

Focus:

* thread block clusters,
* distributed shared memory,
* PTX and inline PTX,
* state spaces,
* pointers and address-space conversion.

### Lesson 3 — Asynchronicity and Barriers

Focus:

* latency hiding,
* asynchronous execution,
* producer/consumer coordination,
* `mbarrier`,
* hazards like RAW/WAR,
* fences and synchronization patterns.

### Lesson 4 — cuTensorMap

Focus:

* TMA descriptors,
* tensor shapes and strides,
* swizzling,
* interleaving,
* L2 promotion,
* descriptor-driven async movement.

### Lesson 5 — `cp.async.bulk`

Focus:

* Hopper bulk async copy instructions,
* structured vs unstructured transfers,
* barrier-based completion,
* bulk groups,
* multicast,
* prefetching,
* async reductions.

### Lesson 6 — WGMMA Part 1

Focus:

* warpgroups,
* `wgmma.mma_async`,
* register vs shared-memory sourcing,
* `ldmatrix`,
* descriptors,
* tensor core dataflow.

### Lesson 7 — WGMMA Part 2

Focus:

* group commit/wait,
* `stmatrix`,
* FP8 behavior,
* packing,
* K-major constraints,
* sparse WGMMA.

### Lesson 8 — Kernel Design

Focus:

* compute-bound kernels,
* warp specialization,
* pipelining,
* circular buffering,
* persistent scheduling,
* epilogues,
* multi-kernel optimization.

### Lesson 9 — Multi GPU Part 1

Focus:

* NVLink,
* NVSwitch,
* node topology,
* interconnect bottlenecks,
* scaling beyond one GPU,
* P2P and MPI context.

### Lesson 10 — Multi GPU Part 2

Focus:

* Slurm,
* PMIx,
* NCCL initialization,
* communicators,
* collectives,
* data / tensor / pipeline / expert parallelism,
* distributed orchestration.

---

## Canonical technical themes

Agents should recognize that the website and project revolve around a few central ideas.

### 1. Asynchronous execution is the core paradigm shift

H100 is presented as a machine designed to keep compute busy by decoupling request from completion.

### 2. Data movement is first-class

The course repeatedly emphasizes that moving data is often harder and more important than doing the math.

### 3. Synchronization is about correctness under overlap

Barriers, fences, proxy semantics, and wait patterns are not side details. They are central.

### 4. Tensor core programming requires layout literacy

Swizzling, descriptors, packing, register layout, and tile geometry matter deeply.

### 5. Kernel design is about utilization

Warp specialization, pipelining, stage management, and scheduling exist to keep tensor cores near full utilization.

### 6. Multi-GPU is not optional at scale

The course frames distributed systems as necessary for modern training, not an optional appendix.

---

## Code references mentioned in the course

The slides explicitly reference the following code artifacts. Agents should treat these as high-priority anchors when they exist in the local repository.

### CUTLASS / Hopper-related references

* `sm90_pipeline.hpp`
* `sm90_gemm_tma_warpspecialized_pingpong.hpp`
* `sm90_gemm_tma_warpspecialized_cooperative.hpp`
* `sm90_mma_tma_gmma_ss_warpspecialized.hpp`
* `sm90_mma_tma_gmma_rs_warpspecialized.hpp`
* `sm90_tile_scheduler.hpp`
* `sm90_tile_scheduler_group.hpp`
* `sm90_tile_scheduler_stream_k.hpp`
* `sm90_tma_epilogue_warpspecialized`
* `sm90_builder.inl`

### Custom/example code references

* `fast.cu`
* `matmul_12.cuh`

If these files exist in the local `files/` directory or elsewhere in the repo, agents should inspect them carefully and use them as the most concrete implementation references for technical explanations, lesson-page content, code callouts, and architecture diagrams.

If these files do not exist exactly as named, agents should search for close variants before assuming they are missing.

---

## Expected agent behavior around the codebase

When working with the repository, agents should:

1. look for the slide-derived concepts in the code,
2. map each important code file to the lesson it supports,
3. identify reusable diagrams, snippets, or visual explanations for the website,
4. avoid changing technical meaning when simplifying language,
5. keep advanced terminology when it is necessary,
6. prefer precision over generic explanation.

### If reading CUDA / PTX code

Agents should pay special attention to:

* async copy pipelines,
* barriers and wait logic,
* shared memory layout,
* swizzle assumptions,
* WGMMA operand setup,
* descriptor construction,
* cluster semantics,
* scheduler logic,
* epilogue/store paths,
* multi-stage pipelines,
* distributed setup code for NCCL / PMIx / Slurm.

### If generating summaries

Do not reduce everything to “faster GPU training.”
Be explicit about the actual mechanisms.

Bad summary:

* “This course teaches H100 optimization techniques.”

Better summary:

* “This course teaches the asynchronous Hopper programming model, including TMA, `cp.async.bulk`, `mbarrier`, WGMMA, shared-memory layout, warp-specialized kernel design, and multi-GPU orchestration.”

---

## Website direction

The future website should feel:

* bold,
* original,
* deeply technical,
* minimal but not empty,
* premium,
* GPU-native,
* visually disciplined.

### Visual direction

Agents should bias toward:

* dark, high-contrast visual language,
* typography that feels technical and cinematic,
* strong section rhythm,
* motion that feels intentional and structural,
* diagrammatic storytelling,
* a sense of depth, hardware, flow, and data movement.

### Avoid

* cheesy neon cyberpunk overload,
* generic devtool gradients,
* overused startup aesthetics,
* cluttered cards everywhere,
* fake benchmark chest-thumping without substance,
* stock illustrations,
* shallow “learn X in Y minutes” vibes.

### Good visual metaphors

* pipelines,
* execution lanes,
* memory movement,
* tiles,
* descriptors,
* synchronization points,
* node/mesh/fabric structures,
* systems diagrams,
* wavefronts and staged execution.

---

## Website content priorities

The homepage and supporting pages should communicate these points clearly.

### 1. Why this course exists

Because modern GPU programming on Hopper is fundamentally different from older synchronous mental models.

### 2. Why H100 matters

Because Hopper introduces new primitives and a deeper asynchronous pipeline model that changes how high-performance kernels are built.

### 3. What learners will actually learn

Not just CUDA basics, but the mechanisms behind high-performance AI kernels and distributed systems on H100.

### 4. Why this course is different

Because it teaches the architecture, the PTX-level primitives, the scheduling logic, the descriptors, the math pipeline, and the systems context together.

### 5. What code anchors the material

CUTLASS SM90 kernels, custom matmul code, Hopper primitives, and distributed orchestration examples.

---

## Messaging guidelines for agents

### When writing headlines

Prefer:

* mechanism,
* specificity,
* weight,
* confidence.

Examples of the kind of direction to aim for:

* “Learn the asynchronous machine.”
* “From TMA to WGMMA.”
* “How Hopper actually moves data and does math.”
* “The programming model behind H100-class kernels.”

Avoid generic headline language like:

* “Master CUDA Today”
* “Level up your GPU programming”
* “The ultimate AI acceleration course”

### When writing body copy

Use concise but dense language. Favor sentences with technical meaning.
Do not write bloated ad copy.

### When explaining concepts

Start from purpose, then mechanism, then consequence.

Good pattern:

* what problem exists,
* what Hopper primitive solves it,
* what correctness/performance constraint comes with it,
* where it shows up in real kernels.

---

## Non-negotiable technical accuracy rules

Agents must not casually invent or blur details around:

* `mbarrier` semantics,
* proxy fences,
* TMA descriptors,
* `cp.async.bulk` behavior,
* WGMMA operand placement,
* FP8 packing and scaling,
* cluster rank vs block ID,
* PMIx / NCCL setup details,
* collective operation meaning.

If unsure, inspect the actual local code or source notes instead of guessing.

When there is a conflict between aesthetic simplicity and technical accuracy, favor accuracy.

---

## Recommended repository organization (if missing)

If the repo is being shaped from scratch, agents may suggest or help create a structure like:

* `app/` or `src/` for the site
* `content/` for lesson metadata and course copy
* `components/` for reusable UI
* `public/` for diagrams, visuals, thumbnails
* `data/` for structured lesson definitions
* `notes/` for extracted technical references
* `files/` for referenced CUDA/CUTLASS source materials
* `AGENTS.md` for agent context
* `CLAUDE.md` if a tool specifically expects that filename

If both `AGENTS.md` and `CLAUDE.md` are needed, `CLAUDE.md` can be a lightweight pointer that says Claude should read `AGENTS.md` as the main project brief.

---

## Suggested structured data agents may create

Agents may create and maintain structured metadata for the website from the course material, such as:

* lesson slug,
* lesson title,
* lesson summary,
* key primitives,
* code references,
* architecture topics,
* visual ideas,
* difficulty level,
* prerequisite links,
* glossary entries.

Example lesson fields:

```json
{
  "slug": "wgmma-part-1",
  "title": "WGMMA Part 1",
  "summary": "Introduces warpgroup matrix multiply-accumulate on Hopper, including warpgroups, operand sourcing, ldmatrix, and descriptor-driven tensor core execution.",
  "key_primitives": ["wgmma.mma_async", "ldmatrix", "wgmma descriptor"],
  "code_refs": ["sm90_mma_tma_gmma_ss_warpspecialized.hpp", "sm90_mma_tma_gmma_rs_warpspecialized.hpp"],
  "difficulty": "advanced"
}
```

---

## How agents should help with the website

Agents should be especially useful for:

### Content work

* writing technical landing-page copy,
* writing lesson summaries,
* generating FAQs,
* building concept glossaries,
* drafting section architecture,
* creating code-to-concept mappings.

### Design work

* proposing homepage structure,
* inventing visually distinct interactions,
* building animated sections that explain pipelines,
* creating tasteful motion systems,
* designing lesson index pages and code reference pages.

### Engineering work

* implementing the site,
* creating content schemas,
* wiring MDX or JSON-backed lesson content,
* optimizing responsive layouts,
* building reusable components for technical storytelling.

---

## Specific UX ideas agents may explore

Agents may propose experiences such as:

* animated async pipeline hero sections,
* lesson timelines showing the staircase of concepts,
* architecture maps that link H100 primitives together,
* code-reference pages that map source files to lessons,
* glossary overlays for terms like TMA, WGMMA, DSMEM, PMIx, NCCL,
* diagrams showing data movement from HBM → L2 → shared memory → tensor cores,
* multi-GPU topology visualizations for NVLink / NVSwitch / rails / collectives.

These should remain elegant and not become noisy gimmicks.

---

## Tone and brand constraints

The brand should feel like:

* elite technical education,
* uncompromising systems clarity,
* modern AI infrastructure literacy,
* engineering craft.

The brand should not feel like:

* mass-ed startup marketing,
* hype-first influencer content,
* a copied documentation site,
* a generic Tailwind template with course cards.

---

## Working assumptions agents should keep

* The instructor is Prateek Shukla.
* The course already exists.
* FreeCodeCamp distribution is already handled.
* The next major deliverable is the website.
* The repo may contain slide-derived text and referenced CUDA/CUTLASS code.
* The local `files/` directory is important and should be inspected when available.
* The website should be unique enough to stand out as a flagship technical course project.

---

## If a coding agent is starting work fresh

Read this file first.
Then do the following:

1. inspect the repo structure,
2. inspect the local `files/` directory,
3. identify which referenced CUDA/CUTLASS files are present,
4. map code files to lessons,
5. propose an information architecture for the site,
6. preserve technical fidelity while improving presentation.

If asked to build the website immediately, start by producing:

* homepage architecture,
* visual design direction,
* lesson data model,
* first-pass hero / curriculum / code-reference sections.

---

## If `CLAUDE.md` is required

Use the following minimal approach:

```md
# CLAUDE.md

Read `AGENTS.md` first. It contains the full project brief, course context, technical scope, messaging rules, and guidance for building the CUDA Programming for NVIDIA H100s website.
```

---

## Final instruction to all agents

Do not make this project feel generic.

This course is about one of the most interesting programming model shifts in modern GPU systems. The site should reflect that.

Build for technical people.
Preserve the mechanism.
Respect the architecture.
Make it feel inevitable, sharp, and world-class.


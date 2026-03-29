---
title: "Lesson 10 - Multi GPU Part 2"
lesson_number: "10"
lesson_slug: "multi-gpu-part-2"
instructor: "Prateek Shukla"
course: "CUDA Programming for NVIDIA H100s"
source_page: "pages/lesson-10.html"
source_slide_pdf: "H100-Course/slides/10. Multi GPU  Part 2.pdf"
published_lesson_page: "/pages/lesson-10.html"
published_markdown_path: "/markdown/lessons/lesson-10-multi-gpu-part-2.md"
topics:
  - "Slurm"
  - "PMIx"
  - "NCCL"
  - "Collectives"
  - "Parallelism"
code_refs: []
generated_from:
  - "pages/lesson-10.html"
  - "H100-Course/slides/10. Multi GPU  Part 2.pdf"
---

# Lesson 10 - Multi GPU Part 2

This file combines the published lesson page with full slide-deck text so agents can fetch and search it directly.

## Sources

- Lesson page: `pages/lesson-10.html`
- Slide deck: `H100-Course/slides/10. Multi GPU  Part 2.pdf`
- Published lesson URL: `https://cudacourseh100.github.io/pages/lesson-10.html`
- Published markdown URL: `https://cudacourseh100.github.io/markdown/lessons/lesson-10-multi-gpu-part-2.md`

## Lesson Summary

Lesson 10 is the software side of the cluster. Slurm allocates the job, PMIx introduces ranks to one another, NCCL turns that bootstrap state into communicators and routes, and the parallelism strategies of modern training map their tensor movement onto those collectives.

## Why This Lesson Matters

**The communication fabric finally gets a control plane, a bootstrap path, and a runtime API.**

Lesson 9 explained the hardware. Lesson 10 explains how that hardware becomes a usable distributed execution engine: who launches the ranks, how they discover each other, how the communicator is formed, and which collective or parallelism pattern actually fits the workload.

## Topics

- Slurm
- PMIx
- NCCL
- Collectives
- Parallelism

## Lesson Facts

- **Course Position:** Lesson 10 of 10
- **Main Shift:** Distributed execution becomes a launch, discovery, and communication protocol problem.
- **Companion Deck:** `10. Multi GPU Part 2.pdf`
- **Repo Note:** No local Slurm / PMIx / NCCL source files are published in `files/`, so this page stays anchored to the course deck and the lesson 9 systems context.

## Key Takeaways

- Slurm allocates and launches, PMIx exchanges bootstrap metadata, and NCCL turns that metadata into GPU communication paths.
- `CUDA_VISIBLE_DEVICES` and local rank are part of correctness, not just convenience, because every task sees a filtered view of the node.
- Data, tensor, pipeline, and expert parallelism are really communication-pattern choices built out of collectives and point-to-point transfers.

## Web Lesson Content

### How the distributed stack fits together

The final lesson is about separation of responsibilities. A training job is not "NCCL doing everything." Slurm decides where the work runs, PMIx gives isolated ranks a way to publish and retrieve bootstrap metadata, and NCCL builds the communicator and issues the collective GPU work once the participants know who they are.

#### Allocation and launch

Slurm controls nodes, GPUs, CPUs, memory, task placement, and the basic launch environment for each process.

#### Discovery and communication

PMIx handles metadata exchange. NCCL uses that state plus topology discovery to construct rings, trees, and direct paths for collective communication.

> **The clean summary from the course:** Slurm allocates, PMIx introduces peers, and NCCL moves tensors efficiently across the cluster.

### Slurm is the cluster control plane, not the tensor communication layer

Slurm's job is resource allocation, placement, environment setup, and process launch. It knows the job shape, the node list, and which devices are assigned to each task, but it does not by itself solve rank-to-rank discovery or collective GPU communication.

| Responsibility | What it gives you | Why it matters |
| --- | --- | --- |
| Resource allocation | Nodes, GPUs, CPUs, memory, and wall-clock limits. | Defines the physical footprint of the distributed job. |
| Task launch | Process startup across the allocated nodes, often via `srun --mpi=pmix`. | Replaces slower or more ad hoc host-by-host bootstrap flows. |
| Device isolation | `CUDA_VISIBLE_DEVICES` and cgroup-level device filtering. | Ensures each task sees only the GPUs it was allocated. |

The notes also emphasize the hierarchy of terms: job, task, rank, local rank, and namespace. Those are not vocabulary trivia. They are the handles the rest of the startup path uses to bind processes to devices and to each other.

```text
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1

srun --mpi=pmix python train.py
```

### PMIx solves the moment after launch and before communication

A newly started rank knows it exists, but it does not automatically know where rank 0 lives, which GPU or NIC the others are bound to, or where the bootstrap data for the communicator has been published. PMIx acts as the temporary data exchange layer for that phase.

| Operation | Purpose | Typical use in this flow |
| --- | --- | --- |
| `PMIx_Put` | Publish bootstrap metadata. | Rank 0 posts IDs or connection state into the PMIx store. |
| `PMIx_Commit` | Push local data so it becomes visible at the desired scope. | Makes the published bootstrap state readable by peers. |
| `PMIx_Get` | Retrieve data that another participant published. | Follower ranks fetch the NCCL unique ID or peer metadata. |
| `PMIx_Fence` | Synchronization point before dependent startup continues. | Prevents some ranks from racing ahead before bootstrap state is globally visible. |

Local rank is the especially important bridge into CUDA. Because Slurm filters and re-indexes visible devices, the process usually maps `PMIX_LOCAL_RANK` onto the GPUs it can see after filtering, not onto the raw physical numbering on the node.

> **Debugging detail worth remembering:** if every task sees only one filtered device, every task may report it is running on GPU 0. In that context, PCI bus IDs are often more trustworthy than the visible CUDA index.

### NCCL turns bootstrap metadata into a communicator and a route plan

Before collectives can run, every participant needs to join the same communicator. The slides describe the common pattern clearly: rank 0 creates an `ncclUniqueId`, publishes it through PMIx, peers fence and retrieve it, and then every rank calls `ncclCommInitRank` with the same unique ID, world size, and rank.

```text
ncclUniqueId id;
if (global_rank == 0) {
  ncclGetUniqueId(&id);
  // publish through PMIx
}

// synchronize, retrieve, initialize
ncclCommInitRank(&comm, world_size, id, global_rank);
```

Initialization is expensive because the ID is only the entry point. NCCL still has to build the actual transport graph: which peers are local, which routes should prefer NVSwitch, which NICs should carry inter-node jumps, and whether the communication structure should look more like a ring or a tree.

#### Why the ID matters

Every participant must possess the exact same unique ID, otherwise they are not joining the same communicator.

#### Why the fence matters

PMIx visibility rules are real. Peers cannot safely fetch communicator bootstrap state until rank 0 has published and committed it at the correct scope.

Teardown matters too. `ncclCommDestroy` invalidates the communicator and frees the scratch space, mappings, and transport state built during initialization.

### Collectives are the real language of multi-GPU tensor movement

The lesson frames H100 clusters in terms of collective dataflow patterns, not manual send/receive. Broadcast, Reduce, AllReduce, AllGather, ReduceScatter, and All-to-All each solve a different tensor movement problem, and those patterns map directly onto different training strategies.

| Primitive | What it does | Typical use |
| --- | --- | --- |
| Broadcast | One root rank sends the same data to every rank. | Weight initialization, checkpoint restore, shared metadata distribution. |
| Reduce | All ranks contribute values; one root receives the reduced result. | Centralized consumption of a reduced tensor. |
| AllReduce | All ranks contribute and all receive the reduced result. | Gradient synchronization in data parallel training. |
| AllGather | Every rank contributes a shard and all ranks receive the concatenated whole. | Reconstructing full activations or parameters from shards. |
| ReduceScatter | Reduce first, then distribute disjoint slices of the reduced result. | Sharded gradient or activation flows, especially in tensor and sequence parallel paths. |
| All-to-All | Every rank sends a different chunk to every other rank. | Expert parallel routing and token exchange. |

The notes also emphasize that the operation itself is only part of the story. Datatype, reduction operator, communicator, and CUDA stream are part of the call signature because NCCL is meant to compose with the rest of the CUDA execution model rather than live outside it.

### The four major parallelism patterns are really communication-pattern choices

The final conceptual move of the course is to tie AI training strategies back to the collectives they require. Data, tensor, pipeline, and expert parallelism are not just framework buzzwords. Each one implies a specific pattern of tensor replication, sharding, and communication.

#### Data parallelism

Replicate the model on every GPU and split the data. The defining collective is `AllReduce`, with `Broadcast` often used for initialization.

#### Tensor parallelism

Split large matrices across GPUs. This introduces `AllGather`, `AllReduce`, and `ReduceScatter` depending on the sharding direction and whether sequence parallelism is in play.

#### Pipeline parallelism

Split the model vertically by layers. Communication becomes neighbor-to-neighbor point-to-point traffic as micro-batches move through the pipeline.

#### Expert parallelism

Route tokens to whichever GPU hosts the chosen expert. This is where the system starts to look like large-scale `All-to-All` traffic instead of simple replicated gradient exchange.

The sequence-parallel discussion in the slides is especially important because it shows how an `AllReduce` can often be decomposed into `ReduceScatter + AllGather`, leaving room to do useful work on the sharded intermediate state rather than replicating everything immediately.

> **The final mental model:** once you know the primitive each parallelism strategy requires, you can start reasoning clearly about whether the bottleneck is bandwidth, latency, topology, or buffering rather than treating "distributed training" as one opaque step.

### Practical guidance

1. **Keep the layers of responsibility straight.** Job scheduling, bootstrap metadata exchange, and collective execution solve different problems even though they appear in one startup path.
2. **Bind GPUs using the filtered local view.** Local rank plus visible device count are usually the correct inputs after Slurm has applied device filtering.
3. **Treat communicator setup as a real phase.** The unique ID is only the beginning; topology discovery and route construction are where startup cost comes from.
4. **Choose collectives from the tensor pattern, not from habit.** AllReduce is not the answer to every distributed step once the model is sharded in different ways.
5. **See the course as one staircase.** The kernel lessons explain how one GPU stays busy. The final two lessons explain how many GPUs become one training machine.

#### Glossary

| Term | Definition |
| --- | --- |
| Slurm | Cluster workload manager and scheduler that allocates resources and launches tasks. |
| PMIx | Process bootstrap and metadata exchange interface used for scalable peer discovery after launch. |
| `ncclUniqueId` | Bootstrap token shared across ranks so they can join the same NCCL communicator. |
| AllReduce | Collective where all ranks contribute data and all ranks receive the reduced result. |
| Expert parallelism | Distributed routing strategy where tokens are sent to whichever device hosts the selected experts. |

### Course wrap-up

This final lesson closes the loop: Hopper primitives, kernel pipelines, schedulers, node fabrics, and distributed orchestration are all part of the same machine model. The course is complete, but the site now keeps the whole path connected so you can revisit any step without dropping out of the main UI.

## Full Slide Deck Text

Extracted from `H100-Course/slides/10. Multi GPU  Part 2.pdf` with `pdftotext -layout`. Total slides: 57.

### Slide 1: Multi GPU

Part 2
Prateek Shukla

### Slide 2: Slurm

SLURM is an open-source workload manager and job scheduler designed for
Linux-based high-performance computing clusters.

It allocates the GPUs, CPUs, and memory. It knows where your job is running but
doesn't necessarily know how your application talks to itself across those nodes.

In the old days, mpirun would use SSH to connect to every node, check hostnames,
and exchange keys. On large clusters (e.g., 64+ nodes), this "handshake" could take
minutes

With srun --mpi=pmix, Slurm launches the processes on all nodes simultaneously.

### Slide 3: PMIx (Process Management Interface for Exascale)

When Slurm launches your 1,000 processes. Now rank 5 (Process #5) knows it is
alive, but it doesn't know the IP address of Rank 0, or how to talk to Rank 999. They
are isolated islands.

Before heavy training begins, the processes need to exchange technical details (IP
addresses, GPU handles). PMIx provides a temporary database.

PMIx_Put: A process pushes its metadata (e.g., host IP, port, CUDA IPC handle) to
the local PMIx server.

PMIx_Commit: Pushes local data to the global namespace.

PMIx_Get: Other processes query this data to discover their peers.

### Slide 4: Why SLURM and PMIx are important together

Slurm alone handles resource allocation (nodes, GPUs, CPUs) and task
placement, but it historically relied on older PMI versions that don't scale well
beyond ~10k GPUs.

PMIx is the modern, exascale-ready process management interface (replacing
PMI-1/PMI-2).

Slurm + PMIx integration allows:

Direct launch of processes without mpirun in many cases.

Efficient job info exchange (rank, node list, endpoints) at scale.

Tight integration with NCCL via direct NCCL-PMIx plugins.

### Slide 5: SLURM scripts

It is a simple text file (usually ending in .sh or .sbatch) that tells Slurm two things:

What resources you need (Time, GPUs, CPUs, Memory).

What you want to do once you get those resources (Run Python, compile code,
etc.).

Once we write the script we need to launch using sbatch script.sh

Once the script is launched, Slurm logs into the compute node, sets up the
environment, and runs the commands listed in the script.

### Slide 6: Terminology

Job: The Job is the highest-level unit of work in Slurm. It represents the resource
allocation. When you run sbatch or salloc, you are creating a Job.

Task: A Task is a single process of the running application. It is the number of
processes which run per node/gpu.

Rank: rank is simply the ID of a task, unless specified it refers to the global rank
which is the rank in the whole job.

Local rank: This is the unique ID of a task within a specific node. Ranges from 0 -
Number of tasks run on a single node

Namespace: for every rank inside a job this is the jobID. Unique for every rank in the
job

### Slide 7: Device isolation

If you set slurm to use a single GPU per task then slurm doesn't just politely ask your
program to use one GPU; it forces it at the operating system level using two
mechanisms:

CUDA_VISIBLE_DEVICES: Slurm sets this environment variable inside the process.
This prevents CUDA to see any other devices than the ones listed here. Works for a
single node

Slurm uses the Linux Control Groups (cgroups) feature, specifically the device
allowlist where it creates a "sandbox" for Task 0.

This means that if you debug and every GPU have different task then when you look
at logs every single error would point to GPU 0 even if its not actually GPU 0

### Slide 8: A sample script

### Slide 9: How SLURM scripts launch PMIx for kernels

When you run srun --mpi=pmix ./my_cuda_kernel_app, Slurm doesn't just fork
processes blindly. It uses PMIx to orchestrate the startup.

After this we have to initialize PMIx using these steps:

1.   Step 1: Connecting to slurm daemon
2.   Step 2: Initialize PMIx, get namespace and global rank
3.   Step 3: Get local rank,
4.   Step 4: Use the Local Rank to select the specific GPU/GPUs
5.   Step 5: finish the process

### Slide 10: Initializing PMIx and connecting process to slurm

### Slide 11: Getting identity (global rank and local rank)

### Slide 12: Slurm typically restricts GPU visibility per task by setting CUDA_VISIBLE_DEVICES,

so your process sees a filtered, re-indexed set of GPUs starting at device 0.

PMIx provides the node-local rank (PMIX_LOCAL_RANK), which is the natural index to
use when distributing tasks across the GPUs assigned to that node.

Use cudaGetDeviceCount to learn how many devices are visible after filtering,
and optionally parse CUDA_VISIBLE_DEVICES if you need original physical indices.

For deeper debugging, query PCI bus IDs (cudaDeviceGetPCIBusId) so you can
correlate "visible device 0" with the actual hardware GPU Slurm allocated.

### Slide 13: Binding to local GPU

After this launch the kernels and once its done, use PMIx_Finalize(NULL, 0); in
the host function to tell SLURM we are done

### Slide 14: So what is NCCL

NVIDIA Collective Communications Library. It's a GPU-focused library that makes
it fast (and relatively painless) to do collective communication between GPUs

In distributed training, GPUs constantly need to exchange tensors
(gradients/parameters). NCCL provides highly optimized primitives for that, such
as Allreduce, Allgather, ReduceScatter etc.

It integrates at the CUDA stream + pointer level: you hand NCCL device pointers
and a cudaStream_t, and NCCL schedules its own GPU work (kernels +
memcopies + network ops) on that stream, so it composes with your kernels via
normal CUDA stream ordering.

When you do an allReduce operation the CPU just launches a NCCL Kernel onto
the GPU stream.

### Slide 15: Setting up communication with NCCL

Before any communication happens, NCCL needs a way for all processes to find a
common meeting point. This meeting point is represented by a piece of data called
the NCCL Unique ID.

Only one process can generate this ID, Every single process that wants to be part of
the group must possess this exact same ID.

Rank 0 asks NCCL to create a new Unique ID, takes it and puts it into the PMIx
Key-Value Store (KVS) so that other processes can read.

Rank 0 tells PMIx to commit the data, ensuring it is actually pushed out to the server
and visible to the network. A fence ensures everyone can read the data and then
Followers check the Key Value Store and call NCCL initialization function

### Slide 16: ncclGetUniqueId

When we begin the GPUs don't know about any other GPU in their surroundings
and how to reach them using nvlink/infiniband.

To communicate with other GPUs NCCL needs to build a 'communicator'

NCCL calls ncclGetUniqueId to create ncclUniqueId struct and logs in the ip
address of the first GPU. Now every GPU comes in and logs its own IP in this struct
to communicate

We will see how to use PMIx to do this

### Slide 17: usage

### Slide 18: NCCL is very fast, but it doesn't know where the other GPUs are when it starts.

Different processes are running on different GPUs. They have separate memory.
Process B cannot see variables inside Process A.

We need to ensure that data committed by participants is "collected" and made
available according to scope rules (e.g., PMIX_GLOBAL). We create a fence where
nobody returns until everyone has arrived.

Then every single process needs to find the key value store, for that we must
specify who owns it. To do that we use the nspace of the processes and set rank 0.

We use PMIx_Get to get the key which we posted through PMIx_Put, this key serves
no security or verification purpose; it is purely for addressing the data.

Once that's done we copy the NCCL unique ID bytes out of PMIx's temporary buffer
into our own local ncclUniqueId id variable.

### Slide 19: ncclCommonInitRank

This is the most expensive and complex function in the setup phase. It is where the
ID maps to Hardware Connections, we log IDs of nrank ranks to the GPU0

Every rank looks at the ncclUniqueId struct. It extracts the IP:Port of Rank 0.

Every rank creates a standard TCP socket and connects to Rank 0.

Rank 0 waits until it receives exactly nranks connections.

It's a barrier. If Rank 799 is slow, everyone waits here.

Once the TCP connections are alive, they don't send data yet. They send metadata
about their rank, connection type, device etc.

### Slide 20: The next steps

Rank 0 acts as the architect. It analyzes the global graph to find the path with the
highest bandwidth and lowest latency for operations like AllReduce.

It prioritizes NVSwitch for intra-node communication. It selects specific InfiniBand
NICs for inter-node jumps. It decides whether to build Rings (latency-optimized) or
Trees (bandwidth-optimized) for the communication pattern.

Rank 0 sends the specific "routing table" (who to send to, who to receive from) back
to every rank. Inside a single node, Ranks map each other's memory. Configure the
NVSwitch to allow direct GPU-to-GPU memory access

Ranks identify their paired peers on other nodes and RDMA Handshake to ensure
they can write directly into each other's memory buffers without CPU involvement.

### Slide 21: ncclCommDestroy

Its the way to dismantling hardware paths we created earlier

It marks the communicator object as invalid for future kernel launches. If the GPU is
currently executing an NCCL kernel (like an AllReduce inside a CUDA stream),
ncclCommDestroy will not yank the memory out from under it immediately. It relies
on internal reference counting

The 4MB-8MB scratchpad buffers in HBM that were allocated during Init are freed.

The staging areas in CPU RAM used for PCIe transfers are released

NvLink Mappings and Infiniband Queue Pairs are destroyed

### Slide 22: NCCL Collective Primitives

On an H100 cluster we don't think in terms of send and receive. We think in terms of
patterns of data manipulation across the NVLink fabric. Here are big 6 primitives
Broadcast - one GPU copies its data to all GPUs.
Reduce - all GPUs combine data, result lands on one GPU.
AllReduce - all GPUs get the reduced sum of everyone's data.
AllGather - GPUs start with fragments, end with the full combined buffer.
ReduceScatter - reduce first, then split output so each GPU gets a different chunk.
All-to-All - every GPU sends a piece to every other GPU, and receives pieces back

### Slide 23: 4 Important types of parallelism in AI

There are 4 types of parallelism which one deals with when working with AI
applications in these multi GPU systems

1.   Data Parallelism
2.   Tensor Parallelism
3.   Pipeline Parallelism
4.   Expert Parallelism

Let's discuss each one of them in detail

### Slide 24: Data parallelism

Data parallelism is the most common distributed training strategy used in deep
learning. It allows you to train a model faster by distributing the input data across
multiple GPUs, while replicating the model itself on every device

A copy of the entire model is placed on every GPU. The global batch of training data
is split into smaller "mini-batches". Each GPU receives its unique slice of data. Every
GPU performs forward and backward passes on its own piece of data. Before the
optimizer updates the model weights, the system must ensure that every model
copy stays identical. The gradients from all GPUs are aggregated (usually
averaged).

For Data Parallelism, strictly speaking, you primarily rely on AllReduce, but
Broadcast is essential for initialization

### Slide 25: NCCL operations involved

AllReduce is the single most important operation in Data Parallel training. It
happens at the end of every backward pass. GPU 1 has gradient g1, GPU 2 has
gradient g2, etc. We need every GPU to have the average gradient (1/N)gi.

AllReduce sums up vectors from all GPUs and distributes the result back to all
GPUs.

Broadcast is typically used at the very beginning of training or when resuming
from a checkpoint. During weight initialization, Rank 0 initializes the parameters
and Broadcasts them to all other ranks. This guarantees that at step 0, all replicas
are mathematically identical

### Slide 26: Tensor Parallelism

If you try to load a model that is too big for a single GPU, the program will simply
crash before computation even begins. Tensor Parallelism is used when the
weight matrices are so big that you can't do matrix multiplies in a single GPU

If you are training the model (not just running it), the memory requirements triple
or quadruple. You don't just store the weights; you need to store gradients,
optimizer states, activations.

What we do here is that we divide our matrix into pieces and distribute it to
different GPUs for computation. Once the computation is done then we put all
the results together.

### Slide 27: Sequence parallelism

In standard Tensor Parallelism, we split the heavy Matrix Multiplications (Linear
Layers) across GPUs. However, we do not split the operations that occur between
the Linear Layers, specifically: LayerNorm, Dropout, GeLU/SiLU activations

In standard TP, the output of the AllReduce (after a Linear Layer) is a replicated
tensor. This means if you have 8 GPUs, every single GPU stores an identical copy of
the full activation matrix (size: [Sequence Length, Hidden Dimension]) just to perform
LayerNorm or Dropout.

LayerNorm acts on the Hidden Dimension of a single token. It is independent across
the Sequence Dimension. Thus we don't need to replicate the full sequence on every
GPU. We can partition the sequence across GPUs for these operations.

### Slide 28: Breaking the allreduce on TP

Standard TP uses AllReduce to synchronize the output of a matrix multiplication.
Mathematically, AllReduce is actually a composition of two primitive operations:

AllReduce = ReduceScatter + AllGather

When using SP, instead of doing the AllReduce atomically (NCCL fusion), SP injects
the LayerNorm/Dropout operations inside the communication loop

We perform the ReduceScatter, stop there, do our LayerNorm on the sharded data,
and only AllGather later when we absolutely need the full data for the next Matrix
Multiplication.

### Slide 29: NCCL operations involved in TP

Broadcast: Used in Column-wise sharding to copy complete input matrices to each
worker.
All-Gather: Used in Column-wise sharding to combine the results after multiplication,
Used in Sequence Parallelism during the forward pass to combine sharded sequence
chunks.
All-Reduce: Used in Row-wise sharding to sum the results from different workers for
the final result. In a standard Transformer block, Tensor Parallelism implies two
all-reduce per transformer block (one for the Attention block and one for the
Feedforward block).
Reduce-Scatter: Used in Sequence Parallelism to reduce gradients while scattering
along the sequence dimension. In TP, Reducescatter is used for gradients during the
backward pass of a column linear operation.

### Slide 30: Pipeline parallelism

If Tensor Parallelism is slicing a single layer horizontally, Pipeline Parallelism is
slicing the model vertically (splitting the stack of layers).

By splitting the model layers across different GPUs, each GPU only needs to store
the parameters and optimizer states for its specific chunk of layers.

If you simply passed one large batch through the pipeline, most GPUs would sit
idle waiting for their turn. Pipeline Parallelism breaks a large batch of data into tiny
micro-batches. As soon as GPU 1 finishes the first micro-batch, it passes it to
GPU 2 and immediately starts working on the second micro-batch

Because PP only requires communication between "neighbors" in the pipeline, it
only uses point to point send receive operations.

### Slide 31: Expert parallelism

Unlike standard "dense" models where every parameter is used for every input,
MoE models activate only a small subset of parameters (experts) for each input.

This is good for cost. You can increase the parameter count by 100x (adding more
experts), but if you only select the top-2 experts per token, the compute cost
remains relatively low.

The model learns that certain "experts" are good at coding, while others are good
at creative writing. Expert parallelism ensures that when a coding token comes in, it
is routed specifically to the device holding the "coding expert."

### Slide 32: How expert parallelism works

A small network decides if the token needs expert A or expert B etc.

Experts are distributed across GPUs. GPU 1 holds Expert A & B. GPU 2 holds
Expert C & D. GPU 3 holds Expert E & F and so on

If a token on GPU 1 needs Expert D (which is on GPU 2), it must be sent over the
network to GPU 2. Eventually every GPU needs to send data to every other GPU.
This is called "all to all" communication.

Specialized libraries like DeepEP expertise on such situations by bundling data to
send efficiently across the inter node connections first, then distributing it quickly
locally with faster nvlink connections.

### Slide 33: General format of NCCL operations

sendbuff, recvbuff, count, datatype tell the GPU how many elements to move
and how to interpret them.

op tells what math operation to perform

comm is the communication handle storing all the metadata for the node and cluster

stream helps specify the asynchronous behavior

### Slide 34: Broadcast

The Broadcast operation in NCCL is a collective communication primitive where a
single "Root" GPU sends a tensor of data to every other GPU in the
communicator.

Generally it was implemented using a ring/tree operation but with H100 DGX, the
Root GPU sends the data packet once to the NVSwitch. The NVSwitch itself
physically replicates the packet to all 7 other ports (in a single node)
simultaneously.

Because the switch handles the replication, the time to broadcast to 8 GPUs is
roughly the same as sending to just 1 GPU.

### Slide 35: ncclBroadcast

root: this is the rank of GPU holding the data we want to broadcast
sendbuff: this is the location of the buffer of data we want to broadcast
recvbuff: The destination pointer on all GPUs (including the root)(if they are same
then in place reduction is enabled).
Datatype: if you pass ncclFloat8e4m3 or ncclFloat8e5m2, NCCL switches to using
Hopper-specific intrinsic instructions
comm: The communicator object
stream: The CUDA stream

### Slide 36: The math of the operation

Let P={0,1,...,N1} be the set of N processes (GPUs) in the communicator. Let V(i)
represent the data vector held by process i.

### Slide 37: Reduction

Reduction takes an array of data from every participating GPU, combines them
using a mathematical operator (like sum or max), and stores the single
consolidated result on a specific GPU (or all of them).

In previous systems, GPUs had to pass data between each other (like a bucket
brigade) to sum it up. In the DGX H100, the NVSwitch chips themselves perform
the math, offloading the work from the GPUs.

the general reduction (ncclReduce) typically uses NVLSTREE (introduced in NCCL
2.18+) to handle the specific flow of reducing data to a single root, especially
when scaling across multiple nodes

### Slide 38: ncclReduce

You use ncclReduce when the consumer of the data is centralized (usually Rank
0) and the other ranks do not need the result to proceed immediately.

It is generally used in the last layer during inference when only Rank 0 needs the
full logits to perform argmax or top-k sampling.

The count is the number of elements each GPU contributes. The root receives
exactly count elements.

### Slide 39: Let P be the number of processors (GPUs), indexed 0 to P1. Each processor p

holds an input vector Vp of size N:
Vp=[vp,0,vp,1,...,vp,N1]

### Slide 40: AllReduce

AllReduce is one of the most important operations in training of AI models

Combines data from N GPUs -> get a global result -> distributes to N GPUs

All 8 GPUs read their local gradients from their HBM3 memory

Instead of sending data to a peer GPU, all 8 GPUs simultaneously push their data
onto the NVLink lanes, targeting the NVSwitch chips

The data goes to switch where the reduction happens "in flight" the switch
immediately multicasts this result back to all 8 GPUs simultaneously.

The result lands directly in the destination buffer in HBM3 on all GPUs.

### Slide 41: ncclAllReduce

This is one of the most popular and important operation for deep learning use
cases

AllReduce is the right primitive when every participant both contributes data and
needs the final reduced result, such as gradient or metric aggregation across
workers

### Slide 42: ncclRedOp_t

This is the mathematical operation for reduction(ncclSum, ncclProd, ncclMin,
ncclMax, ncclAvg)

ncclSum is the only operator that is fully optimized for every hardware path on
H100. If you use this NCCL will offload all the operation to nvswitch

ncclMin and ncclMax are also offloaded from hardware(fp8)

ncclProd will offload all the operations to tensor cores because nvswitch can't do
floating point multiplications

ncclAvg is done in two parts, the sum part done in nvswitch and the division is
done in GPUs

### Slide 43: Math of AllReduce

Let there be P processes. Each process k holds a vector Vk of size N. Let Vk[i]
denote the i-th element of the vector on process k.

The goal of AllReduce is for every process k to end up with a result vector R,
where the i th element is the sum (or other associative operator ) of that element
across all processes

this is used to sum gradients: Vk is the gradient vector calculated by GPU k, and R
is the total gradient used to update the model.

### Slide 44: ReduceScatter

ReduceScatter is an operation where every GPU starts with a full buffer of gradients,
and the goal is to sum these buffers across all GPUs, but then scatter the results so
that each GPU ends up holding only a distinct "slice" of the final summed vector
Input: Each GPU has a vector [A,B,C,D].
Math: Element-wise sum across all GPUs
Output:
GPU 0 holds Sum(Part A)
GPU 1 holds Sum(Part B)
GPU 2 holds Sum(Part C)
GPU 3 holds Sum(Part D)

### Slide 45: ncclReduceScatter

size_t recvcount: This is the element count of the output buffer, not the total
input buffer. You get this by total elements in gradients buffer / number of GPUs
Total Elements Reduced = recvcount * nranks

If you have a total vector size of 100 and 3 GPUs: You cannot divide 100 by 3 evenly
(33.33...). NCCL does not support "jagged" arrays where GPU 0 gets 34, and GPU 1
gets 33. Thus you must pad your data to the next multiple of the GPU count

### Slide 46: The math of ReduceScatter

### Slide 47: AllGather

AllGather is an operation where every GPU starts up with different pieces of the
gradients and after the transformation every GPU gets the complete copy of the
whole data

All 8 GPUs push their gradients into the NVSwitch fabric simultaneously

The NVSwitch combines the data and route the result to all the GPUs

Total bus utilization is maximized. You get closer to the theoretical 900 GB/s
aggregate throughput because you aren't waiting for a "ring" to cycle

### Slide 48: sendcount is the number of elements this rank contributes (the size of its local

slice). Every rank ends up with all ranks' slices in recvbuff.

### Slide 49: Mathematically

### Slide 50: All to All

Every GPU holds data for every other GPU in the node and it transfers that data to
its respective GPU

The way it works is that each GPU receives its piece of data from every GPU and it
sends data which belongs to every GPU

Unlike AllReduce, which often uses complex Ring or Tree algorithms and hardware
offloading (SHARP), AllToAll on this system is physically simpler but
bandwidth-intensive.

### Slide 51: Assume nranks = N GPUs in a communicator.

You have a send buffer logically split into N blocks: block j is the data you want to
send to rank j.

After all-to-all, each rank has a receive buffer split into N blocks: block i contains
what rank i sent to you.

So it's like doing Nx(N1) point-to-point transfers (plus self) but coordinated and
optimized as one collective.

### Slide 52: The MOE problem

Using NCCL's standard AllToAll for MoE models is generally avoided because it was
designed for static, bulk synchronous communication, whereas MoE routing is
inherently dynamic, sparse, and latency-sensitive.

Major problems with NCCL All to all include NCCL being CPU-driven (Host API),
NCCL using static data sizes, NCCL requiring data in contagious blocks and just the
blocking nature of the instruction.

Specialized libraries like DeepEP (and the underlying NVSHMEM primitives) are
preferred because they allow for device-initiated, fine-grained data movement that
can handle the irregular traffic patterns of expert routing without stalling the GPU.

### Slide 53: ncclGroupStart ncclGroupEnd

In a distributed system a lot of times you have a single CPU thread managing
multiple GPUs. when CPU launches a NCCL instruction the CPU will be blocked by
NCCL if its a blocking call. Most NCCL collective calls (like ncclAllReduce) are
asynchronous from the CPU, The more blocking parts are often communicator
initialization (ncclCommInitRank),

If one host thread is issuing NCCL calls for multiple GPUs, launching them
one-by-one can create partial states where some participants have started an
operation while others haven't been issued yet which can lead to hangs

ncclGroupStart()/ncclGroupEnd() let you batch a set of NCCL calls: NCCL collects
the calls during the group and then submits them together at ncclGroupEnd(). This
avoids problems caused by partially launched NCCL work.

### Slide 54: ncclSend

It is a non-blocking operation used to send data from one specific GPU (the
sender) to another specific GPU (the receiver). It is almost always paired with a
corresponding ncclRecv on the target device.

peer: The rank (ID) of the destination GPU.

NCCL finds the fastest physical path available between the two GPUs, can be
using NVLink, PCIe or IB RDMA

### Slide 55: ncclRecv

Tells a specific GPU to allocate space in its memory and wait for incoming data
from a designated "peer." It is asynchronous, meaning the CPU resumes control
immediately after the command is enqueued on the CUDA stream
peer: The rank (ID) of the GPU that is sending the data.
A ncclRecv must have a matching ncclSend on the source GPU. If you are doing
multiple P2P transfers, they must be enqueued in a compatible order across the
different GPUs to avoid circular dependencies

### Slide 56: Prevent all GPUs from launching the same instruction

### Slide 57: Probably end here

---
title: "Lesson 9 - Multi GPU Part 1"
lesson_number: "9"
lesson_slug: "multi-gpu-part-1"
instructor: "Prateek Shukla"
course: "CUDA Programming for NVIDIA H100s"
source_page: "pages/lesson-9.html"
source_slide_pdf: "H100-Course/slides/9. Multi GPU.pdf"
published_lesson_page: "/pages/lesson-9.html"
published_markdown_path: "/markdown/lessons/lesson-09-multi-gpu-part-1.md"
topics:
  - "NVLink"
  - "NVSwitch"
  - "ConnectX"
  - "UVA"
  - "P2P"
code_refs: []
generated_from:
  - "pages/lesson-9.html"
  - "H100-Course/slides/9. Multi GPU.pdf"
---

# Lesson 9 - Multi GPU Part 1

This file combines the published lesson page with full slide-deck text so agents can fetch and search it directly.

## Sources

- Lesson page: `pages/lesson-9.html`
- Slide deck: `H100-Course/slides/9. Multi GPU.pdf`
- Published lesson URL: `https://cudacourseh100.github.io/pages/lesson-9.html`
- Published markdown URL: `https://cudacourseh100.github.io/markdown/lessons/lesson-09-multi-gpu-part-1.md`

## Lesson Summary

Lesson 9 leaves the single-GPU kernel and asks what real training systems have to move through. The answer is a fabric: NVLink and NVSwitch inside the node, ConnectX and rail-aligned InfiniBand outside it, plus CUDA peer access and UVA as the lowest software layer above that hardware.

## Why This Lesson Matters

**Performance stops being only a kernel problem and becomes a fabric problem.**

Hopper kernels explain how one GPU stays busy. Multi-GPU systems explain how many GPUs stay coordinated once memory, bandwidth, topology, and software discovery all live outside a single SM.

## Topics

- NVLink
- NVSwitch
- ConnectX
- UVA
- P2P

## Lesson Facts

- **Course Position:** Lesson 09 of 10
- **Main Shift:** The bottleneck moves from one kernel pipeline to the communication fabric between accelerators.
- **Companion Deck:** `9. Multi GPU.pdf`
- **Systems Focus:** `NVLink`, `NVSwitch`, `ConnectX-7`, rails, and `cudaDeviceEnablePeerAccess`

## Key Takeaways

- Multi-GPU training exists because even one extremely fast H100 is nowhere near enough for frontier-scale workloads.
- Inside the node, NVLink and NVSwitch create the fast path. Outside the node, ConnectX and the network fabric dominate the scaling story.
- CUDA P2P and UVA expose basic direct access, but higher-level communication is still needed once the topology stops being trivial.

## Web Lesson Content

### Why scale out at all

The slides start with a blunt calculation. If training roughly costs six FLOPs per parameter per token, then a 1T parameter model over 10T tokens lands near `6 x 10^25` FLOPs. Even pretending you sustain 1000 TFLOP/s, that is measured in centuries on one device. Multi-GPU systems are not a luxury. They are the only way the training budget enters human time.

#### More throughput

Additional GPUs cut wall-clock time by spreading the math over more tensor cores and more HBM capacity.

#### More memory surface

Bigger models and optimizer state stop fitting on one device, so the system needs many pools of HBM plus a fabric fast enough to keep them coherent enough for training.

> **The point of the lesson:** once you leave one GPU, the machine is no longer just SMs, tensor cores, and shared memory. The machine is now a communication hierarchy.

### A DGX H100 node is already a small distributed system

The notes describe the DGX H100 node as eight SXM5 H100 GPUs connected by four NVSwitch chips, plus eight ConnectX-7 NICs and four OSFP cages for the external compute fabric. That matters because the node is not one GPU with accessories. It is a coordinated mesh of accelerators and network endpoints.

| Component | Role | Why it matters |
| --- | --- | --- |
| 8 x H100 SXM5 GPUs | Compute and HBM capacity inside the node. | They are the endpoints that need to exchange activations, gradients, and model state. |
| 4 x NVSwitch | Internal full-speed switching fabric. | Lets every GPU reach every other GPU without collapsing to a slow PCIe-style path. |
| 8 x ConnectX-7 NICs | External network interfaces for the compute fabric. | Once traffic leaves the node, these devices own the critical bandwidth transition. |
| 4 x OSFP cages | Physical outbound high-speed links. | Expose the node to the rail-aligned InfiniBand fabric. |

The lesson also separates compute fabric from storage fabric. The OSFP path is for inter-node GPU communication. Dataset ingest and checkpoint traffic use a different set of interfaces and a different network design.

### Inside the node, NVLink and NVSwitch define the fast path

H100's fourth-generation NVLink gives each GPU about 900 GB/s of bidirectional bandwidth. The notes frame that as the reason GPUs can skip the CPU path and communicate directly out of HBM instead of bouncing through ordinary host-centric links.

#### NVLink

The direct GPU-to-GPU link layer. The notes call out 18 individual NVLink links per H100 and treat the bandwidth jump over PCIe as one of the main reasons modern multi-GPU systems behave differently.

#### NVSwitch

The node-wide switching layer that makes the interconnect fully non-blocking. Every GPU can reach every other GPU at full speed without being trapped in a daisy-chain story.

The important conceptual transition is that the node starts to feel like a fabric rather than eight isolated devices. The notes even emphasize that switch-side operations matter, because the switch layer is not just a dumb cable crossbar.

> **The bandwidth cliff:** inside one DGX node, GPUs talk over NVLink. Once traffic leaves the box, it hits the NIC and the external network. The whole scaling story is shaped by how well you manage that drop in bandwidth and added distance.

### Outside the node, ConnectX, rails, and the network topology decide whether scaling holds

ConnectX-7 is the exit point from the node. The notes describe eight compute NICs per DGX H100 and a rail-aligned system where GPU 0 across all nodes shares one rail, GPU 1 shares another, and so on. This creates eight parallel traffic planes instead of one giant mixed queue.

| Concept | What it means | Why it matters |
| --- | --- | --- |
| Rail alignment | Each GPU index across the cluster maps onto its own independent network plane. | Reduces interference and keeps traffic patterns cleaner at the leaf layer. |
| Leaf / spine / core | A hierarchical InfiniBand fat-tree that expands from one scalable unit to larger pods. | Determines path diversity, oversubscription behavior, and cluster-wide reachability. |
| Adaptive routing | Switch hardware chooses among multiple viable paths based on congestion. | Prevents flows from piling onto one hot uplink when alternatives are available. |
| SHIELD and in-network logic | Hardware features that help isolate faults or offload parts of communication behavior. | Scaling fails fast if the fabric cannot handle faults and reductions efficiently. |

The notes describe SuperPOD-style deployments as rail-optimized, three-tier fabrics built from Quantum-2 InfiniBand switches. The exact numbers matter less than the lesson's main point: once you scale beyond one node, topology is no longer background detail. It is part of performance engineering.

### CUDA P2P and UVA expose the basic direct path, but not the whole communication problem

CUDA gives two key mechanisms at this level. Unified Virtual Addressing means one pointer space can identify whether data lives in host memory or a GPU's HBM. Peer access then allows one GPU to directly access another GPU's memory over the available fabric.

#### `cudaDeviceEnablePeerAccess`

Enables direct peer access. The notes stress that it is unidirectional, so bidirectional access needs the call on both devices.

#### `cudaMemcpyPeer`

Issues direct memory copies between devices without staging through host memory, assuming peer access and the right topology are available.

But the lesson also argues that raw P2P is not the whole answer. On an HGX H100 board, the topology is a switch-connected mesh, not a straight line. A manual load/store path from one SM to another GPU's HBM does not automatically saturate the fabric or solve multi-party coordination. That is exactly why the next lesson turns to NCCL, PMIx, and Slurm.

```text
// Conceptual CUDA-side flow
cudaDeviceEnablePeerAccess(peer, 0);
cudaMemcpyPeer(dst_ptr, dst_device, src_ptr, src_device, bytes);
```

### Practical guidance

1. **Measure the bandwidth hierarchy, not just the GPU.** Intra-node NVLink behavior and inter-node NIC behavior are very different scaling regimes.
2. **Read the node as a fabric.** NVSwitch, ConnectX, OSFP, rails, and topology are part of the machine your kernel actually runs inside.
3. **Use peer access deliberately.** Direct GPU memory access is powerful, but it is not an automatic replacement for collective libraries or topology-aware scheduling.
4. **Expect topology to shape performance.** Rail alignment, adaptive routing, and switch hierarchy determine whether communication stays balanced as the cluster grows.
5. **Treat lesson 9 as the hardware preface to lesson 10.** Once you understand the fabric, the orchestration stack makes much more sense.

#### Glossary

| Term | Definition |
| --- | --- |
| NVLink | Direct GPU-to-GPU interconnect used for high-bandwidth communication inside the node. |
| NVSwitch | Switch fabric that lets GPUs inside a node communicate with each other at full speed. |
| ConnectX-7 | The node's high-speed network interface for external GPU-to-GPU communication. |
| Rail | An isolated network plane that aligns the same GPU index across all nodes onto one communication path. |
| UVA | Unified Virtual Addressing, which lets the system determine memory location from the pointer value. |

### Continue the course

Lesson 9 explains the hardware and CUDA-side path. The final lesson moves into orchestration: how jobs launch across nodes, how ranks discover each other, and how collective communication actually gets initialized and scheduled.

## Full Slide Deck Text

Extracted from `H100-Course/slides/9. Multi GPU.pdf` with `pdftotext -layout`. Total slides: 26.

### Slide 1: Multi GPU

Prateek Shukla

### Slide 2: Let's train a 1T parameter model on 10T tokens with

H100
Empirical observation shows it takes roughly 6 operations (FLOPs) per parameter
for every token of training data (forward pass + backward pass + update).

Total FLOPs=6x(10^12)x(10^13)=6x10^25 FLOPs

Let's say we can get 1000tflops per second

That's around 1900 years!!

This is why we need multiple GPUs

### Slide 3: Multiple GPUs

Instead of spending thousands of years waiting for operations to complete we can
just connect multiple GPUs together and get higher throughput and run bigger
models. This is powered by two technologies

NVLink Gen 4

Direct GPU-to-GPU interconnect bypassing the CPU.

900 GB/s Bidirectional Bandwidth.

7x Faster than PCIe Gen 5.

NVSwitch

Enables every GPU in a node to talk to every other GPU at full speed

### Slide 4: H100 clusters vs older clusters

In traditional computing, GPUs are discrete units that communicate over slower
PCIe lanes. The H100 paradigm shifts this by creating a "mesh" where every GPU
can access the memory of every other GPU at extremely high speeds.

An 8-GPU cluster can behave as if it has a single pool of memory and compute
cores

This architecture allows systems to scale from 8 GPUs in a single server to
clusters of 256 GPUs (using the NVLink Switch System) that communicate at
native chip speeds.

### Slide 5: H100 DGX system or a node

It features 8 NVIDIA H100 GPUs using the SXM5 form factor.

The board includes 4 third-generation NVSwitches

The GPUs and switches are connected via fourth-generation NVLink, providing a
massive 900 GB/s bandwidth per GPU

8 Network interface cards(NICs) NVIDIA ConnectX-7: specialized processor
designed to offload networking tasks from the main CPU.

4x OSFP (Octal Small Form-factor Pluggable) cages

### Slide 6: The Nvidia H100 Superpod

An nvidia DGX h100 superpod have around 127-128 GPUs organized in 4
SUs(Scalable units) each having 32 DGX H100 nodes.

The number 32 allows for a perfectly balanced "Level 1" network using standard
switch port counts (typically 64 ports per switch).

The H100 SuperPod is designed around the fact that a single DGX H100 node
has 8 separate NVIDIA ConnectX-7 Network Interfaces (HCAs) for compute, one
for each GPU.

### Slide 7: Nvlink

Each H100 GPU features 900 GB/s of bidirectional bandwidth. This is
approximately 7x faster than the standard PCIe Gen5 interface used to connect
the GPU to the CPU. Nvlink allows GPUs to use this to communicate between
their HBM and skip the CPU path.

H100 uses 18 individual NVLink "links" to achieve this speed. Unlike previous
generations, the 4th Gen NVLink is more density-optimized, using only two
high-speed differential pairs per link (down from four), allowing more links to be
packed into the same space.

### Slide 8: Nvswitch 3

On a standard 8-GPU HGX board, four NVSwitch chips are used to connect all
eight H100 GPUs. 4 - 5 Nvlink ports are connected to a single switch

The switches create a fully non-blocking interconnect. This means every GPU can
talk to any other GPU at full 900 GB/s speed simultaneously, without waiting in line
for data lanes to clear

NVIDIA uses NVSwitch chips in external trays called NVLink Switch Systems.
These connect up to 256 H100 GPUs (32 server racks) into a single "SuperPOD"

The NVSwitch itself has ALUs which can perform additions/reductions in the switch
itself, this is one of the most important feature which results in massive performance
gains.

### Slide 9: Connectx

Inside the server box, GPUs talk via NVLink (900 GB/s). Once data needs to leave
the box to reach another server, it hits ConnectX-7. There are also 8 Compute
Network Interfaces (ConnectX-7) in a DGX.

This is the bottleneck. Your training speed is determined by how efficiently you
manage this 18x drop in bandwidth. ConnectX-7 exists to minimize the penalty of
leaving the box.

ConnectX7 allows the network to read/write GPU memory without the CPU
knowing.

The NIC works with the network switches to sum gradients in flight. This turns the
network traffic from O(N) (linear growth with cluster size) to O(1) (constant traffic).
This is the only reason large-scale training scales linearly

### Slide 10: The OSPF cages

These 4 physical cages provide 8 separate 400Gb/s network links (totaling 3.2
Tb/s of bandwidth) by using "Twin-port" technology. They connect internally to 8x
NVIDIA ConnectX-7 network cards.

This allows for GPU Direct RDMA, enabling GPUs to talk to GPUs in other DGX
nodes without burdening the system CPU.

The 4 OSFP cages are exclusively for the Compute Fabric. Storage traffic does
not touch those cables.

### Slide 11: The storage fabric

This helps in loading massive datasets into the GPUs and saving checkpoints

This uses separate PCIe cards located in the standard PCIe slots on the back of
the chassis, not the OSFP cages.

This is usually a Fat Tree or standard leaf-spine network. unlike the "Rail" network,
any DGX node needs to be able to talk to any storage array.

### Slide 12: The rail aligned system

In a standard network, a server might have one cable that carries traffic for all its
components. In a DGX SuperPOD, the network is physically split into 8 parallel,
isolated networks these are the "Rails."

Inside a Node, there are 8 GPUs (numbered 0 to 7). Rail 1 connects GPU 0 from
every single node in the cluster to the same set of switches. Rail 2 connects GPU 1
from every single node to a different set of switches. And so on

Traffic on Rail 1 never interferes with traffic on Rail 2 at the leaf switch level. This
creates 8 independent planes of connectivity across the entire cluster.

The rail system leverages NVIDIA's SHARP technology, which offloads data
operations to the network switches themselves.

### Slide 13: The rail optimized fat tree

The SuperPOD uses a 3-tier hierarchy (Leaf, Spine, Core) to connect the SUs
together

Layer 1: The Leaf Layer (Connection to Nodes)

Layer 2: The Spine Layer (Connection between SUs)

Layer 3: The Core / Super-Spine Layer (Maximum Scale)

These layers are nothing but a bunch of Quantumn 2 infiniband switches
connected together

### Slide 14: Quantum-2 QM9700 InfiniBand Switch

It provides ultra high bandwidth low latency GPU-GPU interconnect

It has 64 ports of 400Gb/s connectivity

When you look at the front of the switch, you will see 32 cages. These use the
OSFP (Octal Small Form-factor Pluggable) standard. Each OSFP cage actually
carries two separate 400Gb/s links. 32 Cages x 2 Links = 64 Logical Ports.

Since the QM9700 has 64 ports, it is the perfect size for a SU of 32 nodes.

32 Ports (Downlinks): Connect to the 32 Nodes in the SU (e.g., Rail 1 connects to
GPU 0 on all 32 nodes).

32 Ports (Uplinks): Connect "up" to the Spine Switches (Layer 2) to talk to other
SUs.

### Slide 15: The leaf layer

This is where the cables physically leave the back of the DGX H100 server.

Rail Segregation: There are 8 separate "planes" of switches:

Rail 1 Switches: Connect only to the 1st network card (GPU 0) of every node.

Rail 8 Switches: Connect only to the 8th network card (GPU 7) of every node.

The leaf switches handle traffic within the local Scalable Unit. If GPU 0 on Node 1
needs to talk to GPU 0 on Node 2 (in the same SU), traffic goes Node -> Leaf
Switch -> Node. It never needs to go higher up the chain.

The QM9700 uses SHARPv3, which is 32x more capable than the previous
generation, allowing it to handle complex AI math.

### Slide 16: Adaptive routing in the leaf layer

In the leaf layer, adaptive routing is critical for upstream traffic.

When a packet arrives at a leaf switch from a GPU and needs to travel to a different
SU, it must go up to a spine switch. In a non-blocking or even oversubscribed
fat-tree, there are multiple available spine switches it can use.

Instead of using a static hash (which always sends a specific flow to the same spine
switch, potentially causing collisions), the Quantum-2 switch hardware monitors the
queue depth and congestion levels of all uplink ports

The switch dynamically sends packets to the least congested spine switch link on a
per-packet or per-message basis. This ensures that even if one path to the spine is
clogged, traffic flows smoothly through others.

### Slide 17: The spine layer

The Spine layer connects the Leaf switches together. This allows nodes in SU-1 to
talk to nodes in SU-2.

Spine Groups: The spine switches are also organized into groups that align with
the rails.

Spine Group 1: Connects only to the Leaf switches that handle Rail 1.

Isolation: This ensures that traffic from Rail 1 never accidentally "leaks" over to
Rail 2's cables, which would cause congestion (blocking).

If GPU 0 in SU-1 needs to talk to GPU 0 in SU-2, the traffic flows: Node (SU1) ->
Leaf (Rail 1) -> Spine (Group 1) -> Leaf (Rail 1) -> Node (SU2)

### Slide 18: Adaptive routing in the spine layer

In the spine layer, adaptive routing manages traffic traversing the core of the
network.

Typically, in a standard 2-layer fat-tree, there is only one path "down" from a
specific spine switch to a specific destination leaf switch. However, adaptive
routing is still vital here for handling faults and multi-pathing if parallel links exist

If the buffers towards a specific leaf switch are full, the spine switches
communicate this backpressure. Adaptive routing features in Quantum-2 (like
SHIELD) help isolate this congestion so it doesn't spread to other unaffected traffic
flows in the spine.

### Slide 19: The Core / Super-Spine Layer (Maximum Scale)

For massive clusters (e.g., 127 nodes or more), a third layer is added.

Function: This layer connects multiple "Pods" (clusters of SUs) together.

Optics: At this level, the system often switches to 800G optical transceivers to
reduce the number of cables required, while logically splitting them back into two
400G links.

### Slide 20: SHIELD

In traditional InfiniBand networks, if a cable breaks or a link flaps, it takes around 5
- 30 seconds for the software controller to calculate a new routing table and then
push it to all the switches. This crashes entire training run

SHIELD moves this recovery logic directly into the switch hardware ASIC.

If a switch sees a link go down, it doesn't just drop packets. It immediately checks
for alternative valid paths in its local hardware table. If it finds one then it updates
its own table to avoid the bad node and goes to healthy neighbor, if it fails then it
sends a hardware signal to the neighbour so that no traffic is sent to it in future.

### Slide 21: P2p mechanism for communication between GPUs

The P2P (Peer-to-Peer) CUDA mechanism is a feature that allows two GPUs to
communicate without using CPUs

We can do p2p memory copy explicitly or we can get p2p direct access and the
whole transfer uses Nvlink.

Unified Virtual addressing enables this whole system

### Slide 22: Unified Virtual Addressing (UVA)

UVA allows the CPU and GPU to share a single virtual address space.

Before UVA: The CPU had its own memory pointers and the GPU had its own.
You had to manually manage which pointer pointed to which physical memory.

With UVA: The system determines specifically where the data is physically
located (in the system RAM or on the H100's HBM3 memory) based on the pointer
value alone.

You need to enable Peer access using cudaDeviceEnablePeerAccess otherwise
this feature can't work

### Slide 23: cudaDeviceEnablePeerAccess

Without this call the GPU might not use Nvlink and go through the PCIe

### Slide 24: A few important points

cudaDeviceEnablePeerAccess(peerDevice, 0) is unidirectional. If you need to
copy back and forth or access memory kernels on both sides, you must call it on
both devices.

If you don't enable this then you might fall back to a slower PCIe path

On H100 servers, this function will automatically utilize NVLink (900 GB/s) if
available; otherwise, it uses PCIe Gen5

### Slide 25: cudaMemCpyPeer

enables the direct transfer of data between the memory of two separate GPUs
without involving the host (CPU) memory

You must enable cudaDeviceEnablePeerAccess to use Nvlink for copies

### Slide 26: Truth about raw p2p

On an H100 HGX board, you don't just have 8 GPUs connected in a line. You
have a complex mesh connected via NVSwitches.

P2P: You have to manually manage which link you are traversing. If GPU 0 talks
to GPU 7, does it go direct? Does it hop through GPU 3?

The H100 has a bidirectional NVLink bandwidth of 900 GB/s. If you issue a
standard LD/ST (Load/Store) instruction across the NVLink fabric from an SM, you
are likely using a single distinct NVLink lane or a subset of them. You will struggle
to saturate that pipe. You are using a straw to drink from a firehose.

These are the reasons because of which we need libraries such as NCCL for
managing multi GPU connections

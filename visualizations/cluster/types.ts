/**
 * DGX H100 SuperPOD Topology Types
 * 
 * KEY ARCHITECTURAL DISTINCTION:
 * 
 * INTERNAL (Intra-Node):
 *   - NVSwitch (Cedar-7): Connects 8 GPUs within a single DGX H100
 *   - Uses NVLink 4.0 @ 900 GB/s per GPU
 *   - Enables direct GPU-to-GPU communication WITHIN the node
 * 
 * EXTERNAL (Inter-Node):
 *   - ConnectX-7 NICs: 8 NICs per node, one dedicated to each GPU
 *   - Uses InfiniBand NDR @ 400 Gb/s per port
 *   - Forms the "Compute Fabric" connecting nodes via leaf/spine switches
 *   - Rail-optimized: Each GPU's NIC connects to its own independent fat-tree
 */

export enum NodeType {
  DGX_H100 = 'DGX_H100',
  UFM_MGMT = 'UFM_MGMT',
  SWITCH_LEAF = 'SWITCH_LEAF',
  SWITCH_SPINE = 'SWITCH_SPINE'
}

export enum CableType {
  DAC = 'DAC', // Direct Attach Copper (Intra-rack, <3m)
  AOC = 'AOC'  // Active Optical Cable (Inter-rack, longer runs)
}

/**
 * ConnectX-7 Port Mapping
 * Each DGX H100 has 8 ConnectX-7 NICs in 4 OSFP cages (2 ports per cage)
 * Each port is dedicated to one GPU and connects to one rail's leaf switch
 */
export interface ConnectX7Port {
  portLabel: string;      // e.g., "OSFP 4 / P2" - physical cage and port
  nicId: string;          // e.g., "CX7-0" - ConnectX-7 NIC identifier
  gpuId: number;          // 0-7: Which GPU this NIC is dedicated to
  railId: number;         // 1-8: Which rail's fat-tree this connects to
  connectedLeafId?: string; // The leaf switch this port connects to
}

export interface DGXNode {
  id: string;
  suId: number;           // Scalable Unit ID (0-3)
  rackId: number;         // Physical rack number
  uPosition: number;      // Vertical position in rack (rack units)
  type: NodeType;
  ports: ConnectX7Port[]; // 8 ConnectX-7 ports for external fabric
}

/**
 * InfiniBand Switch (QM9700)
 * 64-port NDR switches used for both leaf and spine tiers
 */
export interface Switch {
  id: string;
  type: NodeType.SWITCH_LEAF | NodeType.SWITCH_SPINE;
  railId: number;         // 1-8: Which rail this switch belongs to
  suId?: number;          // Only for Leaf switches (one leaf per rail per SU)
  tier: 'Leaf' | 'Spine';
  portCount: number;      // 64 for QM9700
}

/**
 * InfiniBand Link
 * Connects nodes to leafs (downlinks) or leafs to spines (uplinks)
 */
export interface Link {
  source: string;
  target: string;
  railId: number;
  type: CableType;
  bandwidth: number;      // Gb/s (400 for NDR)
}

export interface ClusterData {
  nodes: DGXNode[];
  switches: Switch[];
  links: Link[];
}

/**
 * TOPOLOGY SUMMARY (Standard 4-SU SuperPOD):
 * 
 * Nodes: 128 (127 compute + 1 UFM management)
 * GPUs: 1,016 H100 (127 × 8)
 * 
 * Per Rail (8 total, independent fat-trees):
 *   - 4 Leaf Switches (one per SU)
 *   - 2 Spine Switches (shared across SUs)
 *   - Each leaf: 32 downlinks to nodes, 32 uplinks to spines
 * 
 * Total Switches:
 *   - 32 Leaf Switches (8 rails × 4 SUs)
 *   - 16 Spine Switches (8 rails × 2 spines)
 * 
 * Bisection Bandwidth per Rail:
 *   - 32 uplinks × 400 Gb/s = 12.8 Tb/s per rail
 *   - Total: 8 rails × 12.8 Tb/s = 102.4 Tb/s
 */

/**
 * DGX H100 SuperPOD Constants
 */

// Rail Color Coding - consistent across all visualizations
export const RAIL_COLORS: Record<number, string> = {
  1: '#ef4444', // Red
  2: '#f97316', // Orange
  3: '#eab308', // Yellow
  4: '#22c55e', // Green
  5: '#06b6d4', // Cyan
  6: '#3b82f6', // Blue
  7: '#8b5cf6', // Violet
  8: '#ec4899', // Pink
};

export const RAIL_NAMES: Record<number, string> = {
  1: 'Rail 1 → GPU 0',
  2: 'Rail 2 → GPU 1',
  3: 'Rail 3 → GPU 2',
  4: 'Rail 4 → GPU 3',
  5: 'Rail 5 → GPU 4',
  6: 'Rail 6 → GPU 5',
  7: 'Rail 7 → GPU 6',
  8: 'Rail 8 → GPU 7',
};

// Cluster Topology Constants
export const NODES_PER_SU = 32;           // Compute nodes per Scalable Unit
export const SU_COUNT = 4;                 // Number of Scalable Units
export const TOTAL_NODES = 127;            // 128 slots - 1 for UFM management
export const GPUS_PER_NODE = 8;            // H100 GPUs per DGX node
export const TOTAL_GPUS = TOTAL_NODES * GPUS_PER_NODE; // 1,016 GPUs

// Switch Configuration (NVIDIA QM9700)
export const SWITCH_PORTS = 64;            // Ports per QM9700
export const LEAFS_PER_SU = 8;             // One leaf switch per rail per SU
export const SPINES_PER_RAIL = 2;          // Spine switches per rail
export const TOTAL_LEAFS = LEAFS_PER_SU * SU_COUNT;  // 32 leaf switches
export const TOTAL_SPINES = SPINES_PER_RAIL * 8;     // 16 spine switches

// Bandwidth (InfiniBand NDR)
export const PORT_BANDWIDTH_GBPS = 400;    // Gb/s per NDR port
export const NVLINK_BW_PER_GPU = 900;      // GB/s NVLink bandwidth per GPU (internal)

/**
 * ConnectX-7 Port Mapping (from DGX H100 specifications)
 * 
 * Physical Layout (Rear View):
 *   Compute Tray has 4 OSFP cages, each with 2 ports (P1, P2)
 *   Module 1 (Left):  OSFP 1, OSFP 2
 *   Module 0 (Right): OSFP 3, OSFP 4
 * 
 * Each port connects to one ConnectX-7 NIC, which is dedicated to one GPU
 */
export const PORT_MAPPING_TEMPLATE = [
  // Module 0 (Right side, looking at rear)
  { label: 'OSFP 4 / P2', nic: 'CX7-0', module: 0, cage: 4, port: 2, gpu: 0, rail: 1 },
  { label: 'OSFP 3 / P2', nic: 'CX7-1', module: 0, cage: 3, port: 2, gpu: 1, rail: 2 },
  { label: 'OSFP 3 / P1', nic: 'CX7-2', module: 0, cage: 3, port: 1, gpu: 2, rail: 3 },
  { label: 'OSFP 4 / P1', nic: 'CX7-3', module: 0, cage: 4, port: 1, gpu: 3, rail: 4 },
  // Module 1 (Left side, looking at rear)
  { label: 'OSFP 1 / P2', nic: 'CX7-4', module: 1, cage: 1, port: 2, gpu: 4, rail: 5 },
  { label: 'OSFP 2 / P2', nic: 'CX7-5', module: 1, cage: 2, port: 2, gpu: 5, rail: 6 },
  { label: 'OSFP 2 / P1', nic: 'CX7-6', module: 1, cage: 2, port: 1, gpu: 6, rail: 7 },
  { label: 'OSFP 1 / P1', nic: 'CX7-7', module: 1, cage: 1, port: 1, gpu: 7, rail: 8 },
];

// Rack Layout
export const NODES_PER_RACK = 4;           // DGX nodes per compute rack
export const RACKS_PER_SU = 8;             // Compute racks per SU
export const U_HEIGHT_PER_NODE = 8;        // Rack units per DGX H100

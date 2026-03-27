import { 
  ClusterData, DGXNode, NodeType, Switch, Link, CableType, ConnectX7Port 
} from '../types';
import { 
  PORT_MAPPING_TEMPLATE, NODES_PER_SU, SU_COUNT, SPINES_PER_RAIL,
  SWITCH_PORTS, PORT_BANDWIDTH_GBPS
} from '../constants';

/**
 * Generates the complete SuperPOD topology data
 * 
 * Architecture:
 * - 8 independent fat-trees (one per rail)
 * - Each rail connects all GPU N's across all nodes
 * - 2-tier fat-tree: Leaf switches (per-SU) and Spine switches (global)
 */
export const generateClusterData = (): ClusterData => {
  const nodes: DGXNode[] = [];
  const switches: Switch[] = [];
  const links: Link[] = [];

  // ============================================
  // STEP 1: Generate Spine Switches (Global)
  // ============================================
  // 8 rails × 2 spines = 16 total spine switches
  // Spines provide inter-SU connectivity within each rail
  for (let rail = 1; rail <= 8; rail++) {
    for (let s = 1; s <= SPINES_PER_RAIL; s++) {
      switches.push({
        id: `SPINE-R${rail}-${s}`,
        type: NodeType.SWITCH_SPINE,
        railId: rail,
        tier: 'Spine',
        portCount: SWITCH_PORTS
      });
    }
  }

  // ============================================
  // STEP 2: Generate Scalable Units
  // ============================================
  let globalNodeIndex = 0;

  for (let su = 0; su < SU_COUNT; su++) {
    
    // --------------------------------------------
    // 2A: Generate Leaf Switches for this SU
    // --------------------------------------------
    // 8 leaf switches per SU (one per rail)
    for (let rail = 1; rail <= 8; rail++) {
      const leafId = `LEAF-SU${su}-R${rail}`;
      switches.push({
        id: leafId,
        type: NodeType.SWITCH_LEAF,
        railId: rail,
        suId: su,
        tier: 'Leaf',
        portCount: SWITCH_PORTS
      });

      // --------------------------------------------
      // 2B: Connect Leaf to Spines (Uplinks)
      // --------------------------------------------
      // Each leaf connects to both spines on its rail
      // This provides redundancy and full bisection bandwidth
      const railSpines = switches.filter(
        s => s.type === NodeType.SWITCH_SPINE && s.railId === rail
      );
      
      railSpines.forEach(spine => {
        // In reality, multiple uplink ports - simplified to single link here
        links.push({
          source: leafId,
          target: spine.id,
          railId: rail,
          type: CableType.AOC,  // Inter-SU always optical
          bandwidth: PORT_BANDWIDTH_GBPS
        });
      });
    }

    // --------------------------------------------
    // 2C: Generate Compute Nodes
    // --------------------------------------------
    for (let n = 0; n < NODES_PER_SU; n++) {
      const isManagementSlot = su === 0 && n === 0;
      const rackId = (su * 8) + Math.floor(n / 4); // 4 nodes per rack
      
      if (isManagementSlot) {
        // UFM Management Node - doesn't participate in compute fabric
        nodes.push({
          id: `UFM-MGMT`,
          suId: su,
          rackId: rackId,
          uPosition: n % 4,
          type: NodeType.UFM_MGMT,
          ports: []
        });
        globalNodeIndex++;
        continue;
      }

      const nodeId = `DGX-${String(globalNodeIndex).padStart(3, '0')}`;

      // --------------------------------------------
      // 2D: Map ConnectX-7 Ports
      // --------------------------------------------
      // Each DGX H100 has 8 ConnectX-7 NICs
      // Each NIC is dedicated to one GPU and connects to one rail
      const nodePorts: ConnectX7Port[] = PORT_MAPPING_TEMPLATE.map(template => ({
        portLabel: template.label,
        nicId: template.nic,
        gpuId: template.gpu,
        railId: template.rail,
        connectedLeafId: `LEAF-SU${su}-R${template.rail}`
      }));

      nodes.push({
        id: nodeId,
        suId: su,
        rackId: rackId,
        uPosition: n % 4,
        type: NodeType.DGX_H100,
        ports: nodePorts
      });

      // --------------------------------------------
      // 2E: Connect Node to Leafs (Downlinks)
      // --------------------------------------------
      // Each ConnectX-7 port connects to its rail's leaf switch
      nodePorts.forEach(port => {
        if (port.connectedLeafId) {
          links.push({
            source: nodeId,
            target: port.connectedLeafId,
            railId: port.railId,
            type: CableType.DAC,  // Within SU, typically copper
            bandwidth: PORT_BANDWIDTH_GBPS
          });
        }
      });

      globalNodeIndex++;
    }
  }

  return { nodes, switches, links };
};

/**
 * Calculate topology statistics
 */
export const getTopologyStats = (data: ClusterData) => {
  const computeNodes = data.nodes.filter(n => n.type === NodeType.DGX_H100);
  const leafSwitches = data.switches.filter(s => s.tier === 'Leaf');
  const spineSwitches = data.switches.filter(s => s.tier === 'Spine');
  
  const totalGPUs = computeNodes.length * 8;
  const linksPerRail = data.links.filter(l => l.railId === 1).length;
  
  // Bisection bandwidth calculation
  // Per rail: 32 node-to-leaf links × 400 Gb/s going up through spines
  const bisectionBWPerRail = (NODES_PER_SU * SU_COUNT - 1) * PORT_BANDWIDTH_GBPS / 1000; // Tb/s
  const totalBisectionBW = bisectionBWPerRail * 8;

  return {
    computeNodes: computeNodes.length,
    totalGPUs,
    leafSwitches: leafSwitches.length,
    spineSwitches: spineSwitches.length,
    totalLinks: data.links.length,
    linksPerRail,
    bisectionBWPerRail: bisectionBWPerRail.toFixed(1),
    totalBisectionBW: totalBisectionBW.toFixed(1)
  };
};

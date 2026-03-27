import React from 'react';
import { DGXNode, NodeType } from '../types';
import { RAIL_COLORS, PORT_MAPPING_TEMPLATE } from '../constants';

interface Props {
  node: DGXNode;
  highlightRail?: number | null;
  showEducationalInfo?: boolean;
}

/**
 * DGX H100 Chassis Visualization (Rear View)
 * 
 * Accurately depicts the compute tray with 4 OSFP cages containing
 * ConnectX-7 NICs for the external InfiniBand fabric.
 * 
 * IMPORTANT DISTINCTION:
 * - These are NOT NVSwitch/NVLink ports (that's internal, not visible here)
 * - These are ConnectX-7 InfiniBand NICs for inter-node communication
 */
const DGXChassis: React.FC<Props> = ({ node, highlightRail, showEducationalInfo = false }) => {
  
  if (node.type === NodeType.UFM_MGMT) {
    return (
      <div className="w-full h-28 bg-gray-800 border-2 border-dashed border-gray-600 rounded flex items-center justify-center">
        <div className="text-center">
          <h3 className="text-gray-400 font-mono text-sm">UFM MANAGEMENT</h3>
          <p className="text-[10px] text-gray-500 mt-1">Unified Fabric Manager</p>
        </div>
      </div>
    );
  }

  // Group ports by OSFP cage for rendering
  const cages = [1, 2, 3, 4].map(cageNum => {
    const cagePorts = node.ports.filter(p => {
      const template = PORT_MAPPING_TEMPLATE.find(t => t.label === p.portLabel);
      return template?.cage === cageNum;
    });
    return { cageNum, ports: cagePorts };
  });

  const renderOSFPCage = (cageNum: number, ports: typeof node.ports) => {
    // Sort ports: P1 first, P2 second
    const sortedPorts = [...ports].sort((a, b) => {
      const aPort = a.portLabel.includes('P1') ? 1 : 2;
      const bPort = b.portLabel.includes('P1') ? 1 : 2;
      return aPort - bPort;
    });

    return (
      <div key={cageNum} className="flex flex-col items-center">
        <span className="text-[7px] text-gray-500 mb-0.5">OSFP {cageNum}</span>
        <div className="flex gap-0.5 bg-gray-950 p-1 rounded border border-gray-700">
          {sortedPorts.map(port => {
            const isDimmed = highlightRail !== null && highlightRail !== port.railId;
            return (
              <div
                key={port.portLabel}
                className={`w-3 h-5 rounded-sm transition-all duration-300 cursor-pointer
                  ${isDimmed ? 'opacity-15' : 'opacity-100'}`}
                style={{ 
                  backgroundColor: RAIL_COLORS[port.railId],
                  boxShadow: isDimmed ? 'none' : `0 0 6px ${RAIL_COLORS[port.railId]}40`
                }}
                title={`${port.portLabel}\n${port.nicId} → GPU ${port.gpuId}\nRail ${port.railId} → ${port.connectedLeafId}`}
              />
            );
          })}
        </div>
        <span className="text-[6px] text-gray-600 mt-0.5">
          {sortedPorts.map(p => `G${p.gpuId}`).join(' ')}
        </span>
      </div>
    );
  };

  return (
    <div className="relative w-full bg-gray-900 rounded-md border border-gray-700 overflow-hidden 
                    group hover:border-green-500/50 transition-colors">
      
      {/* Chassis Header */}
      <div className="bg-gray-800 px-2 py-1 flex justify-between items-center border-b border-gray-700">
        <span className="text-xs font-mono font-bold text-gray-300">{node.id}</span>
        <span className="text-[9px] text-green-500 tracking-wider font-medium">DGX H100</span>
      </div>

      {/* Main Panel - Rear View */}
      <div className="p-2 flex gap-2 h-[88px]">
        
        {/* Compute Tray - ConnectX-7 Ports */}
        <div className="flex-1 bg-gray-950 rounded border border-gray-800 p-1.5 flex flex-col">
          {/* CORRECTED LABEL: This is ConnectX-7, NOT NVSwitch */}
          <div className="text-[7px] text-gray-500 mb-1 text-center tracking-wide">
            COMPUTE FABRIC · ConnectX-7 NICs
          </div>
          
          {/* Module Layout */}
          <div className="flex-1 flex items-center justify-center gap-3">
            {/* Module 1 (Left) */}
            <div className="flex gap-1">
              {renderOSFPCage(1, cages[0].ports)}
              {renderOSFPCage(2, cages[1].ports)}
            </div>
            
            {/* Divider */}
            <div className="w-px h-8 bg-gray-700" />
            
            {/* Module 0 (Right) */}
            <div className="flex gap-1">
              {renderOSFPCage(3, cages[2].ports)}
              {renderOSFPCage(4, cages[3].ports)}
            </div>
          </div>
          
          <div className="text-[6px] text-gray-600 text-center mt-1">
            InfiniBand NDR · 400 Gb/s per port
          </div>
        </div>

        {/* IO Tray - Storage/Management */}
        <div className="w-16 bg-gray-950 rounded border border-gray-800 p-1 flex flex-col items-center justify-center gap-1">
          <span className="text-[6px] text-gray-600">STORAGE</span>
          <div className="w-full h-2 bg-purple-900/40 rounded border border-purple-700/30" />
          <div className="w-full h-2 bg-purple-900/40 rounded border border-purple-700/30" />
          <span className="text-[6px] text-gray-600 mt-1">MGMT</span>
          <div className="w-full h-2 bg-blue-900/40 rounded border border-blue-700/30" />
        </div>
      </div>

      {/* Educational Overlay */}
      {showEducationalInfo && (
        <div className="absolute inset-0 bg-black/90 flex items-center justify-center p-2 opacity-0 
                        group-hover:opacity-100 transition-opacity pointer-events-none">
          <div className="text-[8px] text-gray-300 space-y-1">
            <p className="text-green-400 font-bold">DGX H100 Compute Fabric</p>
            <p>• 8 ConnectX-7 NICs (one per GPU)</p>
            <p>• Each connects to its rail's leaf switch</p>
            <p>• Enables GPU-direct RDMA across nodes</p>
            <p className="text-gray-500 mt-2">NVSwitch (internal) connects GPUs within node</p>
          </div>
        </div>
      )}

      {/* Power Indicators */}
      <div className="absolute bottom-1 left-2 flex gap-0.5">
        {[1, 2, 3, 4, 5, 6].map(i => (
          <div key={i} className="w-1.5 h-0.5 bg-green-500 rounded-full opacity-60" />
        ))}
      </div>
    </div>
  );
};

export default DGXChassis;

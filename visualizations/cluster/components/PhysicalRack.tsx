import React from 'react';
import { DGXNode } from '../types';
import DGXChassis from './DGXChassis';

interface Props {
  rackId: number;
  nodes: DGXNode[];
  activeRail: number | null;
}

const PhysicalRack: React.FC<Props> = ({ rackId, nodes, activeRail }) => {
  // Sort nodes by U position descending (top of rack = highest U)
  const sortedNodes = [...nodes].sort((a, b) => b.uPosition - a.uPosition);
  const emptySlots = 4 - sortedNodes.length;

  return (
    <div className="w-72 flex-shrink-0 flex flex-col bg-gray-900 border border-gray-800 rounded-lg">
      {/* Rack Header */}
      <div className="bg-gray-800 text-gray-400 text-xs font-mono py-2 px-3 
                      border-b border-gray-700 rounded-t-lg flex justify-between items-center">
        <span className="tracking-wider">RACK {String(rackId).padStart(2, '0')}</span>
        <span className="text-[10px] text-gray-600">{sortedNodes.length} nodes</span>
      </div>
      
      {/* Nodes */}
      <div className="p-2 space-y-2 flex-grow bg-gradient-to-b from-gray-900 to-gray-950">
        {sortedNodes.map(node => (
          <DGXChassis 
            key={node.id} 
            node={node} 
            highlightRail={activeRail}
            showEducationalInfo={true}
          />
        ))}
        
        {/* Empty Slots */}
        {Array.from({ length: emptySlots }).map((_, i) => (
          <div 
            key={`empty-${i}`}
            className="h-24 border border-dashed border-gray-800 rounded flex items-center justify-center"
          >
            <span className="text-gray-700 text-[10px] font-mono">EMPTY SLOT</span>
          </div>
        ))}
      </div>
      
      {/* Rack Footer - PDU indicators */}
      <div className="h-6 bg-gray-800 border-t border-gray-700 rounded-b-lg 
                      flex items-center justify-center gap-4 px-3">
        <div className="flex items-center gap-1">
          <span className="text-[8px] text-gray-600">PDU A</span>
          <div className="w-1.5 h-1.5 rounded-full bg-green-500" />
        </div>
        <div className="flex items-center gap-1">
          <span className="text-[8px] text-gray-600">PDU B</span>
          <div className="w-1.5 h-1.5 rounded-full bg-green-500" />
        </div>
      </div>
    </div>
  );
};

export default PhysicalRack;

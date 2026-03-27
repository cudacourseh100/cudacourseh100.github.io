import React, { useMemo, useEffect, useRef } from 'react';
import { select, zoom, ZoomBehavior } from 'd3';
import { ClusterData, NodeType } from '../types';
import { RAIL_COLORS, SU_COUNT } from '../constants';

interface Props {
  data: ClusterData;
  activeRail: number | null;
  width: number;
  height: number;
}

/**
 * Hierarchical Fat-Tree Visualization
 * 
 * Shows the rail-optimized topology as a proper tiered layout:
 * - Top tier: Spine switches (2 per rail)
 * - Middle tier: Leaf switches (1 per rail per SU)
 * - Bottom tier: DGX nodes (grouped by SU)
 * 
 * When a rail is selected, only that rail's independent fat-tree is highlighted
 */
const LogicalGraph: React.FC<Props> = ({ data, activeRail, width, height }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const gRef = useRef<SVGGElement>(null);

  const layout = useMemo(() => {
    const padding = { top: 60, bottom: 40, left: 60, right: 60 };
    const usableWidth = width - padding.left - padding.right;
    const usableHeight = height - padding.top - padding.bottom;
    
    // Tier Y positions
    const spineY = padding.top + usableHeight * 0.1;
    const leafY = padding.top + usableHeight * 0.4;
    const nodeY = padding.top + usableHeight * 0.85;
    
    // Filter by active rail
    const activeSpines = data.switches.filter(s => 
      s.tier === 'Spine' && (activeRail === null || s.railId === activeRail)
    );
    const activeLeafs = data.switches.filter(s => 
      s.tier === 'Leaf' && (activeRail === null || s.railId === activeRail)
    );
    const activeLinks = data.links.filter(l => 
      activeRail === null || l.railId === activeRail
    );
    
    // Position spines
    const spinePositions = new Map<string, { x: number; y: number }>();
    activeSpines.forEach((spine, i) => {
      const x = padding.left + (i + 0.5) * (usableWidth / activeSpines.length);
      spinePositions.set(spine.id, { x, y: spineY });
    });
    
    // Position leafs (grouped by SU)
    const leafPositions = new Map<string, { x: number; y: number }>();
    const leafsPerSU = activeLeafs.length / SU_COUNT;
    activeLeafs.forEach((leaf, i) => {
      const suIndex = leaf.suId ?? 0;
      const inSuIndex = i % leafsPerSU;
      const suWidth = usableWidth / SU_COUNT;
      const suStart = padding.left + suIndex * suWidth;
      const x = suStart + (inSuIndex + 0.5) * (suWidth / leafsPerSU);
      leafPositions.set(leaf.id, { x, y: leafY });
    });
    
    // Position nodes (simplified - just show counts per SU)
    const nodePositions = new Map<string, { x: number; y: number }>();
    const computeNodes = data.nodes.filter(n => n.type === NodeType.DGX_H100);
    
    // Group by SU for positioning
    const nodesBySU = new Map<number, typeof computeNodes>();
    computeNodes.forEach(node => {
      if (!nodesBySU.has(node.suId)) nodesBySU.set(node.suId, []);
      nodesBySU.get(node.suId)?.push(node);
    });
    
    nodesBySU.forEach((suNodes, suId) => {
      const suWidth = usableWidth / SU_COUNT;
      const suStart = padding.left + suId * suWidth;
      suNodes.forEach((node, i) => {
        // Arrange in a grid within SU space
        const cols = Math.ceil(Math.sqrt(suNodes.length));
        const col = i % cols;
        const row = Math.floor(i / cols);
        const x = suStart + (col + 0.5) * (suWidth / cols);
        const y = nodeY + row * 15;
        nodePositions.set(node.id, { x, y });
      });
    });
    
    return {
      padding,
      spineY,
      leafY,
      nodeY,
      spinePositions,
      leafPositions,
      nodePositions,
      activeSpines,
      activeLeafs,
      activeLinks
    };
  }, [data, activeRail, width, height]);

  useEffect(() => {
    if (!svgRef.current || !gRef.current) return;

    const svg = select(svgRef.current);
    const g = select(gRef.current);

    const zoomBehavior = zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoomBehavior);

    // Optional: Initial zoom centering could be added here
  }, [width, height]); // Re-attach if dimensions change

  const getNodePos = (id: string) => {
    return layout.spinePositions.get(id) || 
           layout.leafPositions.get(id) || 
           layout.nodePositions.get(id) ||
           { x: 0, y: 0 };
  };

  return (
    <svg ref={svgRef} width={width} height={height} className="bg-gray-950 cursor-move">
      {/* Background grid pattern definition */}
      <defs>
        <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
          <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#1f2937" strokeWidth="0.5" />
        </pattern>
      </defs>

      {/* Zoomable Content Group */}
      <g ref={gRef}>
        {/* Infinite Grid Background */}
        {/* We make this large enough to cover zooming out, centered on the view */}
        <rect 
          x={-width * 2} 
          y={-height * 2} 
          width={width * 5} 
          height={height * 5} 
          fill="url(#grid)" 
        />
        
        {/* Tier Labels */}
        <text x={20} y={layout.spineY + 4} className="fill-gray-600 text-[10px] font-mono">
          SPINE
        </text>
        <text x={20} y={layout.leafY + 4} className="fill-gray-600 text-[10px] font-mono">
          LEAF
        </text>
        <text x={20} y={layout.nodeY + 4} className="fill-gray-600 text-[10px] font-mono">
          NODES
        </text>
        
        {/* SU Boundaries */}
        {Array.from({ length: SU_COUNT }).map((_, i) => {
          const suWidth = (width - layout.padding.left - layout.padding.right) / SU_COUNT;
          const x = layout.padding.left + i * suWidth;
          return (
            <g key={i}>
              <rect
                x={x + 5}
                y={layout.leafY - 30}
                width={suWidth - 10}
                height={height - layout.leafY - 20}
                fill="none"
                stroke="#374151"
                strokeWidth="1"
                strokeDasharray="4 4"
                rx="8"
              />
              <text
                x={x + suWidth / 2}
                y={layout.leafY - 40}
                textAnchor="middle"
                className="fill-gray-500 text-[11px] font-mono"
              >
                SU {i}
              </text>
            </g>
          );
        })}
        
        {/* Links - Node to Leaf */}
        <g opacity={0.4}>
          {layout.activeLinks
            .filter(link => link.source.startsWith('DGX'))
            .map((link, i) => {
              const source = getNodePos(link.source);
              const target = getNodePos(link.target);
              const color = RAIL_COLORS[link.railId] || '#666';
              return (
                <line
                  key={`n2l-${i}`}
                  x1={source.x}
                  y1={source.y}
                  x2={target.x}
                  y2={target.y}
                  stroke={color}
                  strokeWidth={activeRail ? 1.5 : 0.3}
                  opacity={activeRail ? 0.6 : 0.2}
                />
              );
            })}
        </g>
        
        {/* Links - Leaf to Spine */}
        <g>
          {layout.activeLinks
            .filter(link => link.source.startsWith('LEAF'))
            .map((link, i) => {
              const source = getNodePos(link.source);
              const target = getNodePos(link.target);
              const color = RAIL_COLORS[link.railId] || '#666';
              return (
                <line
                  key={`l2s-${i}`}
                  x1={source.x}
                  y1={source.y}
                  x2={target.x}
                  y2={target.y}
                  stroke={color}
                  strokeWidth={activeRail ? 3 : 1}
                  opacity={activeRail ? 0.8 : 0.4}
                />
              );
            })}
        </g>
        
        {/* Spine Switches */}
        {layout.activeSpines.map(spine => {
          const pos = layout.spinePositions.get(spine.id);
          if (!pos) return null;
          const color = RAIL_COLORS[spine.railId];
          return (
            <g key={spine.id}>
              <circle
                cx={pos.x}
                cy={pos.y}
                r={activeRail ? 14 : 10}
                fill="#111"
                stroke={color}
                strokeWidth={2}
              />
              <text
                x={pos.x}
                y={pos.y + 3}
                textAnchor="middle"
                className="fill-white text-[8px] font-mono font-bold"
              >
                S
              </text>
              {activeRail && (
                <text
                  x={pos.x}
                  y={pos.y - 20}
                  textAnchor="middle"
                  className="fill-gray-400 text-[8px] font-mono"
                >
                  {spine.id.split('-').slice(1).join('-')}
                </text>
              )}
            </g>
          );
        })}
        
        {/* Leaf Switches */}
        {layout.activeLeafs.map(leaf => {
          const pos = layout.leafPositions.get(leaf.id);
          if (!pos) return null;
          const color = RAIL_COLORS[leaf.railId];
          return (
            <g key={leaf.id}>
              <rect
                x={pos.x - (activeRail ? 10 : 6)}
                y={pos.y - (activeRail ? 10 : 6)}
                width={activeRail ? 20 : 12}
                height={activeRail ? 20 : 12}
                rx={3}
                fill="#111"
                stroke={color}
                strokeWidth={2}
              />
              <text
                x={pos.x}
                y={pos.y + 3}
                textAnchor="middle"
                className="fill-white text-[7px] font-mono font-bold"
              >
                L
              </text>
            </g>
          );
        })}
        
        {/* Node Dots (simplified) */}
        {data.nodes
          .filter(n => n.type === NodeType.DGX_H100)
          .map(node => {
            const pos = layout.nodePositions.get(node.id);
            if (!pos) return null;
            return (
              <circle
                key={node.id}
                cx={pos.x}
                cy={pos.y}
                r={activeRail ? 4 : 3}
                fill="#22c55e"
                opacity={activeRail ? 1 : 0.6}
              >
                <title>{node.id}</title>
              </circle>
            );
          })}
      </g>
      
      {/* STATIC HUD ELEMENTS (Do not zoom) */}
      
      {/* Legend */}
      <g transform={`translate(${width - 180}, 20)`}>
        <rect x={0} y={0} width={160} height={activeRail ? 100 : 70} rx={6} fill="#111" stroke="#333" />
        <text x={10} y={20} className="fill-gray-400 text-[10px] font-mono">
          {activeRail ? `RAIL ${activeRail} (GPU ${activeRail - 1})` : 'ALL RAILS'}
        </text>
        <g transform="translate(10, 30)">
          <circle cx={6} cy={6} r={5} fill="#111" stroke="#fff" strokeWidth={1.5} />
          <text x={20} y={10} className="fill-gray-500 text-[9px]">Spine (QM9700)</text>
        </g>
        <g transform="translate(10, 48)">
          <rect x={1} y={1} width={10} height={10} rx={2} fill="#111" stroke="#fff" strokeWidth={1.5} />
          <text x={20} y={10} className="fill-gray-500 text-[9px]">Leaf (QM9700)</text>
        </g>
        {activeRail && (
          <g transform="translate(10, 66)">
            <circle cx={6} cy={6} r={4} fill="#22c55e" />
            <text x={20} y={10} className="fill-gray-500 text-[9px]">DGX H100 Node</text>
          </g>
        )}
      </g>
      
      {/* Rail indicator */}
      {activeRail && (
        <g transform={`translate(${width - 180}, ${height - 60})`}>
          <rect x={0} y={0} width={160} height={40} rx={6} fill="#111" stroke={RAIL_COLORS[activeRail]} />
          <text x={80} y={16} textAnchor="middle" className="fill-gray-300 text-[10px] font-mono">
            Independent Fat-Tree
          </text>
          <text x={80} y={30} textAnchor="middle" style={{ fill: RAIL_COLORS[activeRail] }} className="text-[9px]">
            All GPU {activeRail - 1}s communicate on this rail
          </text>
        </g>
      )}
    </svg>
  );
};

export default LogicalGraph;

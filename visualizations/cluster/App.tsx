import React, { useState, useMemo } from 'react';
import { generateClusterData, getTopologyStats } from './services/topologyGenerator';
import LogicalGraph from './components/LogicalGraph';
import PhysicalRack from './components/PhysicalRack';
import EducationPanel from './components/EducationPanel';
import { RAIL_COLORS, RAIL_NAMES, SU_COUNT } from './constants';
import { LayoutGrid, Network, Cpu, Info, BookOpen, FileText } from 'lucide-react';

/**
 * DGX H100 SuperPOD Topology Visualizer
 * 
 * Educational tool showing the rail-optimized fat-tree network architecture.
 * 
 * KEY CONCEPTS:
 * 
 * 1. INTERNAL vs EXTERNAL connectivity:
 *    - INTERNAL: NVSwitch (Cedar-7) connects 8 GPUs within each node via NVLink
 *    - EXTERNAL: ConnectX-7 NICs connect nodes via InfiniBand (this visualizer)
 * 
 * 2. RAIL-OPTIMIZED topology:
 *    - Each GPU has a dedicated ConnectX-7 NIC
 *    - All GPU 0s connect to Rail 1, all GPU 1s to Rail 2, etc.
 *    - Each rail forms an independent fat-tree
 *    - Optimizes collective operations (all-reduce stays within one rail)
 * 
 * 3. FAT-TREE structure:
 *    - 2 tiers: Leaf switches (per-SU) and Spine switches (global)
 *    - Full bisection bandwidth (non-blocking)
 *    - QM9700 64-port InfiniBand NDR switches
 */

const App: React.FC = () => {
  const [viewMode, setViewMode] = useState<'PHYSICAL' | 'LOGICAL'>('LOGICAL');
  const [activeRail, setActiveRail] = useState<number | null>(null);
  const [showEducation, setShowEducation] = useState(true);
  const [showReport, setShowReport] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(300);
  const [isResizing, setIsResizing] = useState(false);

  const startResizing = React.useCallback(() => {
    setIsResizing(true);
  }, []);

  const stopResizing = React.useCallback(() => {
    setIsResizing(false);
  }, []);

  const resize = React.useCallback(
    (mouseMoveEvent: MouseEvent) => {
      if (isResizing) {
        const newWidth = mouseMoveEvent.clientX;
        if (newWidth > 240 && newWidth < 600) {
          setSidebarWidth(newWidth);
        }
      }
    },
    [isResizing]
  );

  React.useEffect(() => {
    window.addEventListener("mousemove", resize);
    window.addEventListener("mouseup", stopResizing);
    return () => {
      window.removeEventListener("mousemove", resize);
      window.removeEventListener("mouseup", stopResizing);
    };
  }, [resize, stopResizing]);

  const data = useMemo(() => generateClusterData(), []);
  const stats = useMemo(() => getTopologyStats(data), [data]);

  // Group nodes by rack for Physical View
  const racks = useMemo(() => {
    const rackMap = new Map<number, typeof data.nodes>();
    data.nodes.forEach(node => {
      if (!rackMap.has(node.rackId)) {
        rackMap.set(node.rackId, []);
      }
      rackMap.get(node.rackId)?.push(node);
    });
    return Array.from(rackMap.entries()).sort((a, b) => a[0] - b[0]);
  }, [data]);

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-gray-200 font-sans overflow-hidden">

      <EducationPanel isOpen={showReport} onClose={() => setShowReport(false)} />

      {/* Header */}
      <header className="h-14 border-b border-gray-800 bg-gray-900/80 backdrop-blur 
                         flex items-center px-6 justify-between flex-shrink-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-green-500 rounded flex items-center justify-center">
            <Cpu className="text-black w-5 h-5" />
          </div>
          <div>
            <h1 className="font-bold text-base tracking-tight text-white">
              DGX H100 <span className="text-green-500">SuperPOD</span>
            </h1>
            <p className="text-[9px] text-gray-500 font-mono tracking-wide">
              RAIL-OPTIMIZED FAT-TREE TOPOLOGY
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* Report Button */}
          <button
            onClick={() => setShowReport(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs bg-green-600 hover:bg-green-500 text-white transition-colors font-medium shadow-lg shadow-green-900/20"
          >
            <FileText size={12} />
            Full Technical Report
          </button>

          {/* Education Toggle */}
          <button
            onClick={() => setShowEducation(!showEducation)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs transition-colors
              ${showEducation ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400'}`}
          >
            <BookOpen size={12} />
            Learn
          </button>

          {/* View Toggle */}
          <div className="flex bg-gray-800 p-0.5 rounded-lg border border-gray-700">
            <button
              onClick={() => setViewMode('LOGICAL')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs transition-all
                ${viewMode === 'LOGICAL' ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'}`}
            >
              <Network size={12} /> Topology
            </button>
            <button
              onClick={() => setViewMode('PHYSICAL')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs transition-all
                ${viewMode === 'PHYSICAL' ? 'bg-gray-700 text-white' : 'text-gray-500 hover:text-gray-300'}`}
            >
              <LayoutGrid size={12} /> Physical
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow flex relative overflow-hidden">

        {/* Sidebar */}
        <aside
          className="bg-gray-900 border-r border-gray-800 p-4 flex flex-col gap-5 
                     overflow-y-auto flex-shrink-0 z-40 relative group"
          style={{ width: sidebarWidth }}
        >
          {/* Drag Handle */}
          <div
            className="absolute right-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-green-500/50 transition-colors z-50"
            onMouseDown={startResizing}
          />

          {/* Rail Selector */}
          <div>
            <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">
              InfiniBand Rails
            </h3>
            <div className="space-y-0.5">
              <button
                onClick={() => setActiveRail(null)}
                className={`w-full text-left px-2.5 py-2 rounded text-sm font-mono transition-colors
                  ${activeRail === null
                    ? 'bg-gray-800 text-white border border-gray-600'
                    : 'text-gray-500 hover:bg-gray-800/50 border border-transparent'}`}
              >
                ALL RAILS
              </button>
              {[1, 2, 3, 4, 5, 6, 7, 8].map(rail => (
                <button
                  key={rail}
                  onClick={() => setActiveRail(rail)}
                  className={`w-full flex items-center justify-between px-2.5 py-2 rounded 
                              text-sm font-mono transition-all
                    ${activeRail === rail
                      ? 'bg-gray-800 border border-gray-600'
                      : 'border border-transparent hover:bg-gray-800/50'}`}
                >
                  <span className={activeRail === rail ? 'text-white' : 'text-gray-500'}>
                    {RAIL_NAMES[rail]}
                  </span>
                  <div
                    className="w-2 h-2 rounded-full"
                    style={{
                      backgroundColor: RAIL_COLORS[rail],
                      boxShadow: activeRail === rail ? `0 0 8px ${RAIL_COLORS[rail]}` : 'none'
                    }}
                  />
                </button>
              ))}
            </div>
          </div>

          {/* Statistics */}
          <div className="bg-gray-950 rounded p-4 border border-gray-800 text-sm">
            <h4 className="text-gray-500 font-bold mb-3 flex items-center gap-2">
              <Info size={14} /> Cluster Stats
            </h4>
            <div className="space-y-1.5">
              <div className="flex justify-between">
                <span className="text-gray-600">Compute Nodes</span>
                <span className="text-gray-300 font-mono">{stats.computeNodes}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Total GPUs</span>
                <span className="text-gray-300 font-mono">{stats.totalGPUs} × H100</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Leaf Switches</span>
                <span className="text-gray-300 font-mono">{stats.leafSwitches}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Spine Switches</span>
                <span className="text-gray-300 font-mono">{stats.spineSwitches}</span>
              </div>
              <div className="pt-2 mt-2 border-t border-gray-800">
                <span className="text-gray-600 block mb-1">Bisection Bandwidth</span>
                <span className="text-green-500 font-mono text-base font-bold">
                  {stats.totalBisectionBW} Tb/s
                </span>
              </div>
            </div>
          </div>

          {/* Educational Panel */}
          {showEducation && (
            <div className="bg-blue-950/30 rounded p-4 border border-blue-900/50 text-xs">
              <h4 className="text-blue-400 font-bold mb-2">Why Rail-Optimized?</h4>
              <div className="space-y-2 text-gray-400 leading-relaxed">
                <p>
                  <strong className="text-gray-300">All-reduce optimization:</strong> When GPUs
                  perform collective ops, GPU 0 on all nodes communicate via Rail 1 only.
                </p>
                <p>
                  <strong className="text-gray-300">Traffic isolation:</strong> Each rail is an
                  independent fat-tree. Rail 1 congestion doesn't affect Rail 2.
                </p>
                <p>
                  <strong className="text-gray-300">Locality:</strong> Data-parallel training
                  keeps gradient sync within one rail, reducing cross-rail traffic.
                </p>
              </div>
            </div>
          )}

          {/* Architecture Note */}
          {showEducation && (
            <div className="bg-gray-950 rounded p-4 border border-gray-800 text-xs">
              <h4 className="text-yellow-500 font-bold mb-2">⚠️ Important Distinction</h4>
              <p className="text-gray-400 leading-relaxed">
                This visualizer shows the <strong className="text-gray-300">external InfiniBand
                  fabric</strong> (ConnectX-7 NICs).
              </p>
              <p className="text-gray-500 mt-2">
                The internal NVSwitch (Cedar-7) connecting GPUs <em>within</em> each node
                is not shown here.
              </p>
            </div>
          )}
        </aside>

        {/* Visualization */}
        <div className="flex-grow bg-gray-950 relative overflow-hidden">
          {viewMode === 'PHYSICAL' && (
            <div className="absolute inset-0 overflow-x-auto overflow-y-auto p-6">
              <div className="flex gap-6">
                {Array.from({ length: SU_COUNT }).map((_, suIndex) => {
                  const suRacks = racks.filter(([id]) => Math.floor(id / 8) === suIndex);

                  return (
                    <div key={suIndex}
                      className="flex gap-3 p-4 border border-dashed border-gray-800 
                                    rounded-xl bg-gray-900/10">
                      {/* SU Label */}
                      <div className="w-6 flex items-center justify-center">
                        <span className="-rotate-90 whitespace-nowrap text-xl font-bold 
                                         text-gray-800 tracking-widest">
                          SU {suIndex}
                        </span>
                      </div>

                      {/* Racks */}
                      {suRacks.map(([rackId, nodes]) => (
                        <PhysicalRack
                          key={rackId}
                          rackId={rackId}
                          nodes={nodes}
                          activeRail={activeRail}
                        />
                      ))}

                      {/* Network Rack */}
                      <div className="w-48 bg-gray-900/30 border border-gray-800 rounded-lg 
                                      flex flex-col items-center justify-center p-4">
                        <Network className="text-gray-700 mb-2" size={32} />
                        <span className="text-gray-500 font-mono text-[10px]">NETWORK</span>
                        <span className="text-gray-600 text-[9px] mt-1">8× Leaf (QM9700)</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {viewMode === 'LOGICAL' && (
            <LogicalGraph
              data={data}
              activeRail={activeRail}
              width={Math.max(800, window.innerWidth - sidebarWidth)}
              height={Math.max(500, window.innerHeight - 56)}
            />
          )}

          {/* Active Rail Info */}
          {activeRail && viewMode === 'LOGICAL' && (
            <div className="absolute top-4 left-4 bg-black/80 backdrop-blur border 
                            border-gray-700 p-4 rounded-lg max-w-xs">
              <h4 className="font-bold mb-1" style={{ color: RAIL_COLORS[activeRail] }}>
                {RAIL_NAMES[activeRail]}
              </h4>
              <p className="text-[11px] text-gray-400 leading-relaxed">
                Showing the independent fat-tree for GPU {activeRail - 1}.
                All {stats.computeNodes} nodes connect to this rail's 4 leaf switches,
                which uplink to 2 spine switches.
              </p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default App;

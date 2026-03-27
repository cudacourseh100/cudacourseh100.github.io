import React, { useState } from 'react';
import { X, BookOpen, Server, Network, Layers, Wind, Activity, Cpu } from 'lucide-react';

interface Props {
    isOpen: boolean;
    onClose: () => void;
}

const EducationPanel: React.FC<Props> = ({ isOpen, onClose }) => {
    const [activeSection, setActiveSection] = useState<string>('intro');

    if (!isOpen) return null;

    const sections = [
        { id: 'intro', label: 'Introduction', icon: BookOpen },
        { id: 'dgx-arch', label: 'DGX H100 Architecture', icon: Server },
        { id: 'compute-fabric', label: 'Compute Fabric (Rails)', icon: Network },
        { id: 'storage-mgmt', label: 'Storage & Management', icon: Layers },
        { id: 'physical', label: 'Physical Infrastructure', icon: Wind },
        { id: 'dynamics', label: 'Operational Dynamics', icon: Activity },
    ];

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 md:p-8">
            <div className="bg-gray-900 w-full max-w-6xl h-[90vh] rounded-2xl border border-gray-700 shadow-2xl flex overflow-hidden">

                {/* Sidebar Navigation */}
                <div className="w-64 bg-gray-950 border-r border-gray-800 flex flex-col flex-shrink-0">
                    <div className="p-6 border-b border-gray-800">
                        <h2 className="text-xl font-bold text-white flex items-center gap-2">
                            <Cpu className="text-green-500" />
                            <span>Architecture</span>
                        </h2>
                        <p className="text-xs text-gray-500 mt-1">Deep Dive Report</p>
                    </div>

                    <nav className="flex-1 overflow-y-auto p-4 space-y-1">
                        {sections.map((section) => {
                            const Icon = section.icon;
                            return (
                                <button
                                    key={section.id}
                                    onClick={() => setActiveSection(section.id)}
                                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all
                    ${activeSection === section.id
                                            ? 'bg-green-500/10 text-green-400 border border-green-500/20'
                                            : 'text-gray-400 hover:bg-gray-900 hover:text-gray-200'}`}
                                >
                                    <Icon size={18} />
                                    {section.label}
                                </button>
                            );
                        })}
                    </nav>

                    <div className="p-4 border-t border-gray-800">
                        <button
                            onClick={onClose}
                            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition-colors text-sm"
                        >
                            <X size={16} /> Close Report
                        </button>
                    </div>
                </div>

                {/* Content Area */}
                <div className="flex-1 overflow-y-auto bg-gray-900 p-8 md:p-12">
                    <div className="max-w-3xl mx-auto prose prose-invert prose-green">

                        {activeSection === 'intro' && (
                            <div className="space-y-6 animate-fadeIn">
                                <h1 className="text-3xl font-bold text-white mb-6">The Anatomy of Exascale</h1>
                                <p className="text-lg text-gray-300 leading-relaxed">
                                    The NVIDIA DGX H100 SuperPOD is not merely a collection of servers connected by a generic Ethernet backbone;
                                    it is a meticulously architected <strong>"AI Factory"</strong> where the physical arrangement of cables,
                                    the specific selection of optical transceivers, and the rigid hierarchy of switching planes are as critical
                                    to performance as the silicon of the GPUs themselves.
                                </p>
                                <div className="bg-blue-900/20 border-l-4 border-blue-500 p-4 rounded-r">
                                    <h3 className="text-blue-400 font-bold text-sm uppercase mb-2">Why Visualization Matters</h3>
                                    <p className="text-gray-300 text-sm">
                                        For architects and engineers, a superficial understanding of "racks and cables" is insufficient.
                                        One must grasp the topological imperatives that drive the design—specifically the <strong>"Rail-Optimized"</strong>
                                        architecture that aligns physical network lanes with the logical communication patterns of the NVIDIA Collective Communication Library (NCCL).
                                    </p>
                                </div>
                            </div>
                        )}

                        {activeSection === 'dgx-arch' && (
                            <div className="space-y-8 animate-fadeIn">
                                <div>
                                    <h2 className="text-2xl font-bold text-white mb-4">The Fundamental Building Block: DGX H100</h2>
                                    <p className="text-gray-300">
                                        Unlike commodity servers where NICs can be placed in arbitrary PCIe slots, the DGX H100 is a purpose-built appliance
                                        where internal trace lengths and NUMA affinity dictate a strict external port layout.
                                    </p>
                                </div>

                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <div className="bg-gray-800/50 p-6 rounded-xl border border-gray-700">
                                        <h3 className="text-green-400 font-bold mb-3">Mechanical Specs</h3>
                                        <ul className="space-y-2 text-sm text-gray-300">
                                            <li><strong className="text-white">Height:</strong> 8U (14.0 inches)</li>
                                            <li><strong className="text-white">Weight:</strong> 287.6 lbs (130.45 kg)</li>
                                            <li><strong className="text-white">Power:</strong> 10.2 kW max load</li>
                                            <li><strong className="text-white">Airflow:</strong> Massive front intake plenum</li>
                                        </ul>
                                    </div>
                                    <div className="bg-gray-800/50 p-6 rounded-xl border border-gray-700">
                                        <h3 className="text-green-400 font-bold mb-3">Dual-Tray Design</h3>
                                        <ul className="space-y-2 text-sm text-gray-300">
                                            <li><strong className="text-white">CPU Tray (Top):</strong> Dual Intel Xeon Platinum 8480C, 2TB RAM. Anchors OS/Storage.</li>
                                            <li><strong className="text-white">GPU Tray (Bottom):</strong> 8x H100 GPUs, NVSwitch Fabric (900 GB/s). Heart of the system.</li>
                                        </ul>
                                    </div>
                                </div>

                                <div>
                                    <h3 className="text-xl font-bold text-white mb-3">"Cedar-7" Network Modules</h3>
                                    <p className="text-gray-300 mb-4">
                                        The Compute Fabric utilizes custom "Cedar-7" modules. Each module houses two ConnectX-7 controllers,
                                        driving a single OSFP port each. This results in 4 dual-port OSFP cages on the rear panel.
                                    </p>
                                    <div className="overflow-x-auto">
                                        <table className="w-full text-sm text-left text-gray-400 border border-gray-700 rounded-lg overflow-hidden">
                                            <thead className="bg-gray-800 text-gray-200 uppercase text-xs">
                                                <tr>
                                                    <th className="px-4 py-3">Physical Port</th>
                                                    <th className="px-4 py-3">GPU ID</th>
                                                    <th className="px-4 py-3">Rail Assignment</th>
                                                </tr>
                                            </thead>
                                            <tbody className="divide-y divide-gray-700">
                                                <tr className="bg-gray-900/50"><td className="px-4 py-2">OSFP 1 / P1</td><td className="px-4 py-2">GPU 7</td><td className="px-4 py-2 text-green-400">Rail 8</td></tr>
                                                <tr className="bg-gray-900/50"><td className="px-4 py-2">OSFP 1 / P2</td><td className="px-4 py-2">GPU 4</td><td className="px-4 py-2 text-green-400">Rail 5</td></tr>
                                                <tr className="bg-gray-900/50"><td className="px-4 py-2">OSFP 2 / P1</td><td className="px-4 py-2">GPU 6</td><td className="px-4 py-2 text-green-400">Rail 7</td></tr>
                                                <tr className="bg-gray-900/50"><td className="px-4 py-2">OSFP 2 / P2</td><td className="px-4 py-2">GPU 5</td><td className="px-4 py-2 text-green-400">Rail 6</td></tr>
                                                <tr className="bg-gray-900/50"><td className="px-4 py-2">OSFP 3 / P1</td><td className="px-4 py-2">GPU 2</td><td className="px-4 py-2 text-green-400">Rail 3</td></tr>
                                                <tr className="bg-gray-900/50"><td className="px-4 py-2">OSFP 3 / P2</td><td className="px-4 py-2">GPU 1</td><td className="px-4 py-2 text-green-400">Rail 2</td></tr>
                                                <tr className="bg-gray-900/50"><td className="px-4 py-2">OSFP 4 / P1</td><td className="px-4 py-2">GPU 3</td><td className="px-4 py-2 text-green-400">Rail 4</td></tr>
                                                <tr className="bg-gray-900/50"><td className="px-4 py-2">OSFP 4 / P2</td><td className="px-4 py-2">GPU 0</td><td className="px-4 py-2 text-green-400">Rail 1</td></tr>
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        )}

                        {activeSection === 'compute-fabric' && (
                            <div className="space-y-8 animate-fadeIn">
                                <div>
                                    <h2 className="text-2xl font-bold text-white mb-4">Deconstructing the Rail-Optimized Topology</h2>
                                    <p className="text-gray-300 mb-4">
                                        The SuperPOD uses a "Striped Fat-Tree" topology. Deep learning workloads rely on collective operations like All-Reduce,
                                        where every GPU of a certain rank (e.g., all GPU 0s) must exchange data.
                                    </p>
                                    <div className="bg-yellow-900/20 border-l-4 border-yellow-500 p-4 rounded-r">
                                        <h3 className="text-yellow-500 font-bold text-sm uppercase mb-2">The Concept of Rails</h3>
                                        <p className="text-gray-300 text-sm">
                                            The network is sliced into 8 parallel, isolated networks. <strong>Rail 1</strong> connects exclusively to GPU 0 on every node.
                                            This physical isolation eliminates head-of-line blocking between different GPUs and maximizes Ring-based algorithm efficiency.
                                        </p>
                                    </div>
                                </div>

                                <div>
                                    <h3 className="text-xl font-bold text-white mb-3">Switching Hardware: QM9700</h3>
                                    <ul className="list-disc list-inside space-y-2 text-gray-300">
                                        <li><strong>Throughput:</strong> 64 ports of 400 Gb/s (NDR).</li>
                                        <li><strong>Twin-Port Logic:</strong> 32 OSFP cages, each supporting 2 independent 400G links via "Twin-Port" transceivers.</li>
                                        <li><strong>Fins-Up/Fins-Down:</strong> To manage thermal density, cages are inverted relative to each other.</li>
                                    </ul>
                                </div>

                                <div>
                                    <h3 className="text-xl font-bold text-white mb-3">Hierarchy: The Scalable Unit (SU)</h3>
                                    <p className="text-gray-300 mb-4">
                                        The standard H100 SU consists of 32 DGX nodes.
                                    </p>
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                        <div className="bg-gray-800 p-4 rounded-lg">
                                            <h4 className="text-white font-bold mb-2">Leaf Layer (Per SU)</h4>
                                            <p className="text-sm text-gray-400">
                                                8 Switches (1 per Rail). Each switch has 32 downlinks to nodes and 32 uplinks to Spines.
                                            </p>
                                        </div>
                                        <div className="bg-gray-800 p-4 rounded-lg">
                                            <h4 className="text-white font-bold mb-2">Spine Layer (Global)</h4>
                                            <p className="text-sm text-gray-400">
                                                Connects multiple SUs. Rail integrity is maintained: Rail 1 Leafs only connect to Rail 1 Spines.
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {activeSection === 'storage-mgmt' && (
                            <div className="space-y-8 animate-fadeIn">
                                <div>
                                    <h2 className="text-2xl font-bold text-white mb-4">Storage & Management Fabrics</h2>
                                    <p className="text-gray-300">
                                        While the Compute Fabric is the "race car engine," these fabrics are the "pit crew." They are physically separate
                                        to prevent I/O from interfering with latency-sensitive GPU-GPU traffic.
                                    </p>
                                </div>

                                <div className="space-y-6">
                                    <div className="bg-purple-900/10 border border-purple-500/30 p-6 rounded-xl">
                                        <h3 className="text-purple-400 font-bold mb-2">Storage Fabric</h3>
                                        <p className="text-gray-300 text-sm mb-3">
                                            Dedicated network for massive read throughput (TB/s to PB/s).
                                        </p>
                                        <ul className="list-disc list-inside text-sm text-gray-400">
                                            <li><strong>Hardware:</strong> ConnectX-7 VPI cards (PCIe slots).</li>
                                            <li><strong>Bandwidth:</strong> &gt;400 Gb/s per node.</li>
                                            <li><strong>Topology:</strong> Fat-Tree or Dragonfly+ (vendor dependent).</li>
                                        </ul>
                                    </div>

                                    <div className="bg-blue-900/10 border border-blue-500/30 p-6 rounded-xl">
                                        <h3 className="text-blue-400 font-bold mb-2">Management Networks</h3>
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-3">
                                            <div>
                                                <h4 className="text-white text-sm font-bold">In-Band</h4>
                                                <p className="text-xs text-gray-400 mt-1">
                                                    High-speed control plane (Slurm, K8s, NFS). Uses 100/200GbE via ConnectX-7.
                                                </p>
                                            </div>
                                            <div>
                                                <h4 className="text-white text-sm font-bold">Out-of-Band (OOB)</h4>
                                                <p className="text-xs text-gray-400 mt-1">
                                                    "Lights-out" management (BMC, BIOS, Power). Uses 1GbE RJ45 Copper.
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
                                    <h3 className="text-white font-bold mb-2">The 127-Node Nuance</h3>
                                    <p className="text-sm text-gray-300">
                                        Why 127 nodes and not 128? One DGX position (typically slot 1 of SU 1) is displaced to host
                                        <strong> UFM (Unified Fabric Manager)</strong> appliances. These manage the InfiniBand subnet and congestion control.
                                    </p>
                                </div>
                            </div>
                        )}

                        {activeSection === 'physical' && (
                            <div className="space-y-8 animate-fadeIn">
                                <div>
                                    <h2 className="text-2xl font-bold text-white mb-4">Physical Infrastructure</h2>
                                    <p className="text-gray-300">
                                        To visualize the cluster "in the room," one must model the physical constraints of power, heat, and cable length.
                                    </p>
                                </div>

                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                    <div className="bg-gray-800 p-5 rounded-lg text-center">
                                        <Wind className="mx-auto text-gray-500 mb-3" size={32} />
                                        <h4 className="text-white font-bold">Heat Density</h4>
                                        <p className="text-sm text-gray-400 mt-2">
                                            ~40-45 kW per rack. 4-5x traditional density.
                                        </p>
                                    </div>
                                    <div className="bg-gray-800 p-5 rounded-lg text-center">
                                        <Layers className="mx-auto text-gray-500 mb-3" size={32} />
                                        <h4 className="text-white font-bold">Rack Layout</h4>
                                        <p className="text-sm text-gray-400 mt-2">
                                            4x DGX H100 per rack. Middle gap for airflow. No switches in compute racks.
                                        </p>
                                    </div>
                                    <div className="bg-gray-800 p-5 rounded-lg text-center">
                                        <Network className="mx-auto text-gray-500 mb-3" size={32} />
                                        <h4 className="text-white font-bold">Network Racks</h4>
                                        <p className="text-sm text-gray-400 mt-2">
                                            Switches consolidated in dedicated racks (End-of-Row).
                                        </p>
                                    </div>
                                </div>

                                <div>
                                    <h3 className="text-xl font-bold text-white mb-3">Cooling & Cabling</h3>
                                    <ul className="space-y-3 text-gray-300 text-sm">
                                        <li className="flex gap-3">
                                            <span className="text-green-500 font-bold">•</span>
                                            <span><strong>Fiber Rivers:</strong> Massive bundles of fiber run from Compute Racks to Network Racks, color-coded by Rail.</span>
                                        </li>
                                        <li className="flex gap-3">
                                            <span className="text-green-500 font-bold">•</span>
                                            <span><strong>LinkX Cables:</strong> Connectors act as heatsinks, dissipating heat from the switch silicon into the airflow.</span>
                                        </li>
                                        <li className="flex gap-3">
                                            <span className="text-green-500 font-bold">•</span>
                                            <span><strong>Hot Aisle:</strong> Rear of the rack is an extremely hot zone. Cables route in sidecars to avoid blocking PSU exhaust.</span>
                                        </li>
                                    </ul>
                                </div>
                            </div>
                        )}

                        {activeSection === 'dynamics' && (
                            <div className="space-y-8 animate-fadeIn">
                                <div>
                                    <h2 className="text-2xl font-bold text-white mb-4">Operational Dynamics</h2>
                                    <p className="text-gray-300">
                                        A static topology is only half the picture. A true "digital twin" must simulate the dynamic flow of data during AI workloads.
                                    </p>
                                </div>

                                <div className="space-y-6">
                                    <div className="border border-green-500/30 bg-green-900/10 p-6 rounded-xl">
                                        <h3 className="text-green-400 font-bold mb-2">NCCL Ring Algorithms</h3>
                                        <p className="text-sm text-gray-300 mb-4">
                                            <strong>Scenario: All-Reduce.</strong> All 127 nodes simultaneously transmit data from GPU 0.
                                        </p>
                                        <ol className="list-decimal list-inside text-sm text-gray-400 space-y-1">
                                            <li>Data exits Port 4P2 (Rail 1) on every node.</li>
                                            <li>Enters Rail 1 Leaf Switches.</li>
                                            <li>Traverses Rail 1 Spine (if crossing SUs).</li>
                                            <li>Reaches next GPU 0 in the logical ring.</li>
                                        </ol>
                                        <p className="text-xs text-gray-500 mt-3 italic">
                                            *Simultaneously, independent rings run on Rails 2-8.
                                        </p>
                                    </div>

                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                        <div className="bg-gray-800 p-5 rounded-lg">
                                            <h4 className="text-white font-bold mb-2">In-Network Computing (SHaRP)</h4>
                                            <p className="text-sm text-gray-400">
                                                Switches perform mathematical summation of gradients. Spine switches act as active processing nodes,
                                                reducing data volume and latency.
                                            </p>
                                        </div>
                                        <div className="bg-gray-800 p-5 rounded-lg">
                                            <h4 className="text-white font-bold mb-2">Adaptive Routing</h4>
                                            <p className="text-sm text-gray-400">
                                                InfiniBand splits flows across multiple paths packet-by-packet. If a link saturates, packets
                                                immediately divert to parallel Spine links.
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                    </div>
                </div>
            </div>
        </div>
    );
};

export default EducationPanel;

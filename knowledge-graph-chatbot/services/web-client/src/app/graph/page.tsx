'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
    ArrowLeft,
    Search,
    RefreshCw,
    ZoomIn,
    ZoomOut,
    Maximize2,
    Info,
    GitBranch,
} from 'lucide-react';
import Link from 'next/link';
import { getGraphStats, searchGraph, querySubgraph } from '@/lib/api';
import type { GraphNode, GraphEdge, SubgraphData, GraphStats } from '@/lib/types';

// ============================================================================
// Color mapping for entity types
// ============================================================================

const LABEL_COLORS: Record<string, string> = {
    CVE: '#e63946',
    THREAT_ACTOR: '#f77f00',
    SOFTWARE: '#118ab2',
    ORGANIZATION: '#06d6a0',
    MITIGATION: '#8ac926',
    ATTACK_PATTERN: '#fca311',
    VULNERABILITY: '#ff6b6b',
    MALWARE: '#c77dff',
    PERSON: '#82b1ff',
    LOCATION: '#69db7c',
    TOOL: '#ffd43b',
    CAMPAIGN: '#ff8787',
};

// ============================================================================
// Graph Explorer page
// ============================================================================

export default function GraphPage() {
    const [stats, setStats] = useState<GraphStats | null>(null);
    const [subgraph, setSubgraph] = useState<SubgraphData>({ nodes: [], edges: [] });
    const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
    const [searchQuery, setSearchQuery] = useState('');
    const [isLoading, setIsLoading] = useState(false);

    // Load stats on mount
    useEffect(() => {
        getGraphStats().then(setStats).catch(console.error);

        // Load initial subgraph (demo data)
        querySubgraph({
            seed_node_ids: ['n1'],
            max_hops: 2,
            max_nodes: 50,
        })
            .then(setSubgraph)
            .catch(console.error);
    }, []);

    const handleSearch = useCallback(async () => {
        if (!searchQuery.trim()) return;
        setIsLoading(true);
        try {
            const results = await searchGraph({ query: searchQuery, k: 10, use_hybrid: true });
            // If results contain node IDs, expand their subgraph
            if (results.results?.length > 0) {
                const nodeIds = results.results.map((r: any) => r.node_id);
                const sg = await querySubgraph({
                    seed_node_ids: nodeIds,
                    max_hops: 2,
                    max_nodes: 50,
                });
                setSubgraph(sg);
            }
        } catch (err) {
            console.error('Search failed:', err);
        } finally {
            setIsLoading(false);
        }
    }, [searchQuery]);

    return (
        <div className="h-screen flex flex-col bg-bg-primary">
            {/* Header */}
            <header className="flex items-center justify-between px-4 py-3 border-b border-white/5 bg-bg-secondary/50 backdrop-blur-md">
                <div className="flex items-center gap-3">
                    <Link href="/" className="p-1.5 rounded-lg hover:bg-white/5 transition text-text-secondary">
                        <ArrowLeft className="w-4 h-4" />
                    </Link>
                    <div className="flex items-center gap-2">
                        <GitBranch className="w-5 h-5 text-accent-purple" />
                        <h1 className="font-semibold text-sm">Knowledge Graph Explorer</h1>
                    </div>
                </div>

                {/* Search */}
                <div className="flex items-center gap-2">
                    <div className="flex items-center gap-2 bg-bg-card rounded-lg px-3 py-1.5 border border-white/5">
                        <Search className="w-3.5 h-3.5 text-text-muted" />
                        <input
                            type="text"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                            placeholder="Search entities..."
                            className="bg-transparent text-xs text-text-primary placeholder:text-text-muted outline-none w-48"
                        />
                    </div>
                    <button
                        onClick={handleSearch}
                        disabled={isLoading}
                        className="p-1.5 rounded-lg bg-accent-purple/20 text-accent-purple hover:bg-accent-purple/30 transition disabled:opacity-50"
                    >
                        {isLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Search className="w-3.5 h-3.5" />}
                    </button>
                </div>
            </header>

            {/* Main content */}
            <div className="flex-1 flex overflow-hidden">
                {/* Graph canvas area */}
                <div className="flex-1 relative bg-bg-primary">
                    {/* Canvas — using CSS-based visualization */}
                    <GraphCanvas
                        nodes={subgraph.nodes}
                        edges={subgraph.edges}
                        onNodeClick={setSelectedNode}
                        selectedNodeId={selectedNode?.id}
                    />

                    {/* Legend */}
                    <div className="absolute bottom-4 left-4 glass-card p-3">
                        <h4 className="text-[10px] font-semibold text-text-secondary mb-2 uppercase tracking-wider">Legend</h4>
                        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                            {Object.entries(LABEL_COLORS).slice(0, 8).map(([label, color]) => (
                                <div key={label} className="flex items-center gap-1.5">
                                    <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                                    <span className="text-[10px] text-text-secondary">{label}</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Controls */}
                    <div className="absolute top-4 right-4 flex flex-col gap-1">
                        {[ZoomIn, ZoomOut, Maximize2].map((Icon, i) => (
                            <button
                                key={i}
                                className="p-2 glass-card hover:bg-bg-hover transition text-text-secondary"
                            >
                                <Icon className="w-4 h-4" />
                            </button>
                        ))}
                    </div>
                </div>

                {/* Side panel */}
                <aside className="w-80 border-l border-white/5 bg-bg-secondary/30 overflow-y-auto">
                    {selectedNode ? (
                        <NodeDetailPanel node={selectedNode} onClose={() => setSelectedNode(null)} />
                    ) : (
                        <StatsPanel stats={stats} nodeCount={subgraph.nodes.length} edgeCount={subgraph.edges.length} />
                    )}
                </aside>
            </div>
        </div>
    );
}

// ============================================================================
// Graph canvas — CSS-based circle layout visualization
// ============================================================================

function GraphCanvas({
    nodes,
    edges,
    onNodeClick,
    selectedNodeId,
}: {
    nodes: GraphNode[];
    edges: GraphEdge[];
    onNodeClick: (node: GraphNode) => void;
    selectedNodeId?: string;
}) {
    // Position nodes in a force-directed-like circle layout
    const nodePositions = useMemo(() => {
        const positions: Record<string, { x: number; y: number }> = {};
        const centerX = 50;
        const centerY = 50;
        const radius = 30;

        nodes.forEach((node, i) => {
            const angle = (2 * Math.PI * i) / Math.max(nodes.length, 1);
            positions[node.id] = {
                x: centerX + radius * Math.cos(angle) + (Math.random() * 8 - 4),
                y: centerY + radius * Math.sin(angle) + (Math.random() * 8 - 4),
            };
        });

        return positions;
    }, [nodes]);

    if (nodes.length === 0) {
        return (
            <div className="w-full h-full flex items-center justify-center text-text-muted">
                <div className="text-center">
                    <GitBranch className="w-12 h-12 mx-auto mb-3 opacity-20" />
                    <p className="text-sm">Search for entities to visualize the knowledge graph</p>
                </div>
            </div>
        );
    }

    return (
        <div className="w-full h-full relative overflow-hidden">
            {/* SVG edges */}
            <svg className="absolute inset-0 w-full h-full pointer-events-none">
                {edges.map((edge) => {
                    const src = nodePositions[edge.source];
                    const tgt = nodePositions[edge.target];
                    if (!src || !tgt) return null;

                    return (
                        <g key={edge.id}>
                            <line
                                x1={`${src.x}%`}
                                y1={`${src.y}%`}
                                x2={`${tgt.x}%`}
                                y2={`${tgt.y}%`}
                                stroke="rgba(255,255,255,0.08)"
                                strokeWidth={1.5}
                            />
                            <text
                                x={`${(src.x + tgt.x) / 2}%`}
                                y={`${(src.y + tgt.y) / 2}%`}
                                fill="rgba(255,255,255,0.25)"
                                fontSize={8}
                                textAnchor="middle"
                                dominantBaseline="central"
                            >
                                {edge.relation}
                            </text>
                        </g>
                    );
                })}
            </svg>

            {/* Node elements */}
            {nodes.map((node) => {
                const pos = nodePositions[node.id];
                if (!pos) return null;
                const color = LABEL_COLORS[node.label] || '#6b7280';
                const isSelected = selectedNodeId === node.id;

                return (
                    <motion.button
                        key={node.id}
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        className="absolute flex flex-col items-center cursor-pointer group"
                        style={{
                            left: `${pos.x}%`,
                            top: `${pos.y}%`,
                            transform: 'translate(-50%, -50%)',
                        }}
                        onClick={() => onNodeClick(node)}
                    >
                        <div
                            className={`w-8 h-8 rounded-full flex items-center justify-center text-[10px] font-bold text-white transition-all duration-200 ${isSelected ? 'ring-2 ring-white scale-125' : 'group-hover:scale-110'
                                }`}
                            style={{
                                backgroundColor: color,
                                boxShadow: isSelected ? `0 0 20px ${color}50` : `0 0 8px ${color}30`,
                            }}
                        >
                            {node.name.charAt(0)}
                        </div>
                        <span className="text-[9px] text-text-secondary mt-1 max-w-[80px] truncate text-center group-hover:text-text-primary transition">
                            {node.name}
                        </span>
                    </motion.button>
                );
            })}
        </div>
    );
}

// ============================================================================
// Node detail panel — shown when a node is selected
// ============================================================================

function NodeDetailPanel({
    node,
    onClose,
}: {
    node: GraphNode;
    onClose: () => void;
}) {
    const color = LABEL_COLORS[node.label] || '#6b7280';

    return (
        <div className="p-4">
            <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-sm">Node Details</h3>
                <button onClick={onClose} className="text-xs text-text-muted hover:text-text-secondary">
                    ✕
                </button>
            </div>

            <div className="glass-card p-4 mb-4">
                <div className="flex items-center gap-3 mb-3">
                    <div
                        className="w-10 h-10 rounded-lg flex items-center justify-center text-white font-bold"
                        style={{ backgroundColor: color }}
                    >
                        {node.name.charAt(0)}
                    </div>
                    <div>
                        <p className="font-semibold text-sm">{node.name}</p>
                        <p className="text-[10px] uppercase tracking-wider" style={{ color }}>
                            {node.label}
                        </p>
                    </div>
                </div>

                {/* Confidence bar */}
                <div className="mb-3">
                    <div className="flex justify-between text-[10px] text-text-secondary mb-1">
                        <span>Confidence</span>
                        <span>{(node.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div className="w-full h-1.5 bg-bg-primary rounded-full">
                        <div
                            className="h-full rounded-full transition-all"
                            style={{
                                width: `${node.confidence * 100}%`,
                                backgroundColor: color,
                            }}
                        />
                    </div>
                </div>

                {/* Properties */}
                {Object.keys(node.properties).length > 0 && (
                    <div>
                        <h4 className="text-[10px] font-semibold text-text-secondary uppercase tracking-wider mb-2">
                            Properties
                        </h4>
                        <div className="space-y-1">
                            {Object.entries(node.properties).map(([key, value]) => (
                                <div key={key} className="flex justify-between text-xs">
                                    <span className="text-text-muted">{key}</span>
                                    <span className="text-text-primary font-mono text-[11px]">{value}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>

            <p className="text-[10px] text-text-muted font-mono">ID: {node.id}</p>
        </div>
    );
}

// ============================================================================
// Stats panel — shown when no node is selected
// ============================================================================

function StatsPanel({
    stats,
    nodeCount,
    edgeCount,
}: {
    stats: GraphStats | null;
    nodeCount: number;
    edgeCount: number;
}) {
    return (
        <div className="p-4">
            <div className="flex items-center gap-2 mb-4">
                <Info className="w-4 h-4 text-accent-purple" />
                <h3 className="font-semibold text-sm">Graph Statistics</h3>
            </div>

            {/* Current view stats */}
            <div className="grid grid-cols-2 gap-2 mb-4">
                <div className="glass-card p-3 text-center">
                    <p className="text-xl font-bold text-accent-cyan">{nodeCount}</p>
                    <p className="text-[10px] text-text-muted">Visible Nodes</p>
                </div>
                <div className="glass-card p-3 text-center">
                    <p className="text-xl font-bold text-accent-purple">{edgeCount}</p>
                    <p className="text-[10px] text-text-muted">Visible Edges</p>
                </div>
            </div>

            {/* Global stats */}
            {stats && (
                <>
                    <h4 className="text-[10px] font-semibold text-text-secondary uppercase tracking-wider mb-2">
                        Total Graph
                    </h4>
                    <div className="grid grid-cols-2 gap-2 mb-4">
                        <div className="glass-card p-2 text-center">
                            <p className="text-lg font-bold">{stats.total_nodes}</p>
                            <p className="text-[10px] text-text-muted">Total Nodes</p>
                        </div>
                        <div className="glass-card p-2 text-center">
                            <p className="text-lg font-bold">{stats.total_edges}</p>
                            <p className="text-[10px] text-text-muted">Total Edges</p>
                        </div>
                    </div>

                    {/* Nodes by label */}
                    <h4 className="text-[10px] font-semibold text-text-secondary uppercase tracking-wider mb-2">
                        Entities by Type
                    </h4>
                    <div className="space-y-1.5 mb-4">
                        {Object.entries(stats.nodes_by_label).map(([label, count]) => {
                            const color = LABEL_COLORS[label] || '#6b7280';
                            const maxCount = Math.max(...Object.values(stats.nodes_by_label));

                            return (
                                <div key={label} className="flex items-center gap-2">
                                    <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                                    <span className="text-[10px] text-text-secondary flex-1">{label}</span>
                                    <div className="w-16 h-1.5 bg-bg-primary rounded-full">
                                        <div
                                            className="h-full rounded-full"
                                            style={{
                                                width: `${(count / maxCount) * 100}%`,
                                                backgroundColor: color,
                                            }}
                                        />
                                    </div>
                                    <span className="text-[10px] text-text-muted w-6 text-right">{count}</span>
                                </div>
                            );
                        })}
                    </div>
                </>
            )}
        </div>
    );
}

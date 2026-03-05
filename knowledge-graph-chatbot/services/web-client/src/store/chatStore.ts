/**
 * chatStore.ts — Zustand state management for chat and graph data.
 *
 * Centralized store managing:
 * - Chat sessions and messages
 * - Active subgraph data from responses
 * - UI state (loading, view mode, selected node)
 */

import { create } from 'zustand';
import type {
    ChatMessage,
    ChatSession,
    SubgraphData,
    GraphNode,
    ViewMode,
} from '@/lib/types';

interface ChatState {
    // Chat state
    sessions: ChatSession[];
    activeSessionId: string;
    messages: ChatMessage[];
    isStreaming: boolean;

    // Graph state (from chat responses)
    subgraph: SubgraphData;
    selectedNode: GraphNode | null;

    // UI state
    viewMode: ViewMode;
    sidebarOpen: boolean;

    // Chat actions
    addMessage: (msg: ChatMessage) => void;
    updateLastMessage: (content: string) => void;
    setStreaming: (isStreaming: boolean) => void;
    clearMessages: () => void;
    setActiveSession: (sessionId: string) => void;

    // Graph actions
    setSubgraph: (data: SubgraphData) => void;
    mergeSubgraph: (data: SubgraphData) => void;
    selectNode: (node: GraphNode | null) => void;

    // UI actions
    setViewMode: (mode: ViewMode) => void;
    toggleSidebar: () => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
    // Initial state
    sessions: [],
    activeSessionId: 'default',
    messages: [],
    isStreaming: false,
    subgraph: { nodes: [], edges: [] },
    selectedNode: null,
    viewMode: 'split',
    sidebarOpen: true,

    // Chat actions
    addMessage: (msg) =>
        set((state) => ({ messages: [...state.messages, msg] })),

    updateLastMessage: (content) =>
        set((state) => {
            const messages = [...state.messages];
            const last = messages[messages.length - 1];
            if (last && last.role === 'assistant') {
                messages[messages.length - 1] = {
                    ...last,
                    content: last.content + content,
                };
            }
            return { messages };
        }),

    setStreaming: (isStreaming) => set({ isStreaming }),

    clearMessages: () =>
        set({ messages: [], subgraph: { nodes: [], edges: [] } }),

    setActiveSession: (sessionId) => set({ activeSessionId: sessionId }),

    // Graph actions
    setSubgraph: (data) => set({ subgraph: data }),

    mergeSubgraph: (data) =>
        set((state) => {
            // Merge new data with existing, avoiding duplicates
            const existingNodeIds = new Set(state.subgraph.nodes.map((n) => n.id));
            const existingEdgeIds = new Set(state.subgraph.edges.map((e) => e.id));

            const newNodes = data.nodes.filter((n) => !existingNodeIds.has(n.id));
            const newEdges = data.edges.filter((e) => !existingEdgeIds.has(e.id));

            return {
                subgraph: {
                    nodes: [...state.subgraph.nodes, ...newNodes],
                    edges: [...state.subgraph.edges, ...newEdges],
                },
            };
        }),

    selectNode: (node) => set({ selectedNode: node }),

    // UI actions
    setViewMode: (mode) => set({ viewMode: mode }),
    toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
}));

'use client';

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, GitBranch, ArrowLeft, Loader2, Trash2 } from 'lucide-react';
import Link from 'next/link';
import { useChat } from '@/hooks/useChat';
import { useChatStore } from '@/store/chatStore';
import type { ChatMessage } from '@/lib/types';

// ============================================================================
// Chat page — split view with conversation and mini graph
// ============================================================================

export default function ChatPage() {
    const { messages, isStreaming, sendMessage } = useChat();
    const { subgraph, clearMessages } = useChatStore();
    const [input, setInput] = useState('');
    const scrollRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom on new messages
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages]);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (input.trim()) {
            sendMessage(input);
            setInput('');
        }
    };

    return (
        <div className="h-screen flex flex-col bg-bg-primary">
            {/* Header */}
            <header className="flex items-center justify-between px-4 py-3 border-b border-white/5 bg-bg-secondary/50 backdrop-blur-md">
                <div className="flex items-center gap-3">
                    <Link
                        href="/"
                        className="p-1.5 rounded-lg hover:bg-white/5 transition text-text-secondary"
                    >
                        <ArrowLeft className="w-4 h-4" />
                    </Link>
                    <div className="flex items-center gap-2">
                        <GitBranch className="w-5 h-5 text-accent-cyan" />
                        <h1 className="font-semibold text-sm">Knowledge Graph Chat</h1>
                    </div>
                </div>

                <div className="flex items-center gap-2">
                    {subgraph.nodes.length > 0 && (
                        <span className="text-xs text-accent-cyan bg-accent-cyan/10 px-2 py-1 rounded-full">
                            {subgraph.nodes.length} nodes · {subgraph.edges.length} edges
                        </span>
                    )}
                    <button
                        onClick={clearMessages}
                        className="p-1.5 rounded-lg hover:bg-white/5 transition text-text-muted hover:text-text-secondary"
                        title="Clear conversation"
                    >
                        <Trash2 className="w-4 h-4" />
                    </button>
                </div>
            </header>

            {/* Messages area */}
            <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-6">
                <div className="max-w-3xl mx-auto space-y-4">
                    {messages.length === 0 && (
                        <EmptyState onSuggestion={(s) => { setInput(s); }} />
                    )}

                    <AnimatePresence initial={false}>
                        {messages.map((msg) => (
                            <motion.div
                                key={msg.id}
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ duration: 0.2 }}
                            >
                                <MessageBubble message={msg} />
                            </motion.div>
                        ))}
                    </AnimatePresence>

                    {isStreaming && (
                        <div className="flex items-center gap-2 text-xs text-text-muted pl-10">
                            <Loader2 className="w-3 h-3 animate-spin text-accent-cyan" />
                            Reasoning with knowledge graph...
                        </div>
                    )}
                </div>
            </div>

            {/* Input area */}
            <div className="border-t border-white/5 bg-bg-secondary/30 backdrop-blur-md px-4 py-3">
                <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
                    <div className="flex items-center gap-2 bg-bg-card rounded-xl px-4 py-2 glow-border">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Ask about CVEs, threat actors, attack patterns..."
                            className="flex-1 bg-transparent text-sm text-text-primary placeholder:text-text-muted outline-none"
                            disabled={isStreaming}
                            autoFocus
                        />
                        <button
                            type="submit"
                            disabled={isStreaming || !input.trim()}
                            className="p-2 rounded-lg bg-accent-cyan text-bg-primary hover:bg-accent-cyan/90 transition disabled:opacity-30 disabled:cursor-not-allowed"
                        >
                            <Send className="w-4 h-4" />
                        </button>
                    </div>
                    <p className="text-[10px] text-text-muted text-center mt-2">
                        Answers are grounded in the knowledge graph with [NODE:id] citations
                    </p>
                </form>
            </div>
        </div>
    );
}

// ============================================================================
// Message bubble component
// ============================================================================

function MessageBubble({ message }: { message: ChatMessage }) {
    const isUser = message.role === 'user';

    // Parse [NODE:xxx] citations into styled spans
    const renderContent = (content: string) => {
        const parts = content.split(/(\[NODE:[^\]]+\])/g);
        return parts.map((part, i) => {
            const match = part.match(/\[NODE:([^\]]+)\]/);
            if (match) {
                return (
                    <span key={i} className="citation-marker" title={`Node: ${match[1]}`}>
                        {part}
                    </span>
                );
            }
            return <span key={i}>{part}</span>;
        });
    };

    return (
        <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
            {!isUser && (
                <div className="w-7 h-7 rounded-lg bg-accent-cyan/10 flex items-center justify-center flex-shrink-0 mt-1">
                    <GitBranch className="w-3.5 h-3.5 text-accent-cyan" />
                </div>
            )}

            <div
                className={`max-w-[80%] rounded-xl px-4 py-2.5 text-sm leading-relaxed ${isUser
                        ? 'bg-accent-blue/20 text-text-primary border border-accent-blue/20'
                        : 'bg-bg-card text-text-primary border border-white/5'
                    } ${message.isStreaming ? 'streaming-cursor' : ''}`}
            >
                {renderContent(message.content || '')}
            </div>
        </div>
    );
}

// ============================================================================
// Empty state with suggested questions
// ============================================================================

function EmptyState({ onSuggestion }: { onSuggestion: (s: string) => void }) {
    const suggestions = [
        'What is CVE-2021-44228 and which software does it affect?',
        'Which threat actors exploit Log4Shell?',
        'What mitigations exist for Log4j vulnerabilities?',
        'Show me the relationship between APT41 and CVE-2021-44228',
    ];

    return (
        <div className="flex flex-col items-center justify-center py-16">
            <div className="w-16 h-16 rounded-2xl bg-accent-cyan/10 flex items-center justify-center mb-4">
                <GitBranch className="w-8 h-8 text-accent-cyan" />
            </div>
            <h2 className="text-lg font-semibold mb-1">Knowledge Graph Chat</h2>
            <p className="text-sm text-text-secondary mb-6">
                Ask questions grounded in threat intelligence data
            </p>
            <div className="grid gap-2 w-full max-w-md">
                {suggestions.map((s) => (
                    <button
                        key={s}
                        onClick={() => onSuggestion(s)}
                        className="text-left text-xs text-text-secondary bg-bg-card hover:bg-bg-hover border border-white/5 rounded-lg px-4 py-2.5 transition"
                    >
                        {s}
                    </button>
                ))}
            </div>
        </div>
    );
}

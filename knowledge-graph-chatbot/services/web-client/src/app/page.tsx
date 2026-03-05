'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { MessageCircle, GitBranch, Zap, Shield, Search, Brain } from 'lucide-react';

const features = [
    {
        icon: Brain,
        title: 'Knowledge Graph RAG',
        description: 'Graph-grounded reasoning with entity citations — no hallucinations.',
        color: 'text-accent-cyan',
    },
    {
        icon: Search,
        title: 'Hybrid Search',
        description: 'Vector similarity + graph traversal for deep contextual retrieval.',
        color: 'text-accent-blue',
    },
    {
        icon: Shield,
        title: 'Threat Intelligence',
        description: 'CVEs, APT groups, attack patterns, and mitigations — all linked.',
        color: 'text-accent-pink',
    },
    {
        icon: Zap,
        title: 'Real-Time Streaming',
        description: 'SSE token streaming with live graph visualization updates.',
        color: 'text-accent-amber',
    },
];

export default function HomePage() {
    return (
        <div className="min-h-screen flex flex-col">
            {/* Nav */}
            <nav className="border-b border-white/5 px-6 py-4">
                <div className="max-w-7xl mx-auto flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <GitBranch className="w-6 h-6 text-accent-cyan" />
                        <span className="text-lg font-semibold gradient-text">KG Chatbot</span>
                    </div>
                    <div className="flex items-center gap-4">
                        <Link
                            href="/chat"
                            className="text-sm text-text-secondary hover:text-text-primary transition"
                        >
                            Chat
                        </Link>
                        <Link
                            href="/graph"
                            className="text-sm text-text-secondary hover:text-text-primary transition"
                        >
                            Graph Explorer
                        </Link>
                    </div>
                </div>
            </nav>

            {/* Hero */}
            <main className="flex-1 flex flex-col items-center justify-center px-6">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                    className="text-center max-w-3xl mx-auto"
                >
                    <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-accent-cyan/10 text-accent-cyan text-xs font-medium rounded-full mb-6 border border-accent-cyan/20">
                        <Zap className="w-3 h-3" />
                        Powered by Knowledge Graph + RAG Pipeline
                    </div>

                    <h1 className="text-5xl md:text-6xl font-bold mb-4 leading-tight">
                        <span className="gradient-text">Intelligence</span>
                        <br />
                        <span className="text-text-primary">Grounded in Knowledge</span>
                    </h1>

                    <p className="text-lg text-text-secondary mb-8 leading-relaxed max-w-2xl mx-auto">
                        Ask questions about cybersecurity threats, vulnerabilities, and attack patterns.
                        Every answer is backed by a structured knowledge graph with source citations.
                    </p>

                    <div className="flex flex-col sm:flex-row gap-4 justify-center">
                        <Link
                            href="/chat"
                            className="inline-flex items-center gap-2 px-6 py-3 bg-accent-cyan text-bg-primary font-semibold rounded-lg hover:bg-accent-cyan/90 transition shadow-glow"
                        >
                            <MessageCircle className="w-4 h-4" />
                            Start Chatting
                        </Link>
                        <Link
                            href="/graph"
                            className="inline-flex items-center gap-2 px-6 py-3 border border-white/10 text-text-primary rounded-lg hover:bg-white/5 transition"
                        >
                            <GitBranch className="w-4 h-4" />
                            Explore Graph
                        </Link>
                    </div>
                </motion.div>

                {/* Feature cards */}
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.3 }}
                    className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 max-w-5xl mx-auto mt-16 w-full"
                >
                    {features.map((feature, i) => (
                        <div
                            key={feature.title}
                            className="glass-card p-5 hover:bg-bg-hover transition-colors duration-200"
                        >
                            <feature.icon className={`w-8 h-8 ${feature.color} mb-3`} />
                            <h3 className="font-semibold text-sm mb-1.5">{feature.title}</h3>
                            <p className="text-xs text-text-secondary leading-relaxed">
                                {feature.description}
                            </p>
                        </div>
                    ))}
                </motion.div>
            </main>

            {/* Footer */}
            <footer className="border-t border-white/5 px-6 py-4 text-center text-xs text-text-muted">
                Built with Rust · Go · Python · Next.js — Polyglot microservice architecture
            </footer>
        </div>
    );
}

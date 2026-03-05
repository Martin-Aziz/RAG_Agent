/**
 * useChat.ts — Custom hook for SSE streaming chat.
 *
 * Handles the full chat flow:
 * 1. Add user message to store
 * 2. Create placeholder assistant message
 * 3. Stream response tokens via SSE
 * 4. Parse graph updates and merge into subgraph
 * 5. Handle errors and stream completion
 */

'use client';

import { useCallback } from 'react';
import { useChatStore } from '@/store/chatStore';
import { streamChat } from '@/lib/api';
import type { ChatMessage, SubgraphData } from '@/lib/types';

let messageIdCounter = 0;
function generateId(): string {
    return `msg-${Date.now()}-${++messageIdCounter}`;
}

export function useChat() {
    const {
        messages,
        isStreaming,
        activeSessionId,
        addMessage,
        updateLastMessage,
        setStreaming,
        mergeSubgraph,
    } = useChatStore();

    const sendMessage = useCallback(
        async (content: string) => {
            if (!content.trim() || isStreaming) return;

            // 1. Add user message
            const userMsg: ChatMessage = {
                id: generateId(),
                role: 'user',
                content: content.trim(),
                timestamp: Date.now(),
            };
            addMessage(userMsg);

            // 2. Create placeholder assistant message
            const assistantMsg: ChatMessage = {
                id: generateId(),
                role: 'assistant',
                content: '',
                timestamp: Date.now(),
                isStreaming: true,
            };
            addMessage(assistantMsg);
            setStreaming(true);

            try {
                // 3. Stream response via SSE
                const history = messages.slice(-10).map((m) => ({
                    role: m.role,
                    content: m.content,
                }));

                const stream = streamChat({
                    message: content.trim(),
                    session_id: activeSessionId,
                    history,
                });

                for await (const event of stream) {
                    switch (event.type) {
                        case 'data':
                            // Append text token to the assistant message
                            updateLastMessage(event.data);
                            break;

                        case 'graph':
                            // Parse and merge subgraph update
                            try {
                                const subgraphData: SubgraphData = JSON.parse(event.data);
                                mergeSubgraph(subgraphData);
                            } catch {
                                console.warn('Invalid graph data:', event.data);
                            }
                            break;

                        case 'done':
                            break;

                        case 'error':
                            updateLastMessage(`\n\n⚠️ Error: ${event.data}`);
                            break;
                    }
                }
            } catch (error) {
                const errorMsg =
                    error instanceof Error ? error.message : 'Unknown error';
                updateLastMessage(`\n\n⚠️ Connection error: ${errorMsg}`);
            } finally {
                setStreaming(false);
            }
        },
        [
            messages,
            isStreaming,
            activeSessionId,
            addMessage,
            updateLastMessage,
            setStreaming,
            mergeSubgraph,
        ]
    );

    return {
        messages,
        isStreaming,
        sendMessage,
    };
}

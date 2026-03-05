import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
    title: 'KG Chatbot — Knowledge Graph Intelligence',
    description:
        'Conversational AI powered by a cybersecurity knowledge graph. ' +
        'Ask questions and explore entities, relationships, and threat intelligence.',
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en" className="dark">
            <body className="min-h-screen bg-bg-primary antialiased">
                {children}
            </body>
        </html>
    );
}

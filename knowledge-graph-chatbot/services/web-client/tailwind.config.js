/** @type {import('tailwindcss').Config} */
module.exports = {
    darkMode: 'class',
    content: ['./src/**/*.{ts,tsx}'],
    theme: {
        extend: {
            colors: {
                // Dark cyberpunk palette
                bg: {
                    primary: '#0a0e17',
                    secondary: '#111827',
                    card: '#1a1f2e',
                    hover: '#242b3d',
                },
                accent: {
                    cyan: '#06d6a0',
                    blue: '#118ab2',
                    purple: '#7b2cbf',
                    pink: '#e63946',
                    amber: '#fca311',
                },
                text: {
                    primary: '#e5e7eb',
                    secondary: '#9ca3af',
                    muted: '#6b7280',
                },
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
                mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
            },
            boxShadow: {
                glow: '0 0 20px rgba(6, 214, 160, 0.15)',
                'glow-lg': '0 0 40px rgba(6, 214, 160, 0.2)',
            },
            animation: {
                'pulse-slow': 'pulse 3s ease-in-out infinite',
                'slide-up': 'slideUp 0.3s ease-out',
                'fade-in': 'fadeIn 0.5s ease-out',
            },
            keyframes: {
                slideUp: {
                    '0%': { transform: 'translateY(10px)', opacity: '0' },
                    '100%': { transform: 'translateY(0)', opacity: '1' },
                },
                fadeIn: {
                    '0%': { opacity: '0' },
                    '100%': { opacity: '1' },
                },
            },
        },
    },
    plugins: [],
};

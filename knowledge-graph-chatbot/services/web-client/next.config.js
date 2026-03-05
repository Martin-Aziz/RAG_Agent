/** @type {import('next').NextConfig} */
const nextConfig = {
    output: 'standalone',
    async rewrites() {
        return [
            {
                source: '/api/v1/:path*',
                destination: `${process.env.API_GATEWAY_URL || 'http://api-gateway:8080'}/api/v1/:path*`,
            },
        ];
    },
};

module.exports = nextConfig;

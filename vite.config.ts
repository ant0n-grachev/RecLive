import {defineConfig} from "vitest/config";
import react from "@vitejs/plugin-react";
import {VitePWA} from "vite-plugin-pwa";

// https://vite.dev/config/
export default defineConfig({
    plugins: [
        react(),
        VitePWA({
            strategies: "injectManifest",
            srcDir: "src/pwa",
            filename: "sw.ts",
            registerType: "prompt",
            injectRegister: false,
            manifest: false,
            injectManifest: {
                globPatterns: ["**/*.{js,css,html,svg,png,webp,woff2,webmanifest}"],
            },
        }),
    ],
    test: {
        environment: "jsdom",
        globals: true,
        setupFiles: ["./src/test/setup.ts"],
        clearMocks: true,
        mockReset: true,
        restoreMocks: true,
        include: ["src/**/*.test.{ts,tsx}"],
        coverage: {
            provider: "v8",
            reporter: ["text", "html", "lcov"],
            include: ["src/**/*.{ts,tsx}"],
            exclude: ["src/test/**", "src/main.tsx"],
        },
    },
});

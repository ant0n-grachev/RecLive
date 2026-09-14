import {defineConfig} from "vitest/config";
import react from "@vitejs/plugin-react";
import {VitePWA} from "vite-plugin-pwa";

const noCheckoutDotenvPresent = Object.prototype.hasOwnProperty.call(
    process.env, "RECLIVE_TEST_NO_DOTENV",
);
if (noCheckoutDotenvPresent && process.env.RECLIVE_TEST_NO_DOTENV !== "1") {
    throw new Error("Unsafe test environment configuration: RECLIVE_TEST_NO_DOTENV");
}

// https://vite.dev/config/
export default defineConfig({
    envDir: noCheckoutDotenvPresent ? false : undefined,
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
        // Bound isolated verification resource use; ordinary runs keep Vitest's default.
        maxWorkers: noCheckoutDotenvPresent ? 1 : undefined,
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

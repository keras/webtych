import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: ".",
  testMatch: "*.spec.ts",
  timeout: 30_000,
  retries: 1,
  use: {
    // Serve the trunk-built dist/ directory.
    baseURL: "http://localhost:8383",
  },
  webServer: {
    // Simple static file server for the trunk output.
    command: "npx serve ../../dist -l 8383 --no-clipboard",
    port: 8383,
    reuseExistingServer: !process.env.CI,
  },
  projects: [
    {
      name: "chromium-webgpu",
      use: {
        browserName: "chromium",
        launchOptions: {
          args: [
            // Enable WebGPU in headless Chromium with software rendering.
            "--enable-unsafe-webgpu",
            "--enable-features=Vulkan",
            "--use-angle=swiftshader",
            "--use-webgpu-adapter=swiftshader",
          ],
        },
      },
    },
  ],
});

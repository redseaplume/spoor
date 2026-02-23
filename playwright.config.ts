import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 15_000,
  retries: 0,
  use: {
    baseURL: "http://localhost:3111",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { browserName: "chromium" },
    },
  ],
  webServer: {
    command: "bun run tests/serve.ts",
    port: 3111,
    reuseExistingServer: !process.env["CI"],
  },
});

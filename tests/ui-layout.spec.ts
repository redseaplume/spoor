import { test, expect, type Page } from "@playwright/test";

async function loadWebMode(page: Page) {
  await page.route("**/models/*.onnx", (route) => route.abort());
  await page.goto("/");
  await page.waitForLoadState("domcontentloaded");
}

async function loadTauriMode(page: Page) {
  await page.route("**/models/*.onnx", (route) => route.abort());
  await page.addInitScript(() => {
    (window as any).__TAURI_INTERNALS__ = {
      invoke: async () => {},
      transformCallback: () => 0,
    };
  });
  await page.goto("/");
  await page.waitForLoadState("domcontentloaded");
}

// Helper: measure the height of the options/controls area
async function getControlsHeight(page: Page): Promise<number> {
  const controls = page.locator("[data-testid='controls']");
  const box = await controls.boundingBox();
  return box?.height ?? 0;
}

// ── Controls don't wrap at various viewport widths ───────────
// This is the recurring issue: buttons/labels becoming huge or
// going multiline. We measure the options area height and verify
// it stays consistent (single-line) across viewport sizes.

const viewportWidths = [1200, 1000, 800, 600];

for (const width of viewportWidths) {
  test(`web mode: controls area height stable at ${width}px wide`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 800 });
    await loadWebMode(page);

    const height = await getControlsHeight(page);
    // The options row should not exceed ~60px (single line of buttons
    // + some padding). If it wraps, it'll be much taller.
    expect(height).toBeGreaterThan(0);
    expect(height).toBeLessThan(80);
  });
}

// ── Button sizes stay consistent ─────────────────────────────

test("toggle buttons have consistent font size", async ({ page }) => {
  await page.setViewportSize({ width: 1200, height: 800 });
  await loadWebMode(page);

  const buttons = page.locator(".toggle-btn");
  const count = await buttons.count();

  for (let i = 0; i < count; i++) {
    const fontSize = await buttons.nth(i).evaluate(
      (el) => window.getComputedStyle(el).fontSize
    );
    // All toggle buttons should be the same small size
    // Current CSS sets 0.7rem ≈ 9.8px at 14px base
    const size = parseFloat(fontSize);
    expect(size).toBeLessThan(14); // never full-size text
    expect(size).toBeGreaterThan(7); // never invisibly small
  }
});

test("labels have consistent font size", async ({ page }) => {
  await page.setViewportSize({ width: 1200, height: 800 });
  await loadWebMode(page);

  const labels = page.locator(".label");
  const count = await labels.count();

  for (let i = 0; i < count; i++) {
    const visible = await labels.nth(i).isVisible();
    if (!visible) continue;

    const fontSize = await labels.nth(i).evaluate(
      (el) => window.getComputedStyle(el).fontSize
    );
    const size = parseFloat(fontSize);
    expect(size).toBeLessThan(14);
    expect(size).toBeGreaterThan(7);
  }
});

// ── Tauri mode: controls height with model + resolution ──────

test("tauri mode: controls area height stable at 1200px", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1200, height: 800 });
  await loadTauriMode(page);

  const height = await getControlsHeight(page);
  // With model toggle + resolution toggle, should still be compact
  expect(height).toBeGreaterThan(0);
  expect(height).toBeLessThan(120); // two rows max in current layout
});

// ── Drop zone dimensions ─────────────────────────────────────

test("drop zone has reasonable dimensions", async ({ page }) => {
  await page.setViewportSize({ width: 1200, height: 800 });
  await loadWebMode(page);

  const dropZone = page.locator("[data-testid='drop-zone']");
  const box = await dropZone.boundingBox();

  expect(box).not.toBeNull();
  expect(box!.width).toBeGreaterThan(100);
  expect(box!.height).toBeGreaterThan(40);
});

// ── Header stays single line ─────────────────────────────────

test("header is single line at 800px", async ({ page }) => {
  await page.setViewportSize({ width: 800, height: 800 });
  await loadWebMode(page);

  const header = page.locator("[data-testid='header']");
  const box = await header.boundingBox();
  expect(box).not.toBeNull();
  expect(box!.height).toBeLessThan(70);
});

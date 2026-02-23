import { test, expect, type Page } from "@playwright/test";

// Helper: load page in web mode (default — no __TAURI_INTERNALS__)
// Blocks the ONNX model fetch so the drop zone text doesn't change
async function loadWebMode(page: Page) {
  await page.route("**/models/*.onnx", (route) => route.abort());
  await page.goto("/");
  await page.waitForLoadState("domcontentloaded");
}

// Helper: load page in simulated Tauri mode
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

// ── Web mode: controls visibility ────────────────────────────

test("web mode: model toggle is hidden", async ({ page }) => {
  await loadWebMode(page);
  const modelGroup = page.locator("[data-testid='model-group']");
  await expect(modelGroup).toBeHidden();
});

test("web mode: resolution toggle is visible", async ({ page }) => {
  await loadWebMode(page);
  const resGroup = page.locator("[data-testid='resolution-group']");
  await expect(resGroup).toBeVisible();
});

test("web mode: confidence control is hidden initially", async ({ page }) => {
  await loadWebMode(page);
  const confidenceGroup = page.locator("[data-testid='confidence-group']");
  await expect(confidenceGroup).toBeHidden();
});

test("web mode: species button stays hidden", async ({ page }) => {
  await loadWebMode(page);
  const speciesBtn = page.locator("[data-testid='species-btn']");
  await expect(speciesBtn).toBeHidden();
});

// ── Tauri mode: controls visibility ──────────────────────────

test("tauri mode: model toggle is visible", async ({ page }) => {
  await loadTauriMode(page);
  const modelGroup = page.locator("[data-testid='model-group']");
  await expect(modelGroup).toBeVisible();
});

test("tauri mode: both model buttons exist", async ({ page }) => {
  await loadTauriMode(page);
  const quickBtn = page.locator("[data-testid='model-quick']");
  const thoroughBtn = page.locator("[data-testid='model-thorough']");
  await expect(quickBtn).toBeVisible();
  await expect(thoroughBtn).toBeVisible();
});

test("tauri mode: thorough is default, resolution visible", async ({
  page,
}) => {
  await loadTauriMode(page);
  const thoroughBtn = page.locator("[data-testid='model-thorough']");
  const resGroup = page.locator("[data-testid='resolution-group']");
  await expect(thoroughBtn).toHaveClass(/active/);
  await expect(resGroup).toBeVisible();
});

test("tauri mode: clicking quick hides resolution", async ({ page }) => {
  await loadTauriMode(page);
  const quickBtn = page.locator("[data-testid='model-quick']");
  const resGroup = page.locator("[data-testid='resolution-group']");

  await quickBtn.click();
  await expect(quickBtn).toHaveClass(/active/);
  await expect(resGroup).toBeHidden();
});

test("tauri mode: clicking thorough after quick shows resolution", async ({
  page,
}) => {
  await loadTauriMode(page);
  const quickBtn = page.locator("[data-testid='model-quick']");
  const thoroughBtn = page.locator("[data-testid='model-thorough']");
  const resGroup = page.locator("[data-testid='resolution-group']");

  await quickBtn.click();
  await expect(resGroup).toBeHidden();

  await thoroughBtn.click();
  await expect(resGroup).toBeVisible();
  await expect(thoroughBtn).toHaveClass(/active/);
});

// ── Resolution toggle ────────────────────────────────────────

test("resolution: fast and accurate toggle exclusively", async ({ page }) => {
  await loadWebMode(page);
  const fastBtn = page.locator("[data-testid='res-fast']");
  const accurateBtn = page.locator("[data-testid='res-accurate']");

  // Accurate is default
  await expect(accurateBtn).toHaveClass(/active/);
  await expect(fastBtn).not.toHaveClass(/active/);

  await fastBtn.click();
  await expect(fastBtn).toHaveClass(/active/);
  await expect(accurateBtn).not.toHaveClass(/active/);

  await accurateBtn.click();
  await expect(accurateBtn).toHaveClass(/active/);
  await expect(fastBtn).not.toHaveClass(/active/);
});

// ── Confidence slider ────────────────────────────────────────

test("confidence: slider updates value text", async ({ page }) => {
  await loadWebMode(page);
  // Threshold control is hidden initially but the elements exist in DOM.
  // We test the binding works by evaluating JS directly.
  const newValue = await page.evaluate(() => {
    const slider = document.querySelector(
      "[data-testid='confidence-slider']"
    ) as HTMLInputElement;
    const display = document.querySelector(
      "[data-testid='confidence-value']"
    )!;
    slider.value = "0.5";
    slider.dispatchEvent(new Event("input"));
    return display.textContent;
  });
  expect(newValue).toBe("50%");
});

// ── Drop zone ────────────────────────────────────────────────

test("drop zone: is visible and contains primary text element", async ({
  page,
}) => {
  await loadWebMode(page);
  const dropZone = page.locator("[data-testid='drop-zone']");
  await expect(dropZone).toBeVisible();
  // Text may change to "Downloading model…" or "Failed to load model"
  // in web mode (model fetch runs on load). We verify structure, not text.
  const primary = dropZone.locator(".drop-zone-primary");
  await expect(primary).toBeAttached();
});

// ── Results controls: sort and filter ────────────────────────
// Sort/filter controls are hidden until results appear.
// We test their toggle logic via JS since we can't click hidden elements.

test("sort: buttons toggle exclusively", async ({ page }) => {
  await loadWebMode(page);

  const result = await page.evaluate(() => {
    const orderBtn = document.querySelector(
      '.sort-btn[data-sort="processing"]'
    )!;
    const confBtn = document.querySelector(
      '.sort-btn[data-sort="confidence"]'
    )!;
    const nameBtn = document.querySelector(
      '.sort-btn[data-sort="filename"]'
    )!;

    // Order is default active
    const orderDefault = orderBtn.classList.contains("active");

    // Click confidence
    (confBtn as HTMLElement).click();
    const confActive = confBtn.classList.contains("active");
    const orderAfterConf = orderBtn.classList.contains("active");

    // Click filename
    (nameBtn as HTMLElement).click();
    const nameActive = nameBtn.classList.contains("active");
    const confAfterName = confBtn.classList.contains("active");

    return { orderDefault, confActive, orderAfterConf, nameActive, confAfterName };
  });

  expect(result.orderDefault).toBe(true);
  expect(result.confActive).toBe(true);
  expect(result.orderAfterConf).toBe(false);
  expect(result.nameActive).toBe(true);
  expect(result.confAfterName).toBe(false);
});

test("filter: buttons toggle independently (multi-select)", async ({
  page,
}) => {
  await loadWebMode(page);

  const result = await page.evaluate(() => {
    const animalBtn = document.querySelector(
      '.filter-btn[data-category="animal"]'
    )!;
    const personBtn = document.querySelector(
      '.filter-btn[data-category="person"]'
    )!;

    const bothStartActive =
      animalBtn.classList.contains("active") &&
      personBtn.classList.contains("active");

    // Deactivate animal
    (animalBtn as HTMLElement).click();
    const animalOff = !animalBtn.classList.contains("active");
    const personStillOn = personBtn.classList.contains("active");

    // Reactivate animal
    (animalBtn as HTMLElement).click();
    const animalBackOn = animalBtn.classList.contains("active");

    return { bothStartActive, animalOff, personStillOn, animalBackOn };
  });

  expect(result.bothStartActive).toBe(true);
  expect(result.animalOff).toBe(true);
  expect(result.personStillOn).toBe(true);
  expect(result.animalBackOn).toBe(true);
});

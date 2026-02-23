import { test, expect, type Page } from "@playwright/test";

// ── Mock data ─────────────────────────────────────────────────

const MOCK_ANIMAL = {
  id: "mock-animal",
  fileName: "deer.jpg",
  origWidth: 1920,
  origHeight: 1080,
  inferenceTimeMs: 180,
  detections: [
    {
      bbox: [100, 200, 500, 600],
      bboxNorm: [0.052, 0.185, 0.208, 0.37],
      confidence: 0.92,
      category: 0,
      categoryName: "animal",
    },
  ],
};

const MOCK_PERSON = {
  id: "mock-person",
  fileName: "hiker.jpg",
  origWidth: 1920,
  origHeight: 1080,
  inferenceTimeMs: 195,
  detections: [
    {
      bbox: [300, 100, 700, 900],
      bboxNorm: [0.156, 0.093, 0.208, 0.741],
      confidence: 0.87,
      category: 1,
      categoryName: "person",
    },
  ],
};

const MOCK_EMPTY = {
  id: "mock-empty",
  fileName: "grass.jpg",
  origWidth: 1920,
  origHeight: 1080,
  inferenceTimeMs: 160,
  detections: [],
};

const MOCK_LOW_CONF = {
  id: "mock-lowconf",
  fileName: "maybe-animal.jpg",
  origWidth: 1920,
  origHeight: 1080,
  inferenceTimeMs: 175,
  detections: [
    {
      bbox: [400, 300, 600, 500],
      bboxNorm: [0.208, 0.278, 0.104, 0.185],
      confidence: 0.15,
      category: 0,
      categoryName: "animal",
    },
  ],
};

const MOCK_VEHICLE = {
  id: "mock-vehicle",
  fileName: "truck.jpg",
  origWidth: 1920,
  origHeight: 1080,
  inferenceTimeMs: 190,
  detections: [
    {
      bbox: [200, 300, 800, 700],
      bboxNorm: [0.104, 0.278, 0.313, 0.37],
      confidence: 0.88,
      category: 2,
      categoryName: "vehicle",
    },
  ],
};

const ALL_MOCKS = [MOCK_ANIMAL, MOCK_PERSON, MOCK_EMPTY, MOCK_LOW_CONF, MOCK_VEHICLE];

// ── Helpers ───────────────────────────────────────────────────

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

// Inject mock results as if inference completed.
// Uses page.evaluate with a string so we can access app.js globals
// (const declarations live in the global lexical scope, not on window).
async function injectResults(page: Page, mocks: any[]) {
  const TINY_PNG =
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

  await page.evaluate(
    `
    state.modelReady = true;
    state.totalImages = ${mocks.length};

    // These are normally set by addFiles() which we bypass
    document.getElementById('results').hidden = false;
    document.getElementById('status-bar').classList.add('visible');
    document.getElementById('drop-zone').classList.add('processing');

    const _mocks = ${JSON.stringify(mocks)};
    for (const r of _mocks) {
      objectUrls.set(r.id, "${TINY_PNG}");
      handleResult(r);
    }
  `
  );
}

// ── Post-inference: controls appear ───────────────────────────

test.describe("web mode post-inference", () => {
  test("sort controls become visible", async ({ page }) => {
    await loadWebMode(page);
    await injectResults(page, ALL_MOCKS);
    await expect(page.locator("#sort-controls")).toBeVisible();
  });

  test("filter controls become visible", async ({ page }) => {
    await loadWebMode(page);
    await injectResults(page, ALL_MOCKS);
    await expect(page.locator("#filter-controls")).toBeVisible();
  });

  test("confidence slider becomes visible", async ({ page }) => {
    await loadWebMode(page);
    await injectResults(page, ALL_MOCKS);
    await expect(page.locator("[data-testid='confidence-group']")).toBeVisible();
  });

  test("export JSON button becomes visible", async ({ page }) => {
    await loadWebMode(page);
    await injectResults(page, ALL_MOCKS);
    await expect(page.locator("#export-btn")).toBeVisible();
  });

  test("export CSV button becomes visible", async ({ page }) => {
    await loadWebMode(page);
    await injectResults(page, ALL_MOCKS);
    await expect(page.locator("#export-csv-btn")).toBeVisible();
  });

  test("clear button becomes visible", async ({ page }) => {
    await loadWebMode(page);
    await injectResults(page, ALL_MOCKS);
    await expect(page.locator("#clear-btn")).toBeVisible();
  });

  test("results section is visible with cards", async ({ page }) => {
    await loadWebMode(page);
    await injectResults(page, ALL_MOCKS);
    await expect(page.locator("#results")).toBeVisible();
    const cards = page.locator(".result-card");
    await expect(cards).toHaveCount(ALL_MOCKS.length);
  });

  test("status shows Done", async ({ page }) => {
    await loadWebMode(page);
    await injectResults(page, ALL_MOCKS);
    await expect(page.locator("#status-text")).toHaveText("Done");
  });

  test("species button stays hidden in web mode", async ({ page }) => {
    await loadWebMode(page);
    await injectResults(page, ALL_MOCKS);
    await expect(page.locator("[data-testid='species-btn']")).toBeHidden();
  });
});

// ── Post-inference: Tauri-specific ────────────────────────────

test.describe("tauri mode post-inference", () => {
  test("species button visible when animals detected", async ({ page }) => {
    await loadTauriMode(page);
    await injectResults(page, [MOCK_ANIMAL, MOCK_PERSON]);
    await expect(page.locator("[data-testid='species-btn']")).toBeVisible();
  });

  test("species button hidden when no animals detected", async ({ page }) => {
    await loadTauriMode(page);
    await injectResults(page, [MOCK_PERSON, MOCK_EMPTY]);
    await expect(page.locator("[data-testid='species-btn']")).toBeHidden();
  });

  test("all post-inference controls visible", async ({ page }) => {
    await loadTauriMode(page);
    await injectResults(page, ALL_MOCKS);
    await expect(page.locator("#sort-controls")).toBeVisible();
    await expect(page.locator("#filter-controls")).toBeVisible();
    await expect(page.locator("[data-testid='confidence-group']")).toBeVisible();
    await expect(page.locator("#export-btn")).toBeVisible();
    await expect(page.locator("#export-csv-btn")).toBeVisible();
    await expect(page.locator("#clear-btn")).toBeVisible();
  });
});

// ── Sort interactions ─────────────────────────────────────────

test.describe("sorting", () => {
  test("sort by confidence reorders cards", async ({ page }) => {
    await loadWebMode(page);
    await injectResults(page, ALL_MOCKS);

    // Click confidence sort
    await page.locator('.sort-btn[data-sort="confidence"]').click();

    const filenames = await page.locator(".result-filename").allTextContents();
    // Highest confidence first: animal (0.92), vehicle (0.88), person (0.87), low-conf (0.15), empty (0)
    expect(filenames[0]).toBe("deer.jpg");
    expect(filenames[1]).toBe("truck.jpg");
    expect(filenames[2]).toBe("hiker.jpg");
  });

  test("sort by filename reorders cards alphabetically", async ({ page }) => {
    await loadWebMode(page);
    await injectResults(page, ALL_MOCKS);

    await page.locator('.sort-btn[data-sort="filename"]').click();

    const filenames = await page.locator(".result-filename").allTextContents();
    const sorted = [...filenames].sort((a, b) =>
      a.localeCompare(b, undefined, { numeric: true })
    );
    expect(filenames).toEqual(sorted);
  });

  test("sort by order restores original sequence", async ({ page }) => {
    await loadWebMode(page);
    await injectResults(page, ALL_MOCKS);

    // Shuffle first
    await page.locator('.sort-btn[data-sort="confidence"]').click();
    // Then back to order
    await page.locator('.sort-btn[data-sort="processing"]').click();

    const filenames = await page.locator(".result-filename").allTextContents();
    expect(filenames).toEqual(ALL_MOCKS.map((m) => m.fileName));
  });
});

// ── Filter interactions ───────────────────────────────────────

test.describe("filtering", () => {
  test("hiding animal category hides animal cards", async ({ page }) => {
    await loadWebMode(page);
    await injectResults(page, ALL_MOCKS);

    // Toggle off animal
    await page.locator('.filter-btn[data-category="animal"]').click();

    // The deer card should be hidden. The low-conf animal (0.15) at default
    // threshold (0.2) is classified as "empty" so it stays visible via empty filter.
    const deerCard = page.locator(".result-card", { has: page.locator("text=deer.jpg") });
    await expect(deerCard).toBeHidden();

    // Person card still visible
    const hikerCard = page.locator(".result-card", { has: page.locator("text=hiker.jpg") });
    await expect(hikerCard).toBeVisible();
  });

  test("hiding empty category hides empty cards", async ({ page }) => {
    await loadWebMode(page);
    await injectResults(page, ALL_MOCKS);

    await page.locator('.filter-btn[data-category="empty"]').click();

    // grass.jpg has no detections → empty
    const grassCard = page.locator(".result-card", { has: page.locator("text=grass.jpg") });
    await expect(grassCard).toBeHidden();
  });

  test("re-enabling category shows cards again", async ({ page }) => {
    await loadWebMode(page);
    await injectResults(page, ALL_MOCKS);

    const animalBtn = page.locator('.filter-btn[data-category="animal"]');
    await animalBtn.click(); // hide
    await animalBtn.click(); // show

    const deerCard = page.locator(".result-card", { has: page.locator("text=deer.jpg") });
    await expect(deerCard).toBeVisible();
  });
});

// ── Confidence slider interactions ────────────────────────────

test.describe("confidence slider", () => {
  test("raising threshold hides low-confidence detections", async ({ page }) => {
    await loadWebMode(page);
    await injectResults(page, [MOCK_ANIMAL, MOCK_LOW_CONF]);

    // Low-conf detection (0.15) is below default threshold (0.2),
    // so that card is already "empty". Raise threshold above 0.92 to
    // make the deer card empty too.
    await page.evaluate(() => {
      const slider = document.querySelector("[data-testid='confidence-slider']") as HTMLInputElement;
      slider.value = "0.95";
      slider.dispatchEvent(new Event("input"));
    });

    // Both cards should now show as empty (no visible detections)
    // Check by looking at the card classes
    const cards = page.locator(".result-card");
    const count = await cards.count();
    for (let i = 0; i < count; i++) {
      await expect(cards.nth(i)).toHaveClass(/empty/);
    }
  });

  test("lowering threshold reveals low-confidence detections", async ({ page }) => {
    await loadWebMode(page);
    await injectResults(page, [MOCK_LOW_CONF]);

    // At default 0.2, the 0.15-confidence detection is hidden → card is "empty"
    const card = page.locator(".result-card").first();
    await expect(card).toHaveClass(/empty/);

    // Lower threshold to 0.1
    await page.evaluate(() => {
      const slider = document.querySelector("[data-testid='confidence-slider']") as HTMLInputElement;
      slider.value = "0.1";
      slider.dispatchEvent(new Event("input"));
    });

    // Card should no longer be empty — detection is now visible
    await expect(card).not.toHaveClass(/empty/);
  });

  test("confidence value text updates", async ({ page }) => {
    await loadWebMode(page);
    await injectResults(page, ALL_MOCKS);

    await page.evaluate(() => {
      const slider = document.querySelector("[data-testid='confidence-slider']") as HTMLInputElement;
      slider.value = "0.75";
      slider.dispatchEvent(new Event("input"));
    });

    await expect(page.locator("[data-testid='confidence-value']")).toHaveText("75%");
  });
});

// ── Card interactions ─────────────────────────────────────────

test.describe("card behavior", () => {
  test("card with detections is expandable", async ({ page }) => {
    await loadWebMode(page);
    await injectResults(page, [MOCK_ANIMAL]);

    const card = page.locator(".result-card").first();
    await expect(card).toHaveClass(/expandable/);
  });

  test("empty card is not expandable", async ({ page }) => {
    await loadWebMode(page);
    await injectResults(page, [MOCK_EMPTY]);

    const card = page.locator(".result-card").first();
    await expect(card).not.toHaveClass(/expandable/);
  });

  test("clicking expandable card toggles expanded state", async ({ page }) => {
    await loadWebMode(page);
    await injectResults(page, [MOCK_ANIMAL]);

    const card = page.locator(".result-card").first();
    await expect(card).not.toHaveClass(/expanded/);

    await card.click();
    await expect(card).toHaveClass(/expanded/);

    await card.click();
    await expect(card).not.toHaveClass(/expanded/);
  });

  test("cards have correct category classes", async ({ page }) => {
    await loadWebMode(page);
    await injectResults(page, [MOCK_ANIMAL, MOCK_PERSON, MOCK_VEHICLE, MOCK_EMPTY]);

    const cards = page.locator(".result-card");
    await expect(cards.nth(0)).toHaveClass(/cat-animal/);
    await expect(cards.nth(1)).toHaveClass(/cat-person/);
    await expect(cards.nth(2)).toHaveClass(/cat-vehicle/);
    await expect(cards.nth(3)).toHaveClass(/cat-empty/);
  });
});

// ── Clear resets everything ───────────────────────────────────

test.describe("clear", () => {
  test("clear button resets to initial state", async ({ page }) => {
    await loadWebMode(page);
    await injectResults(page, ALL_MOCKS);

    // Verify post-inference state first
    await expect(page.locator("#sort-controls")).toBeVisible();
    await expect(page.locator("#results")).toBeVisible();

    // Click clear
    await page.locator("#clear-btn").click();

    // Everything should be back to initial state
    await expect(page.locator("#results")).toBeHidden();
    await expect(page.locator("#sort-controls")).toBeHidden();
    await expect(page.locator("#filter-controls")).toBeHidden();
    await expect(page.locator("[data-testid='confidence-group']")).toBeHidden();
    await expect(page.locator("#export-btn")).toBeHidden();
    await expect(page.locator("#export-csv-btn")).toBeHidden();
    await expect(page.locator("#clear-btn")).toBeHidden();
    await expect(page.locator("#status-bar")).not.toHaveClass(/visible/);

    // Drop zone should be back to full size
    await expect(page.locator("[data-testid='drop-zone']")).not.toHaveClass(/processing/);
  });

  test("clear in tauri mode also hides species button", async ({ page }) => {
    await loadTauriMode(page);
    await injectResults(page, [MOCK_ANIMAL]);

    await expect(page.locator("[data-testid='species-btn']")).toBeVisible();

    await page.locator("#clear-btn").click();

    await expect(page.locator("[data-testid='species-btn']")).toBeHidden();
  });
});

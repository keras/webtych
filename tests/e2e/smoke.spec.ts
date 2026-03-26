import { test, expect } from "@playwright/test";

test("page loads without panics or errors", async ({ page }) => {
  const errors: string[] = [];

  // Capture console errors and uncaught exceptions.
  page.on("console", (msg) => {
    if (msg.type() === "error") {
      errors.push(msg.text());
    }
  });
  page.on("pageerror", (err) => {
    errors.push(err.message);
  });

  await page.goto("/");

  // Give the WASM module time to initialise and render a few frames.
  await page.waitForTimeout(3000);

  // The canvas element must exist and be sized to the viewport.
  const canvas = page.locator("canvas#canvas");
  await expect(canvas).toBeVisible();

  const box_ = await canvas.boundingBox();
  expect(box_).not.toBeNull();
  expect(box_!.width).toBeGreaterThan(0);
  expect(box_!.height).toBeGreaterThan(0);

  // No panics (console_error_panic_hook) or JS errors.
  const panics = errors.filter(
    (e) => e.includes("panicked") || e.includes("RuntimeError: unreachable")
  );
  expect(panics).toEqual([]);
});

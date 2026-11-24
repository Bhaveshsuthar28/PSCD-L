import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ABODA_VIDEO10 = path.resolve(__dirname, "../../Dataset/ABODA/video10.avi");
const SCREENSHOT_DIR =
  process.env.TRINETRA_E2E_SCREENSHOT_DIR ||
  path.resolve(__dirname, "../test-results/screenshots");

test.describe.serial("TRINETRA Full-Stack E2E Workflows", () => {
  test("complete upload, GPU processing, SSE, events, evidence, analytics, exports, settings, themes & responsive QA", async ({
    page,
  }) => {
    fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });

    const pageErrors: string[] = [];
    const consoleErrors: string[] = [];
    page.on("pageerror", (err) => pageErrors.push(err.message));
    page.on("console", (msg) => {
      if (msg.type() === "error") {
        const txt = msg.text();
        // Ignore browser-expected 409 error log when deliberately testing duplicate upload
        if (!txt.includes("status of 409")) {
          consoleErrors.push(txt);
        }
      }
    });

    // 1. Dashboard & Backend Connectivity
    await page.goto("/dashboard");
    await expect(page.getByText("Backend Ready")).toBeVisible();
    await expect(page.getByText("Overview Dashboard")).toBeVisible();

    // 2. Theme Toggle & Persistence (Light -> Dark -> Reload)
    await page.getByRole("button", { name: "Dark theme" }).click();
    await expect(page.locator("html")).toHaveClass(/dark/);
    await page.reload();
    await expect(page.locator("html")).toHaveClass(/dark/);
    await page.getByRole("button", { name: "Light theme" }).click();
    await expect(page.locator("html")).toHaveClass(/light/);

    // 3. Video Library: Upload real ABODA video10.avi
    await page.getByRole("link", { name: "Video Library" }).click();
    await expect(page).toHaveURL(/\/videos/);

    await page.getByRole("button", { name: "Upload Video" }).first().click();
    await page.getByLabel("Select video file").setInputFiles(ABODA_VIDEO10);
    await expect(page.getByText("video10.avi")).toBeVisible();
    await page.getByRole("button", { name: "Start Upload" }).click();

    await expect(page.getByText("Video uploaded")).toBeVisible({ timeout: 20_000 });
    await expect(page.getByRole("cell", { name: /video10\.avi/ }).first()).toBeVisible();

    // Verify duplicate upload rejected cleanly with 409
    await page.getByRole("button", { name: "Upload Video" }).first().click();
    await page.getByLabel("Select video file").setInputFiles(ABODA_VIDEO10);
    await page.getByRole("button", { name: "Start Upload" }).click();
    await expect(page.getByRole("alert")).toContainText("DUPLICATE_VIDEO");
    await page.getByRole("button", { name: "Close", exact: true }).click();

    // Open Video Details modal & verify metadata
    await page.getByRole("button", { name: "Details" }).first().click();
    const detailDialog = page.getByRole("dialog");
    await expect(detailDialog.getByText("720×480")).toBeVisible();
    await expect(detailDialog.getByText("SHA-256 Digest")).toBeVisible();
    await detailDialog.getByRole("button", { name: "Configure Processing Job" }).click();

    // 4. Processing Page: Submit validated ABODA window (ref=30, start=48, end=65, interval=1.0)
    await expect(page).toHaveURL(/\/processing/);
    await page.getByLabel("Reference Timestamp (s)").fill("30");
    await page.getByLabel("Start Timestamp (s)").fill("48");
    await page.getByLabel("End Timestamp (s, optional)").fill("65");
    await page.getByLabel("Sampling Interval (s)").fill("1");
    await page.getByRole("button", { name: "Start Processing" }).click();

    await expect(page.getByText("Processing job queued")).toBeVisible();
    await expect(page).toHaveURL(/jobId=/);

    // Wait for completion on real CUDA model
    await expect(page.getByText("Run Complete")).toBeVisible({ timeout: 90_000 });
    await expect(page.getByText(/17 \/ 17/).first()).toBeVisible();

    // Refresh browser and verify state remains accurate
    await page.reload();
    await expect(page.getByText("Run Complete")).toBeVisible();
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, "processing_light.png"), fullPage: true });

    // 5. Events Explorer: Verify persisted event, filter, open detail
    await page.getByRole("link", { name: "View Job Events" }).click();
    await expect(page).toHaveURL(/\/events/);
    await expect(page.getByText("Track #1").first()).toBeVisible();

    await page.getByLabel("Filter by confirmation status").selectOption("true");
    await expect(page.getByText("1 matching tracks")).toBeVisible();

    await page.getByRole("button", { name: "Inspect" }).first().click();
    const evDialog = page.getByRole("dialog");
    await expect(evDialog.getByText(/Persistent Scene-Change Track #1/)).toBeVisible();
    await expect(evDialog.getByText(/heuristic_score_not_probability/)).toBeVisible();
    await expect(evDialog.getByText("Monotonic State History")).toBeVisible();
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, "event_detail_light.png") });
    await evDialog.getByRole("button", { name: "Close", exact: true }).click();
    await expect(evDialog).toBeHidden();

    // 6. Evidence Viewer: Verify gallery images, lightbox, and side-by-side comparison
    await page.getByRole("link", { name: "Evidence", exact: true }).click();
    await expect(page).toHaveURL(/\/evidence/);
    await page.getByLabel("Filter by evidence type").selectOption("annotated_frame");
    const firstImg = page.locator("img").first();
    await expect(firstImg).toBeVisible();
    const imgWidth = await firstImg.evaluate((el: HTMLImageElement) => el.naturalWidth);
    expect(imgWidth).toBeGreaterThan(0);

    await page.getByRole("button", { name: "Inspect" }).first().click();
    const lbDialog = page.getByRole("dialog");
    await lbDialog.getByRole("button", { name: "Compare Event Frames" }).click();
    await expect(lbDialog.getByText("original frame", { exact: false })).toBeVisible();
    await expect(lbDialog.getByText("annotated frame", { exact: false })).toBeVisible();
    await expect(lbDialog.getByText("probability map", { exact: false })).toBeVisible();
    await lbDialog.getByRole("button", { name: "Close dialog" }).click();

    // 7. Analytics & Exports (Light + Dark mode visual check)
    await page.getByRole("link", { name: "Analytics" }).click();
    await expect(page).toHaveURL(/\/analytics/);
    await expect(page.getByText("Per-Video Comparison")).toBeVisible();
    await expect(page.locator("canvas").first()).toBeVisible();

    const downloadPromise = page.waitForEvent("download");
    await page.getByRole("button", { name: "JSON" }).click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toContain(".json");

    await page.getByRole("button", { name: "Dark theme" }).click();
    await expect(page.locator("html")).toHaveClass(/dark/);
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, "analytics_dark.png"), fullPage: true });

    // 8. Dashboard & System Status in Dark Mode
    await page.getByRole("link", { name: "Dashboard" }).click();
    await expect(page.getByText("Overview Dashboard")).toBeVisible();
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, "dashboard_dark.png"), fullPage: true });

    await page.getByRole("link", { name: "System Status" }).click();
    await expect(page.getByText("Model Loaded").first()).toBeVisible();
    await expect(page.getByText("Verified").first()).toBeVisible();

    // 9. Settings Persistence
    await page.getByRole("link", { name: "Settings" }).click();
    await page.getByLabel("Default Sampling Interval (seconds)").fill("1.5");
    await page.getByRole("button", { name: "Save Setting" }).click();
    await expect(page.getByText("Settings saved")).toBeVisible();

    // 10. Invalid Route 404
    await page.goto("/this-route-does-not-exist");
    await expect(page.getByText("Page not found")).toBeVisible();

    // 11. Mobile Responsiveness Check (390x844, no horizontal overflow)
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/dashboard");
    await expect(page.getByText("Overview Dashboard")).toBeVisible();
    const hasOverflow = await page.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth,
    );
    expect(hasOverflow).toBe(false);
    await page.getByRole("button", { name: "Open navigation menu" }).click();
    await expect(page.getByRole("link", { name: "Events", exact: true })).toBeVisible();
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, "dashboard_mobile.png") });

    expect(pageErrors).toEqual([]);
    expect(consoleErrors).toEqual([]);
  });
});

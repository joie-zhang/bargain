// Real-browser check against the live read-only server; needs Playwright Chromium.
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const path = require("node:path");
(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1080 },
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const base = process.env.AUDIT_TEST_URL || "http://127.0.0.1:8003";
  const output = path.resolve(
    __dirname,
    "../docs/analysis/paper_code_audit_20260907",
  );
  await page.goto(base);
  await page
    .getByRole("heading", { name: "What can we safely keep or remove?" })
    .waitFor();
  assert.match(await page.locator("#app").innerText(), /664/);
  await page.screenshot({
    path: path.join(output, "ui_overview.png"),
    fullPage: true,
  });
  await page
    .getByRole("link", { name: "Experiment audits", exact: true })
    .click();
  await page.getByLabel("Search audits").fill("e34");
  await page.getByRole("link", { name: "e34", exact: true }).click();
  await page
    .getByRole("heading", {
      name: "Speaking-order diagnostic and fitted crossovers",
    })
    .waitFor();
  await page.getByRole("link", { name: "Explore listed files" }).click();
  await page.waitForFunction(() =>
    document.getElementById("range")?.textContent.includes("27"),
  );
  await page.getByRole("link", { name: "File explorer", exact: true }).click();
  await page
    .getByLabel("File status")
    .selectOption("conditional-retirement-candidate");
  await page.getByLabel("Search paths").fill("utils");
  await page.waitForFunction(
    () => document.querySelectorAll("tbody tr").length === 7,
  );
  await page.locator("tbody tr a").first().click();
  await page
    .getByRole("heading", { name: "File detail", exact: true })
    .waitFor();
  assert.match(
    await page.locator("#app").innerText(),
    /No deletion is authorized/,
  );
  await page
    .getByRole("link", { name: "Cleanup candidates", exact: true })
    .click();
  await page.waitForFunction(
    () => document.querySelectorAll("tbody tr").length === 9,
  );
  await page.getByLabel("Candidate type").selectOption("safe-generated");
  await page.waitForFunction(
    () => document.querySelectorAll("tbody tr").length === 25,
  );
  await page.getByRole("link", { name: "Paper coverage", exact: true }).click();
  await page.waitForFunction(
    () => document.querySelectorAll("tbody tr").length === 40,
  );
  await page.getByRole("link", { name: "Review checks", exact: true }).click();
  await page.getByRole("heading", { name: "Review of the audit" }).waitFor();
  assert.equal(await page.locator(".check.fail").count(), 0);
  assert.equal(await page.locator(".check").count(), 15);
  await page.getByRole("link", { name: "Full report", exact: true }).click();
  await page.locator("section.report").waitFor();
  const downloaded = page.waitForEvent("download");
  await page
    .getByRole("link", { name: "Download Markdown report", exact: true })
    .click();
  assert.equal((await downloaded).suggestedFilename(), "report.md");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("link", { name: "Overview", exact: true }).click();
  await page
    .getByRole("heading", { name: "What can we safely keep or remove?" })
    .waitFor();
  assert(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
    "Mobile layout overflows",
  );
  await page.screenshot({
    path: path.join(output, "ui_mobile.png"),
    fullPage: true,
  });
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify({
      passed: true,
      views: 7,
      checks: [
        "e34 navigation",
        "audit file filter",
        "path search",
        "candidate type switch",
        "40 paper mappings",
        "15 review checks",
        "report download",
        "mobile width",
        "no page errors",
      ],
      screenshots: ["ui_overview.png", "ui_mobile.png"],
    }),
  );
  await browser.close();
})().catch((error) => {
  console.error(error);
  process.exit(1);
});

// Browser checks for the live read-only UI; never starts an experiment.
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const path = require("node:path");

(async () => {
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.BARGAIN_REVIEW_CHROMIUM
      ? { executablePath: process.env.BARGAIN_REVIEW_CHROMIUM }
      : {}),
  });
  try {
    const context = await browser.newContext({
      viewport: { width: 1440, height: 1080 },
      permissions: ["clipboard-read", "clipboard-write"],
    });
    const page = await context.newPage();
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    const base = process.env.BARGAIN_REVIEW_TEST_URL || "http://127.0.0.1:8004";
    const output = path.resolve(
      __dirname,
      "../docs/analysis/public_launch_review_20260907",
    );
    const review = await (
      await context.request.get(base + "/api/review")
    ).json();
    async function visit(route, selector = "#main h1") {
      await page.goto(base + "/#" + route);
      await page.locator(selector).first().waitFor();
      assert.equal(await page.locator(".notice.error").count(), 0, route);
      assert.equal(
        await page.evaluate(
          () => document.documentElement.scrollWidth > window.innerWidth + 1,
        ),
        false,
        "Page overflow: " + route,
      );
    }
    await visit("overview");
    assert.match(await page.locator(".metrics").innerText(), /7,160/);
    await page.screenshot({
      path: path.join(output, "ui_overview.png"),
      fullPage: true,
    });
    await page.locator(".skip-link").focus();
    await page.keyboard.press("Enter");
    assert.equal(await page.evaluate(() => document.activeElement.id), "main");
    assert.equal(new URL(page.url()).hash, "#overview");

    await visit("experiments");
    assert.equal(await page.locator(".family-card").count(), 7);
    await page.getByLabel("Experiment category").selectOption("Multi-agent");
    assert.equal(await page.locator(".family-card").count(), 3);
    await page.getByLabel("Search experiments").fill("heterogeneous");
    assert.equal(await page.locator(".family-card").count(), 1);

    for (const family of review.families) {
      await visit("experiment/" + family.id);
      assert.match(
        await page.locator("#command-text").innerText(),
        new RegExp("bargain run " + family.id),
      );
      await page.locator("#command-seed").fill("0");
      assert.match(await page.locator("#command-text").innerText(), /--seed 0/);
      await page.locator("#command-output").fill("/tmp/review path 'quoted'");
      const command = await page.locator("#command-text").innerText();
      assert.match(command, /--output '\/tmp\/review path/);
      await page.getByRole("button", { name: "Copy proposed command" }).click();
      assert.equal(
        await page.evaluate(() => navigator.clipboard.readText()),
        command,
      );
      await page.locator("#command-output").fill("relative-path");
      assert.equal(await page.locator("#copy-command").isDisabled(), true);
      assert.equal(await page.locator("#command-error").isVisible(), true);
      await page.locator("#command-output").fill("/tmp/review-output");
      await page.locator("#command-scope").selectOption("paper");
      assert.equal(await page.locator("#seed-label").isVisible(), false);
      assert.match(
        await page.locator("#command-text").innerText(),
        /--plan-only/,
      );
      assert.doesNotMatch(
        await page.locator("#command-text").innerText(),
        /--seed|--execute/,
      );
      assert.match(
        await page.locator("#command-explainer").innerText(),
        new RegExp(family.paper_runs.toLocaleString("en-US")),
      );
    }

    await visit("experiment/ttc");
    await page.locator("#command-effort").selectOption("claude");
    assert.match(
      await page.locator("#command-meta").innerText(),
      /ANTHROPIC_API_KEY/,
    );
    await page.locator("#command-effort").selectOption("gemini");
    assert.match(
      await page.locator("#command-meta").innerText(),
      /OPENROUTER_API_KEY/,
    );
    assert.doesNotMatch(
      await page.locator("#command-meta").innerText(),
      /GOOGLE_API_KEY/,
    );
    assert.match(
      await page.locator("#command-explainer").innerText(),
      /Four negotiations/,
    );
    await page.screenshot({
      path: path.join(output, "ui_experiment.png"),
      fullPage: true,
    });

    await visit("experiment/team");
    assert.equal(await page.locator("#command-game option").count(), 1);
    await page.locator("#command-agents").selectOption("2");
    assert.match(
      await page.locator("#command-explainer").innerText(),
      /no team treatment/,
    );

    await visit("findings");
    assert.equal(
      await page.locator("details.finding").count(),
      review.findings.length,
    );
    await page.getByLabel("Finding area").selectOption("Credentials");
    assert.ok((await page.locator("details.finding").count()) > 0);
    for (const finding of review.findings) {
      await visit("finding/" + finding.id);
      assert.equal(
        await page.locator("#finding-" + finding.id).getAttribute("open"),
        "",
      );
    }
    await visit("patch");
    assert.equal(
      await page.locator(".patch-stage").count(),
      review.patch_stages.length,
    );
    await visit("setup");
    assert.match(
      await page.locator("#main").innerText(),
      /Not a live authentication check/,
    );

    await visit("reports");
    assert.equal(await page.locator(".report-card").count(), 15);
    await page.getByLabel("Search reports").fill("credentials");
    assert.equal(await page.locator(".report-card").count(), 1);
    for (const report of review.reports)
      await visit("report/" + report.id, ".report-body");
    for (const source of review.sources)
      await visit("source/" + source.id, ".source-code");

    await visit("report/implementation", ".report-body");
    assert.ok((await page.locator(".report-body pre").count()) > 0);
    const downloadEvent = page.waitForEvent("download");
    await page.getByRole("link", { name: "Download Markdown" }).click();
    const download = await downloadEvent;
    assert.equal(download.suggestedFilename(), "implementation_proposal.md");
    assert.equal(await download.failure(), null);

    const escaped = await page.evaluate(() =>
      renderMarkdown(
        '<script>alert("not executed")</script>\n\n[bad](javascript:alert)',
      ),
    );
    assert.ok(!escaped.includes("<script>"));
    assert.ok(!escaped.includes('href="javascript:'));

    await page.setViewportSize({ width: 390, height: 844 });
    for (const route of [
      "overview",
      "experiments",
      "experiment/ttc",
      "findings",
      "patch",
      "setup",
      "reports",
      "report/overview",
      "source/runner",
    ]) {
      await visit(
        route,
        route.startsWith("report/")
          ? ".report-body"
          : route.startsWith("source/")
            ? ".source-code"
            : "#main h1",
      );
    }
    await visit("experiment/ttc");
    await page.screenshot({
      path: path.join(output, "ui_mobile.png"),
      fullPage: true,
    });
    assert.deepEqual(errors, []);
    console.log(
      JSON.stringify({
        passed: true,
        families: review.families.length,
        reports: review.reports.length,
        sources: review.sources.length,
        console_errors: errors.length,
        screenshots: output,
      }),
    );
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

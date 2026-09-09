const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

(async () => {
  const browser = await chromium.launch({headless:true,
    ...(process.env.BARGAIN_REVIEW_CHROMIUM ? {executablePath:process.env.BARGAIN_REVIEW_CHROMIUM} : {})});
  try {
    const context = await browser.newContext({viewport:{width:1440,height:1050}, permissions:["clipboard-read","clipboard-write"]});
    const page = await context.newPage();
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => { if (message.type() === "error") errors.push(message.text()); });
    const base = process.env.BARGAIN_README_TEST_URL || "http://127.0.0.1:8005";
    const source = fs.readFileSync(path.resolve(__dirname,"../README-old.md"),"utf8");
    await page.goto(base);
    await page.locator('#document[aria-busy="false"]').waitFor();
    assert.equal(await page.locator("#error").isHidden(),true);
    assert.equal(await page.locator("#document h2").count(),(source.match(/^## /gm)||[]).length);
    const expectedCode = [...source.matchAll(/```[^\n]*\n([\s\S]*?)\n```/g)].map((m)=>m[1]);
    assert.deepEqual(await page.locator("pre code").allTextContents(),expectedCode);
    await page.getByRole("button",{name:"Copy code block"}).first().click();
    assert.equal(await page.evaluate(()=>navigator.clipboard.readText()),expectedCode[0]);
    await page.getByLabel("Find a section or command").fill("heterogeneous");
    assert.equal(await page.locator("#document section:visible").count(),1);
    assert.match(await page.locator("#document section:visible").first().innerText(),/Heterogeneous groups/);
    await page.getByLabel("Find a section or command").fill("no-section-with-this-text");
    assert.equal(await page.locator("#empty").isVisible(),true);
    await page.getByLabel("Find a section or command").fill("");
    await page.locator("nav").getByRole("link",{name:"Coordinated team",exact:true}).click();
    assert.match(page.url(),/coordinated-team$/);
    await page.getByRole("button",{name:"Refresh",exact:true}).click();
    await page.locator('#document[aria-busy="false"]').waitFor();
    assert.deepEqual(await page.locator("pre code").allTextContents(),expectedCode);
    const raw = await context.request.get(base+"/download");
    assert.equal(await raw.text(),source);
    await page.evaluate(()=>{
      const section=renderMarkdown('<img src=x onerror="window.bad=true">\n\n```bash\n<script>window.bad=true</script>\n```')[0];
      const node=document.createElement("div");node.innerHTML=section.html.join("");document.body.append(node);
      window.testMarkup=node;
    });
    assert.equal(await page.evaluate(()=>window.testMarkup.querySelectorAll("img,script").length),0);
    await page.evaluate(()=>window.testMarkup.remove());
    await page.goto(base);
    await page.locator('#document[aria-busy="false"]').waitFor();
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
    const output=process.env.BARGAIN_README_SCREENSHOT_DIR;
    if(output) await page.screenshot({path:path.join(output,"desktop.png"),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
    if(output) await page.screenshot({path:path.join(output,"mobile.png"),fullPage:true});
    assert.deepEqual(errors,[]);
    console.log("Browser checks passed: source fidelity, copy, search, navigation, refresh, download, HTML escaping, desktop/mobile layout.");
  } finally { await browser.close(); }
})().catch((error)=>{console.error(error);process.exit(1);});

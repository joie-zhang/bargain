const {chromium}=require('playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.BARGAIN_REVIEW_CHROMIUM ? {executablePath:process.env.BARGAIN_REVIEW_CHROMIUM} : {})});
 const page=await browser.newPage({viewport:{width:1500,height:1100}});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.BARGAIN_REVIEW_TEST_URL || 'http://127.0.0.1:8010');
 await page.getByText('Conversation',{exact:true}).waitFor({timeout:60000});
 assert.equal(await page.locator('[data-testid="stException"]').count(),0);
 assert.equal(await page.locator('.js-plotly-plot').count(),0);
 await page.getByLabel('Search response text',{exact:true}).fill('impossible_search_982123');
 await page.getByLabel('Search response text',{exact:true}).press('Enter');
 await page.getByText('0 matching messages',{exact:false}).waitFor();
 await page.getByLabel('Search response text',{exact:true}).fill('');
 await page.getByLabel('Search response text',{exact:true}).press('Enter');
 await page.locator('[data-testid="stChatMessage"]').first().waitFor();
 assert.equal(await page.locator('[data-testid="stException"]').count(),0);

 await page.setViewportSize({width:390,height:844});

 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth+1),false);
 console.log('PASS: transcripts render without a map; search works; desktop/mobile render; no app exceptions.',errors);
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});

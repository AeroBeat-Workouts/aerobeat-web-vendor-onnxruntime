// @ts-check

import assert from "node:assert/strict";
import { chromium } from "playwright";
import { createServer } from "vite";

const server = await createServer({
  root: ".testbed/demo",
  logLevel: "silent",
  server: { host: "127.0.0.1", port: 0 }
});
let browser;
try {
  await server.listen();
  const url = server.resolvedUrls?.local[0];
  assert.ok(url, "Vite did not report a local testbed URL.");
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const consoleFailures = [];
  page.on("console", (message) => {
    if (["warning", "error"].includes(message.type())) consoleFailures.push(`${message.type()}: ${message.text()}`);
  });
  page.on("pageerror", (error) => consoleFailures.push(`pageerror: ${error.message}`));
  await page.goto(url, { waitUntil: "networkidle" });
  await page.locator("[data-state=passed]").waitFor();
  assert.equal(await page.locator("[data-state=passed]").textContent(), "ONNX Runtime replay smoke passed (7 landmarks, no model download)." );
  assert.deepEqual(consoleFailures, []);
  console.log(`ONNX Runtime browser-safe smoke passed at ${url}`);
} finally {
  await browser?.close();
  await server.close();
}

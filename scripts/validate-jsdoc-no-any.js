// @ts-check

import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";

const sourceRoot = resolve("src");
for (const name of await readdir(sourceRoot)) {
  if (!name.endsWith(".js")) continue;
  const source = await readFile(resolve(sourceRoot, name), "utf8");
  assert.doesNotMatch(source, /@(?:type|param|returns?)\s*\{\s*any\s*\}/u, `${name} contains an explicit any escape`);
  assert.match(source, /^\/\/ @ts-check/u, `${name} must enable @ts-check`);
}
console.log("JSDoc/no-any validation passed.");

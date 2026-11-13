import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {test} from "node:test";

test("the application version matches both npm lockfile root fields", () => {
    const manifest = JSON.parse(readFileSync(new URL("../../package.json", import.meta.url), "utf8"));
    const lockfile = JSON.parse(readFileSync(new URL("../../package-lock.json", import.meta.url), "utf8"));

    assert.match(manifest.version, /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/);
    assert.equal(lockfile.version, manifest.version);
    assert.equal(lockfile.packages[""].version, manifest.version);
});

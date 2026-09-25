import {existsSync, readFileSync} from "node:fs";
import {expect, test} from "@playwright/test";

test("production output contains a generated worker with precache and cache policy", () => {
    const workerPath = "dist/sw.js";

    expect(existsSync(workerPath)).toBe(true);
    expect(existsSync("public/sw.js")).toBe(false);

    const worker = readFileSync(workerPath, "utf8");
    expect(worker).not.toContain("__WB_MANIFEST");
    expect(worker).toContain("reclive-images");
    expect(worker).toContain("reclive-fonts");
    expect(worker).toContain("SKIP_WAITING");
    expect(worker).toContain("manifest.webmanifest");
});

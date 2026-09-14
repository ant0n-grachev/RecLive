import assert from "node:assert/strict";
import fs from "node:fs";
import {createRequire} from "node:module";
import os from "node:os";
import path from "node:path";
import {spawnSync} from "node:child_process";
import {pathToFileURL, fileURLToPath} from "node:url";
import vm from "node:vm";
import {test} from "node:test";
import {transformSync} from "esbuild";

const root = fileURLToPath(new URL("../../", import.meta.url));
const require = createRequire(new URL("../../vite.config.ts", import.meta.url));
const guardPath = path.join(root, "tests/frontend/no-checkout-dotenv.cjs");
const launcherPath = path.join(root, "tests/frontend/run-isolated.cjs");

// Evaluate the user config export only: resolving Vite's default config could read dotenv.
function configWith(switchValue) {
    const env = {};
    if (switchValue !== undefined) env.RECLIVE_TEST_NO_DOTENV = switchValue;
    const source = fs.readFileSync(path.join(root, "vite.config.ts"), "utf8");
    const {code} = transformSync(source, {loader: "ts", format: "cjs"});
    const module = {exports: {}};
    vm.runInNewContext(code, {module, exports: module.exports, require, process: {env}});
    return module.exports.default;
}

test("unset switch preserves the config default; exactly 1 prevents dotenv candidates", () => {
    assert.equal(configWith(undefined).envDir, undefined);
    assert.equal(configWith("1").envDir, false);
});

test("blank and malformed test switches fail with a name-only message", () => {
    for (const value of ["", "synthetic-invalid-switch"]) {
        assert.throws(() => configWith(value), {
            message: "Unsafe test environment configuration: RECLIVE_TEST_NO_DOTENV",
        });
    }
});

test("preload blocks content APIs and path aliases using synthetic files only", async () => {
    assert.ok(fs.existsSync(guardPath), "dotenv content-read guard is missing");
    const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "reclive-dotenv-"));
    const protectedFile = path.join(fixture, ".env.production.local");
    const ordinary = path.join(fixture, "ordinary.txt");
    const alias = path.join(fixture, "alias.txt");
    const outside = fs.mkdtempSync(path.join(os.tmpdir(), "reclive-outside-"));
    const target = path.join(outside, "target.txt");
    const dangling = path.join(outside, "missing.txt");
    fs.writeFileSync(protectedFile, "SYNTHETIC=fixture-only\n");
    fs.writeFileSync(ordinary, "ordinary fixture");
    fs.symlinkSync(protectedFile, alias);
    fs.writeFileSync(target, "synthetic outside target");
    fs.symlinkSync(target, path.join(fixture, ".env.external"));
    fs.symlinkSync(dangling, path.join(fixture, ".env.dangling"));
    fs.symlinkSync(fixture, path.join(outside, "root-alias"));
    fs.mkdirSync(path.join(outside, "subdir"));
    fs.symlinkSync(path.join(outside, "subdir"), path.join(fixture, "ancestor-alias"));
    fs.writeFileSync(path.join(outside, "ordinary.txt"), "ordinary outside fixture");
    fs.mkdirSync(path.join(fixture, "nested"));
    const {installGuard} = require(guardPath);
    const removeFixtureGuard = installGuard(fixture);
    const failure = {message: "Checkout dotenv content access blocked by test guard"};
    try {
        for (const requested of [
            protectedFile, Buffer.from(protectedFile), pathToFileURL(protectedFile),
            `${fixture}/nested/../.env.production.local`,
            path.relative(process.cwd(), protectedFile), alias,
            path.join(fixture, ".env"), path.join(fixture, ".env.staging"),
            target, dangling, path.join(outside, "root-alias", ".env.production.local"),
            `${fixture}/ancestor-alias/../target.txt`,
        ]) {
            assert.throws(() => fs.readFileSync(requested), failure);
            assert.throws(() => fs.readFile(requested, () => {}), failure);
            assert.throws(() => fs.openSync(requested, "r"), failure);
            assert.throws(() => fs.open(requested, "r", () => {}), failure);
            assert.throws(() => fs.createReadStream(requested), failure);
            await assert.rejects(fs.promises.readFile(requested), failure);
            await assert.rejects(fs.promises.open(requested, "r"), failure);
        }
        const esmFs = await import("node:fs");
        assert.throws(() => esmFs.readFileSync(protectedFile), failure);
        const esmPromises = await import("node:fs/promises");
        await assert.rejects(esmPromises.readFile(protectedFile), failure);
        assert.equal(fs.readFileSync(`${fixture}/ancestor-alias/../ordinary.txt`, "utf8"), "ordinary outside fixture");
        assert.equal(fs.readFileSync(ordinary, "utf8"), "ordinary fixture");
        assert.equal(await fs.promises.readFile(ordinary, "utf8"), "ordinary fixture");
        await new Promise((resolve, reject) => fs.readFile(ordinary, "utf8", (error, value) => {
            if (error) reject(error);
            else { assert.equal(value, "ordinary fixture"); resolve(); }
        }));
        const handle = await fs.promises.open(ordinary, "r");
        await handle.close();
        fs.closeSync(fs.openSync(ordinary, "r"));
        const changedTarget = path.join(outside, "changed.txt");
        fs.writeFileSync(changedTarget, "changed synthetic target");
        fs.unlinkSync(path.join(fixture, ".env.external"));
        fs.symlinkSync(changedTarget, path.join(fixture, ".env.external"));
        assert.throws(() => fs.readFileSync(changedTarget), failure);
        assert.equal(fs.readFileSync(target, "utf8"), "synthetic outside target");
    } finally {
        removeFixtureGuard();
        fs.rmSync(fixture, {recursive: true, force: true});
        fs.rmSync(outside, {recursive: true, force: true});
    }
});

test("case aliases of protected files and parents cannot bypass content APIs", async t => {
    const parent = fs.mkdtempSync(path.join(os.tmpdir(), "reclive-case-"));
    const fixture = path.join(parent, "ProjectRoot");
    fs.mkdirSync(fixture);
    const protectedFile = path.join(fixture, ".env.production.local");
    const upperFile = path.join(fixture, ".ENV.PRODUCTION.LOCAL");
    fs.writeFileSync(protectedFile, "synthetic case marker");
    fs.writeFileSync(path.join(fixture, "ordinary.txt"), "ordinary case marker");
    const aliasesExist = fs.existsSync(upperFile);
    if (!aliasesExist) fs.writeFileSync(upperFile, "distinct ordinary uppercase marker");
    const {installGuard} = require(guardPath);
    const dispose = installGuard(fixture);
    try {
        if (aliasesExist) {
            const failure = {message: "Checkout dotenv content access blocked by test guard"};
            for (const requested of [upperFile, path.join(parent, "projectroot", ".env.production.local"),
                path.join(parent, "PROJECTROOT", ".EnV.PrOdUcTiOn.LoCaL")]) {
                assert.throws(() => fs.readFileSync(requested), failure);
                assert.throws(() => fs.readFile(requested, () => {}), failure);
                assert.throws(() => fs.openSync(requested, "r"), failure);
                assert.throws(() => fs.open(requested, "r", () => {}), failure);
                assert.throws(() => fs.createReadStream(requested), failure);
                await assert.rejects(fs.promises.readFile(requested), failure);
                await assert.rejects(fs.promises.open(requested, "r"), failure);
            }
            assert.equal(fs.readFileSync(path.join(parent, "PROJECTROOT", "ORDINARY.TXT"), "utf8"), "ordinary case marker");
        } else {
            t.diagnostic("Case-sensitive storage: verifying distinct ordinary uppercase entry");
            assert.equal(fs.readFileSync(upperFile, "utf8"), "distinct ordinary uppercase marker");
        }
    } finally {
        dispose();
        assert.equal(fs.readFileSync(protectedFile, "utf8"), "synthetic case marker");
        fs.rmSync(parent, {recursive: true, force: true});
    }
});

test("alternate-case directory entries are protected only when lowercase names alias them", t => {
    const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "reclive-case-entry-"));
    const upperFile = path.join(fixture, ".ENV.PRODUCTION.LOCAL");
    const lowerFile = path.join(fixture, ".env.production.local");
    fs.writeFileSync(upperFile, "synthetic uppercase entry marker");
    const aliasesExist = fs.existsSync(lowerFile);
    const {installGuard} = require(guardPath);
    const dispose = installGuard(fixture);
    try {
        if (aliasesExist) {
            for (const requested of [upperFile, lowerFile]) {
                assert.throws(() => fs.readFileSync(requested), {
                    message: "Checkout dotenv content access blocked by test guard",
                });
            }
        } else {
            t.diagnostic("Case-sensitive storage: uppercase-only entry is ordinary");
            assert.equal(fs.readFileSync(upperFile, "utf8"), "synthetic uppercase entry marker");
        }
    } finally {
        dispose();
        fs.rmSync(fixture, {recursive: true, force: true});
    }
});

test("protected file identity also blocks hard-link path aliases without conflating ordinary files", () => {
    const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "reclive-identity-"));
    const protectedFile = path.join(fixture, ".env.test");
    const alias = path.join(fixture, "ordinary-alias.txt");
    fs.writeFileSync(protectedFile, "synthetic identity marker");
    fs.linkSync(protectedFile, alias);
    fs.writeFileSync(path.join(fixture, "ordinary.txt"), "distinct ordinary marker");
    const {installGuard} = require(guardPath);
    const dispose = installGuard(fixture);
    try {
        assert.throws(() => fs.readFileSync(alias), {message: "Checkout dotenv content access blocked by test guard"});
        assert.equal(fs.readFileSync(path.join(fixture, "ordinary.txt"), "utf8"), "distinct ordinary marker");
    } finally {
        dispose();
        fs.rmSync(fixture, {recursive: true, force: true});
    }
});

test("launcher preserves runner inputs and invokes the original npm script with inherited guard", () => {
    assert.ok(fs.existsSync(launcherPath), "isolated npm launcher is missing");
    const {createGateEnv} = require(launcherPath);
    const env = createGateEnv({
        PATH: "runner-path", HOME: "runner-home", CI: "1", npm_config_cache: "runner-cache",
        VITE_PRIVATE_SENTINEL: "synthetic-only", VITE_API_BASE_URL: "synthetic-conflict",
        FORCE_COLOR: "1", NO_COLOR: "1", DEBUG: "vite:env", NODE_OPTIONS: "--max-old-space-size=4096",
    }, "test:e2e");
    assert.equal(env.PATH, "runner-path");
    assert.equal(env.HOME, "runner-home");
    assert.equal(env.npm_config_cache, "runner-cache");
    assert.equal(env.CI, "1");
    assert.equal(env.RECLIVE_TEST_NO_DOTENV, "1");
    assert.deepEqual(Object.keys(env).filter(name => name.startsWith("VITE_")).sort(), ["VITE_API_BASE_URL", "VITE_SITE_URL"]);
    assert.equal(env.VITE_API_BASE_URL, "http://127.0.0.1:4173");
    assert.equal(env.VITE_SITE_URL, "http://127.0.0.1:4173");
    assert.equal(env.NODE_OPTIONS, `--require=${JSON.stringify(guardPath)}`);
    for (const name of ["FORCE_COLOR", "NO_COLOR", "DEBUG"]) assert.equal(env[name], undefined);
    const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "reclive-npm-"));
    fs.writeFileSync(path.join(fixture, "npm"), `#!${process.execPath}\nconst assert = require('node:assert/strict');\nassert.deepEqual(process.argv.slice(2), ['run', 'test:e2e']);\nassert.equal(process.env.CI, '1');\nassert.equal(process.env.RECLIVE_TEST_NO_DOTENV, '1');\nassert.ok(process.env.NODE_OPTIONS.includes('no-checkout-dotenv.cjs'));\nconsole.log('original npm script invoked under inherited guard');\n`, {mode: 0o700});
    try {
        const result = spawnSync(process.execPath, [launcherPath, "test:e2e"], {
            encoding: "utf8", env: {...process.env, PATH: `${fixture}${path.delimiter}${process.env.PATH}`},
        });
        assert.equal(result.status, 0, result.stderr);
        assert.match(result.stdout, /original npm script invoked under inherited guard/);
    } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
    }
});

test("launcher accepts the original coverage script", () => {
    const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "reclive-coverage-"));
    fs.writeFileSync(path.join(fixture, "npm"), `#!${process.execPath}\nrequire('node:assert/strict').deepEqual(process.argv.slice(2), ['run', 'test:coverage']);\n`, {mode: 0o700});
    try {
        const result = spawnSync(process.execPath, [launcherPath, "test:coverage"], {
            encoding: "utf8", env: {...process.env, NODE_OPTIONS: "", PATH: `${fixture}${path.delimiter}${process.env.PATH}`},
        });
        assert.equal(result.status, 0, result.stderr);
    } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
    }
});

test("npm lifecycle configuration cannot replace the child dotenv preload", () => {
    const {createGateEnv} = require(launcherPath);
    const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "reclive-npm-options-"));
    fs.writeFileSync(path.join(fixture, "package.json"), JSON.stringify({
        private: true, scripts: {verify: "node probe.cjs"},
    }));
    fs.writeFileSync(path.join(fixture, ".npmrc"), "node-options=--max-old-space-size=512\n");
    fs.writeFileSync(path.join(fixture, "probe.cjs"), `if (process.env.NODE_OPTIONS !== process.env.EXPECTED_TEST_PRELOAD) { console.error("npm lifecycle replaced the required test preload"); process.exitCode = 1; } else { console.log("npm lifecycle preserved the required test preload"); }\n`);
    try {
        const env = createGateEnv({
            ...process.env,
            npm_config_node_options: "--max-old-space-size=512",
            NPM_CONFIG_NODE_OPTIONS: "--max-old-space-size=512",
        }, "build");
        env.EXPECTED_TEST_PRELOAD = `--require=${JSON.stringify(guardPath)}`;
        const result = spawnSync("npm", ["run", "verify"], {cwd: fixture, env, encoding: "utf8"});
        assert.equal(result.status, 0, result.stderr);
        assert.match(result.stdout, /npm lifecycle preserved the required test preload/);
    } finally {
        fs.rmSync(fixture, {recursive: true, force: true});
    }
});

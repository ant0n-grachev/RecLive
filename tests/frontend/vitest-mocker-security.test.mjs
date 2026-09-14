import "./no-checkout-dotenv.cjs";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {test} from "node:test";
import {resolveConfig} from "vite";
import {MockerRegistry} from "@vitest/mocker";
import {interceptorPlugin} from "@vitest/mocker/node";

// No listener: only the transport is replaced; registry, URL parsing and file loading are real.
async function fixture(t, options = {}) {
    const parent = fs.mkdtempSync(path.join(os.tmpdir(), "reclive-mocker-"));
    t.after(() => fs.rmSync(parent, {recursive: true, force: true}));
    const root = path.join(parent, "project");
    fs.mkdirSync(root);
    fs.writeFileSync(path.join(root, "allowed.js"), "export const marker = 'allowed-synthetic';\n");
    fs.writeFileSync(path.join(root, "denied.blocked"), "export const marker = 'denied-synthetic';\n");
    fs.writeFileSync(path.join(parent, "outside.js"), "export const marker = 'outside-synthetic';\n");
    const config = await resolveConfig({
        root, configFile: false, envDir: false,
        server: {fs: {strict: true, allow: [root], deny: ["**/*.blocked"]}},
        logLevel: "silent",
    }, "serve");
    const handlers = new Map();
    const acknowledgements = [];
    const registry = new MockerRegistry();
    const plugin = interceptorPlugin({...options, registry});
    plugin.configureServer({config, ws: {
        on: (event, handler) => handlers.set(event, handler),
        send: event => acknowledgements.push(event),
    }});
    return {
        handlers, acknowledgements, registry,
        async loadRedirect(redirect) {
            const id = path.join(root, "requested.js");
            const register = handlers.get("vitest:interceptor:register");
            assert.equal(typeof register, "function");
            await register({type: "redirect", raw: "./requested.js", id, url: "/requested.js", redirect});
            const loaded = await plugin.load.handler(id);
            return {loaded, registered: registry.getById(id) !== undefined};
        },
    };
}

test("ordinary in-policy redirect still returns its real module content", async t => {
    const boundary = await fixture(t);
    const result = await boundary.loadRedirect("http://synthetic.test/allowed.js");
    assert.equal(result.registered, true);
    assert.equal(result.loaded, "export const marker = 'allowed-synthetic';\n");
    assert.deepEqual(boundary.acknowledgements, ["vitest:interceptor:register:result"]);
});

test("opaque-scheme outside-policy redirect cannot reach the real file-read sink", async t => {
    const boundary = await fixture(t);
    const result = await boundary.loadRedirect("synthetic:../outside.js");
    assert.equal(result.loaded, undefined, "outside-policy synthetic content reached the load hook");
    assert.equal(result.registered, false);
    assert.deepEqual(boundary.acknowledgements, ["vitest:interceptor:register:result"]);
});

test("in-root denied-pattern redirect cannot enter the file-read registry", async t => {
    const boundary = await fixture(t);
    const result = await boundary.loadRedirect("http://synthetic.test/denied.blocked");
    assert.equal(result.loaded, undefined, "denied-pattern synthetic content reached the load hook");
    assert.equal(result.registered, false);
    assert.deepEqual(boundary.acknowledgements, ["vitest:interceptor:register:result"]);
});

test("disabled raw WebSocket registration installs no transport handlers", async t => {
    const boundary = await fixture(t, {registerWebSocketEvents: false});
    assert.equal(boundary.handlers.size, 0);
});

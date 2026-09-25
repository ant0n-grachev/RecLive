// Invoke via `env -u NODE_OPTIONS node ...` so unrelated preloads cannot run first.
const path = require("node:path");
const {spawnSync} = require("node:child_process");
const scripts = new Set(["build", "test:run", "test:coverage", "test:e2e"]);

function createGateEnv(inherited, script) {
    const env = {...inherited};
    for (const name of Object.keys(env)) {
        if (name.startsWith("VITE_") || ["FORCE_COLOR", "NO_COLOR", "DEBUG", "NODE_OPTIONS"].includes(name)
            || name.toLowerCase().replaceAll("-", "_") === "npm_config_node_options") {
            delete env[name];
        }
    }
    env.RECLIVE_TEST_NO_DOTENV = "1";
    env.VITE_API_BASE_URL = "http://127.0.0.1:4173";
    env.VITE_SITE_URL = "http://127.0.0.1:4173";
    env.NODE_OPTIONS = `--require=${JSON.stringify(path.join(__dirname, "no-checkout-dotenv.cjs"))}`;
    // npm can overwrite lifecycle NODE_OPTIONS from this higher-precedence config.
    env.npm_config_node_options = env.NODE_OPTIONS;
    if (script === "test:e2e") env.CI = "1";
    return env;
}

if (require.main === module) {
    const [script, ...extra] = process.argv.slice(2);
    if (!scripts.has(script) || extra.length) {
        console.error("Expected one isolated frontend script: build, test:run, test:coverage, or test:e2e");
        process.exitCode = 1;
    } else {
        const result = spawnSync("npm", ["run", script], {
            cwd: path.resolve(__dirname, "../.."), env: createGateEnv(process.env, script), stdio: "inherit",
        });
        if (result.error) console.error("Isolated frontend command could not start");
        process.exitCode = result.status ?? 1;
    }
}
module.exports = {createGateEnv};

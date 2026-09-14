// Bounded Node test tripwire, not an OS sandbox. Never logs paths or values.
const fs = require("node:fs");
const path = require("node:path");
const {fileURLToPath} = require("node:url");
const {syncBuiltinESMExports} = require("node:module");
const roots = new Set();
const message = "Checkout dotenv content access blocked by test guard";
const isDotenv = name => name === ".env" || name.startsWith(".env.");

function identity(value, followLinks, cache) {
    const key = `${followLinks ? "stat" : "lstat"}:${value}`;
    if (cache.has(key)) return cache.get(key);
    try {
        const result = (followLinks ? fs.statSync : fs.lstatSync)(value, {bigint: true});
        cache.set(key, result);
        return result;
    }
    catch (error) {
        if (error.code === "ENOENT" || error.code === "ENOTDIR") {
            cache.set(key, undefined);
            return undefined;
        }
        throw new Error(message);
    }
}

function sameIdentity(left, right, cache, followLinks = true) {
    const a = identity(left, followLinks, cache);
    const b = identity(right, followLinks, cache);
    return a !== undefined && b !== undefined && a.dev === b.dev && a.ino === b.ino;
}

function samePath(left, right, cache) {
    return left === right || sameIdentity(left, right, cache)
        || (path.basename(left) === path.basename(right)
            && sameIdentity(path.dirname(left), path.dirname(right), cache));
}

function protectedEntry(root, name, cache) {
    if (isDotenv(name)) return true;
    const lower = name.toLowerCase();
    // A case-varied entry is protected only if the filesystem aliases the lowercase
    // dotenv name to that same entry. Distinct case-sensitive ordinary files stay readable.
    return isDotenv(lower) && sameIdentity(path.join(root, name), path.join(root, lower), cache, false);
}

// Resolve metadata only, retaining native symlink/.. ordering and dangling targets.
function resolvedPath(value, depth = 0) {
    if (depth > 40) throw new Error(message);
    const absolute = path.isAbsolute(value) ? value : `${process.cwd()}${path.sep}${value}`;
    const parts = absolute.split(path.sep);
    let current = path.parse(absolute).root;
    for (let index = 0; index < parts.length; index++) {
        const part = parts[index];
        if (!part || part === ".") continue;
        if (part === "..") { current = path.dirname(current); continue; }
        current = path.join(current, part);
        try {
            if (fs.lstatSync(current).isSymbolicLink()) {
                const target = fs.readlinkSync(current);
                const linked = path.isAbsolute(target) ? target : `${path.dirname(current)}${path.sep}${target}`;
                return resolvedPath(`${linked}${path.sep}${parts.slice(index + 1).join(path.sep)}`, depth + 1);
            }
        } catch (error) {
            if (error.code !== "ENOENT" && error.code !== "ENOTDIR") throw new Error(message);
        }
    }
    return current;
}

function check(requested) {
    if (typeof requested === "number") return; // Existing descriptors are outside this path tripwire.
    const value = requested instanceof URL ? fileURLToPath(requested)
        : Buffer.isBuffer(requested) ? requested.toString() : requested;
    if (typeof value !== "string") return;
    const lexical = path.resolve(value);
    const physical = resolvedPath(value);
    // Deduplicate identity probes only within this access; never retain stale aliases.
    const identities = new Map();
    for (const root of roots) {
        const physicalRoot = resolvedPath(root);
        if ([lexical, physical].some(candidate =>
            isDotenv(path.basename(candidate))
                && [root, physicalRoot].some(parent => samePath(parent, path.dirname(candidate), identities)))) {
            throw new Error(message);
        }
        // Refresh metadata for each access so newly changed or dangling aliases remain protected.
        let names;
        try { names = fs.readdirSync(root).filter(name => protectedEntry(root, name, identities)); }
        catch { throw new Error(message); }
        for (const name of names) {
            const target = resolvedPath(path.join(root, name));
            if (samePath(lexical, target, identities) || samePath(physical, target, identities)) throw new Error(message);
        }
    }
}

for (const name of ["readFileSync", "readFile", "openSync", "open", "createReadStream"]) {
    const original = fs[name];
    fs[name] = function (requested, ...args) {
        check(requested);
        return Reflect.apply(original, this, [requested, ...args]);
    };
}
for (const name of ["readFile", "open"]) {
    const original = fs.promises[name];
    fs.promises[name] = async function (requested, ...args) {
        check(requested);
        return Reflect.apply(original, this, [requested, ...args]);
    };
}
syncBuiltinESMExports();

function installGuard(root) {
    const normalized = path.resolve(root);
    roots.add(normalized);
    return () => roots.delete(normalized);
}
installGuard(path.resolve(__dirname, "../.."));
module.exports = {installGuard};

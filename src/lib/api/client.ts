import axios from "axios";
import type {ZodType} from "zod";
import {env} from "../config/env";

const DEFAULT_TIMEOUT_MS = 10_000;
const MAX_TIMEOUT_MS = 30_000;
const MAX_ATTEMPTS = 5;
const BASE_RETRY_DELAY_MS = 500;
const MAX_RETRY_DELAY_MS = 60_000;
const IMF_FIXDATE_PATTERN = /^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), [0-9]{2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) [0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2} GMT$/u;
const AXIOS_CLIENT_ERROR_CODES = new Set([
    "ERR_BAD_OPTION",
    "ERR_BAD_OPTION_VALUE",
    "ERR_BAD_REQUEST",
    "ERR_DEPRECATED",
    "ERR_INVALID_URL",
    "ERR_NOT_SUPPORT",
]);
const AXIOS_NETWORK_ERROR_CODES = new Set([
    "EAI_AGAIN",
    "ECONNREFUSED",
    "ECONNRESET",
    "EHOSTUNREACH",
    "ENETUNREACH",
    "ENOTFOUND",
    "EPIPE",
    "ERR_NETWORK",
]);

export type ApiErrorKind = (
    "aborted"
    | "network"
    | "timeout"
    | "http"
    | "invalid_json"
    | "schema"
    | "client"
);

export class ApiError extends Error {
    readonly kind: ApiErrorKind;
    readonly status: number | null;
    readonly retryAfterMs: number | null;

    constructor(
        kind: ApiErrorKind,
        message: string,
        status: number | null = null,
        retryAfterMs: number | null = null,
    ) {
        super(message);
        this.name = "ApiError";
        this.kind = kind;
        this.status = status;
        this.retryAfterMs = retryAfterMs;
    }
}

export const shouldRetryApiError = (error: ApiError): boolean => {
    if (error.kind === "network" || error.kind === "timeout") return true;
    if (error.kind !== "http" || error.status === null) return false;
    return (
        error.status === 408
        || error.status === 429
        || (error.status >= 500 && error.status <= 599)
    );
};

const invalidPathError = (): ApiError => new ApiError(
    "client",
    "Invalid API request path",
);

const canonicalApiPath = (value: string): string => {
    if (
        typeof value !== "string"
        || !value
        || value !== value.trim()
        || value.startsWith("//")
        || /[\\?#%\s]/u.test(value)
        || !/^[A-Za-z0-9_~./-]+$/u.test(value)
    ) {
        throw invalidPathError();
    }

    const withoutLeadingSlash = value.startsWith("/") ? value.slice(1) : value;
    const rawSegments = withoutLeadingSlash.split("/");
    if (rawSegments.some((segment) => segment === "." || segment === "..")) {
        throw invalidPathError();
    }
    const segments = rawSegments.filter(Boolean);
    if (segments[0] !== "api") {
        throw invalidPathError();
    }
    return `/${segments.join("/")}`;
};

export const uniqueApiUrls = (values: string[]): string[] => (
    [...new Set(values.map(canonicalApiPath))]
);

export interface RequestOptions {
    signal?: AbortSignal;
    timeoutMs?: number;
    attempts?: number;
    params?: Record<string, string | number>;
    method?: "GET" | "POST" | "DELETE";
    body?: unknown;
    jitter?: () => number;
    sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
}

const boundedDelay = (value: number): number => (
    Math.min(MAX_RETRY_DELAY_MS, Math.max(0, Math.round(value)))
);

export const parseRetryAfterMs = (value: unknown, now = Date.now()): number | null => {
    if (typeof value !== "string") return null;
    const text = value.trim();
    if (!text) return null;

    if (/^[0-9]+$/u.test(text)) {
        const seconds = Number(text);
        return Number.isFinite(seconds) ? boundedDelay(seconds * 1_000) : MAX_RETRY_DELAY_MS;
    }
    if (!IMF_FIXDATE_PATTERN.test(text) || !Number.isFinite(now)) return null;

    const timestamp = Date.parse(text);
    if (!Number.isFinite(timestamp) || new Date(timestamp).toUTCString() !== text) return null;
    return boundedDelay(timestamp - now);
};

const abortedError = (): ApiError => new ApiError("aborted", "Request aborted");

const throwIfAborted = (signal?: AbortSignal): void => {
    if (signal?.aborted) throw abortedError();
};

const abortableSleep = (ms: number, signal?: AbortSignal): Promise<void> => new Promise(
    (resolve, reject) => {
        if (signal?.aborted) {
            reject(abortedError());
            return;
        }
        const cleanup = () => {
            globalThis.clearTimeout(timer);
            signal?.removeEventListener("abort", handleAbort);
        };
        const finish = () => {
            cleanup();
            resolve();
        };
        const handleAbort = () => {
            cleanup();
            reject(abortedError());
        };

        const timer = globalThis.setTimeout(finish, ms);
        signal?.addEventListener("abort", handleAbort, {once: true});
    },
);

const waitForBackoff = (
    ms: number,
    signal: AbortSignal | undefined,
    sleep: ((delayMs: number, sleepSignal?: AbortSignal) => Promise<void>) | undefined,
): Promise<void> => {
    throwIfAborted(signal);
    const sleeper = sleep ?? abortableSleep;
    if (!signal) return sleeper(ms, undefined);

    return new Promise((resolve, reject) => {
        let settled = false;
        const settle = (callback: () => void) => {
            if (settled) return;
            settled = true;
            signal.removeEventListener("abort", handleAbort);
            callback();
        };
        const handleAbort = () => settle(() => reject(abortedError()));
        signal.addEventListener("abort", handleAbort, {once: true});

        try {
            void sleeper(ms, signal).then(
                () => settle(resolve),
                (cause: unknown) => settle(() => reject(cause)),
            );
        } catch (cause) {
            settle(() => reject(cause));
        }
    });
};

const retryAfterHeader = (headers: unknown): unknown => {
    if (!headers || typeof headers !== "object") return null;
    const candidate = headers as {
        get?: (name: string) => unknown;
        [key: string]: unknown;
    };
    if (typeof candidate.get === "function") return candidate.get("retry-after");
    return candidate["retry-after"];
};

const normalizeApiError = (cause: unknown, signal?: AbortSignal): ApiError => {
    if (signal?.aborted) return abortedError();
    if (cause instanceof ApiError) return cause;
    if (axios.isCancel(cause)) return abortedError();
    if (axios.isAxiosError(cause)) {
        if (cause.code === "ERR_CANCELED") return abortedError();
        if (cause.code === "ECONNABORTED" || cause.code === "ETIMEDOUT") {
            return new ApiError("timeout", "Request timed out");
        }
        if (cause.response) {
            const rawStatus: unknown = cause.response.status;
            const status = typeof rawStatus === "number" && Number.isInteger(rawStatus)
                ? rawStatus
                : null;
            const message = status === null ? "API request failed" : `API returned HTTP ${status}`;
            return new ApiError(
                "http",
                message,
                status,
                parseRetryAfterMs(retryAfterHeader(cause.response.headers)),
            );
        }
        if (cause.code && AXIOS_CLIENT_ERROR_CODES.has(cause.code)) {
            return new ApiError("client", "API request could not be prepared");
        }
        if (
            (cause.code && AXIOS_NETWORK_ERROR_CODES.has(cause.code))
            || (typeof cause.request === "object" && cause.request !== null)
        ) {
            return new ApiError("network", "Network request failed");
        }
        return new ApiError("client", "API request could not be prepared");
    }
    return new ApiError("client", "API request could not be prepared");
};

const resolveMethod = (method: RequestOptions["method"]): "GET" | "POST" | "DELETE" => {
    const resolved = method ?? "GET";
    if (resolved !== "GET" && resolved !== "POST" && resolved !== "DELETE") {
        throw new ApiError("client", "Invalid API request options");
    }
    return resolved;
};

const resolveAttempts = (value: number | undefined, method: RequestOptions["method"]): number => {
    if (value === undefined) return method === "GET" ? 3 : 1;
    if (!Number.isFinite(value) || !Number.isInteger(value) || value < 1) {
        throw new ApiError("client", "Invalid API request options");
    }
    return Math.min(value, MAX_ATTEMPTS);
};

const resolveTimeout = (value: number | undefined): number => {
    if (value === undefined) return DEFAULT_TIMEOUT_MS;
    if (!Number.isFinite(value) || !Number.isInteger(value) || value < 1) {
        throw new ApiError("client", "Invalid API request options");
    }
    return Math.min(value, MAX_TIMEOUT_MS);
};

const boundedJitter = (jitter: (() => number) | undefined): number => {
    if (!jitter) return Math.random();
    try {
        const value = jitter();
        if (!Number.isFinite(value)) return 0.5;
        return Math.min(1, Math.max(0, value));
    } catch {
        return 0.5;
    }
};

const retryDelay = (
    error: ApiError,
    attempt: number,
    jitter: (() => number) | undefined,
): number => {
    if (error.retryAfterMs !== null) return boundedDelay(error.retryAfterMs);
    const exponential = Math.min(
        MAX_RETRY_DELAY_MS,
        BASE_RETRY_DELAY_MS * (2 ** (attempt - 1)),
    );
    return boundedDelay(exponential * (0.5 + boundedJitter(jitter)));
};

export async function requestJson<T>(
    path: string,
    schema: ZodType<T>,
    options: RequestOptions = {},
): Promise<T> {
    throwIfAborted(options.signal);
    const canonicalPath = canonicalApiPath(path);
    const method = resolveMethod(options.method);
    const attempts = resolveAttempts(options.attempts, method);
    const timeout = resolveTimeout(options.timeoutMs);
    const url = env.apiBaseUrl ? `${env.apiBaseUrl}${canonicalPath}` : canonicalPath;

    for (let attempt = 1; attempt <= attempts; attempt += 1) {
        try {
            throwIfAborted(options.signal);
            const response = await axios.request<string>({
                method,
                url,
                signal: options.signal,
                timeout,
                params: options.params,
                data: options.body,
                responseType: "text",
                transformResponse: [(value: string) => value],
            });
            throwIfAborted(options.signal);

            let decoded: unknown;
            try {
                decoded = JSON.parse(response.data);
            } catch {
                throw new ApiError("invalid_json", "API returned invalid JSON");
            }

            let parsed: ReturnType<ZodType<T>["safeParse"]>;
            try {
                parsed = schema.safeParse(decoded);
            } catch {
                throw new ApiError("schema", "API response did not match its contract");
            }
            if (!parsed.success) {
                throw new ApiError("schema", "API response did not match its contract");
            }
            return parsed.data;
        } catch (cause) {
            const error = normalizeApiError(cause, options.signal);
            if (attempt >= attempts || !shouldRetryApiError(error)) throw error;

            try {
                await waitForBackoff(
                    retryDelay(error, attempt, options.jitter),
                    options.signal,
                    options.sleep,
                );
            } catch (sleepCause) {
                throw normalizeApiError(sleepCause, options.signal);
            }
            throwIfAborted(options.signal);
        }
    }

    throw new ApiError("network", "Network request failed");
}

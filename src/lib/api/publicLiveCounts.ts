import {z} from "zod";
import {ApiError} from "./client";
import {explicitIsoDateTimeSchema, localIsoDateTimeSchema} from "./schemas";

const PUBLIC_COUNTS_ENDPOINT = "https://goboardapi.azurewebsites.net/api/FacilityCount/GetCountsByAccount";
// This identifier is published by UW–Madison's public Live Building Usage widget.
const PUBLIC_WIDGET_ACCOUNT_ID = "7938fc89-a15c-492d-9566-12c961bc1f27";
export const PUBLIC_LIVE_COUNTS_TIMEOUT_MS = 4_000;

const safePositiveIntegerSchema = z.number().int().min(1).max(Number.MAX_SAFE_INTEGER);
const safeNonnegativeIntegerSchema = z.number().int().min(0).max(Number.MAX_SAFE_INTEGER);
const publicTimestampSchema = z.union([
    explicitIsoDateTimeSchema,
    localIsoDateTimeSchema,
]).nullable();

const publicObservationSchema = z.object({
    FacilityId: safePositiveIntegerSchema,
    LocationId: safePositiveIntegerSchema,
    IsClosed: z.boolean(),
    LastCount: safeNonnegativeIntegerSchema.nullable(),
    LastUpdatedDateAndTime: publicTimestampSchema,
}).passthrough();

const publicLiveCountsSchema = z.array(publicObservationSchema).min(1).superRefine(
    (rows, context) => {
        const seen = new Set<string>();
        rows.forEach((row, index) => {
            const identity = `${row.FacilityId}:${row.LocationId}`;
            if (seen.has(identity)) {
                context.addIssue({
                    code: "custom",
                    message: "official live-count identities must be unique",
                    path: [index],
                });
            }
            seen.add(identity);
        });
    },
);

export interface PublicLiveCountRow {
    FacilityId: number;
    LocationId: number;
    IsClosed: boolean;
    LastCount: number | null;
    LastUpdatedDateAndTime: string | null;
    FetchedAt: string;
}

export interface PublicLiveCountsOptions {
    signal?: AbortSignal;
    timeoutMs?: number;
}

const callerAbortedError = (): ApiError => new ApiError(
    "aborted",
    "Official live-count request aborted",
);

const resolveTimeout = (value: number | undefined): number => {
    if (value === undefined) return PUBLIC_LIVE_COUNTS_TIMEOUT_MS;
    if (!Number.isFinite(value) || !Number.isInteger(value) || value < 1) {
        throw new ApiError("client", "Invalid official live-count request options");
    }
    return Math.min(value, PUBLIC_LIVE_COUNTS_TIMEOUT_MS);
};

const normalizePublicError = (
    cause: unknown,
    signal: AbortSignal | undefined,
    timedOut: boolean,
): ApiError => {
    if (signal?.aborted) return callerAbortedError();
    if (timedOut) return new ApiError("timeout", "Official live-count request timed out");
    if (cause instanceof ApiError) return cause;
    return new ApiError("network", "Official live-count request failed");
};

export const fetchPublicLiveCounts = async (
    options: PublicLiveCountsOptions = {},
): Promise<PublicLiveCountRow[]> => {
    if (options.signal?.aborted) throw callerAbortedError();

    const timeoutMs = resolveTimeout(options.timeoutMs);
    const requestController = new AbortController();
    let timedOut = false;
    const handleCallerAbort = () => requestController.abort();
    const timeout = globalThis.setTimeout(() => {
        timedOut = true;
        requestController.abort();
    }, timeoutMs);
    options.signal?.addEventListener("abort", handleCallerAbort, {once: true});

    try {
        const url = new URL(PUBLIC_COUNTS_ENDPOINT);
        url.searchParams.set("AccountAPIKey", PUBLIC_WIDGET_ACCOUNT_ID);
        const response = await globalThis.fetch(url, {
            method: "GET",
            signal: requestController.signal,
            cache: "no-store",
            credentials: "omit",
            headers: {Accept: "application/json"},
        });
        if (!response.ok) {
            throw new ApiError(
                "http",
                `Official live counts returned HTTP ${response.status}`,
                response.status,
            );
        }

        const responseText = await response.text();
        const receivedAt = new Date(Date.now()).toISOString();
        let decoded: unknown;
        try {
            decoded = JSON.parse(responseText);
        } catch {
            throw new ApiError("invalid_json", "Official live counts returned invalid JSON");
        }

        const parsed = publicLiveCountsSchema.safeParse(decoded);
        if (!parsed.success) {
            throw new ApiError("schema", "Official live counts did not match the expected format");
        }
        if (options.signal?.aborted) throw callerAbortedError();

        return parsed.data.map((row) => ({
            FacilityId: row.FacilityId,
            LocationId: row.LocationId,
            IsClosed: row.IsClosed,
            LastCount: row.LastCount,
            LastUpdatedDateAndTime: (
                row.LastUpdatedDateAndTime !== null
                && explicitIsoDateTimeSchema.safeParse(row.LastUpdatedDateAndTime).success
            ) ? row.LastUpdatedDateAndTime : null,
            FetchedAt: receivedAt,
        }));
    } catch (cause) {
        throw normalizePublicError(cause, options.signal, timedOut);
    } finally {
        globalThis.clearTimeout(timeout);
        options.signal?.removeEventListener("abort", handleCallerAbort);
    }
};

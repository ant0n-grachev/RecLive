interface RetryOptions<T> {
    attempts: number;
    initialDelayMs?: number;
    backoffMultiplier?: number;
    signal?: AbortSignal;
    shouldRetryError?: (error: unknown, attempt: number) => boolean;
    shouldRetryResult?: (result: T, attempt: number) => boolean;
}

const wait = (ms: number, signal?: AbortSignal): Promise<void> => {
    if (ms <= 0) return Promise.resolve();

    return new Promise((resolve, reject) => {
        const timeoutId = globalThis.setTimeout(() => {
            signal?.removeEventListener("abort", handleAbort);
            resolve();
        }, ms);

        const handleAbort = () => {
            globalThis.clearTimeout(timeoutId);
            signal?.removeEventListener("abort", handleAbort);
            reject(signal?.reason ?? new Error("Aborted"));
        };

        if (signal?.aborted) {
            handleAbort();
            return;
        }

        signal?.addEventListener("abort", handleAbort, {once: true});
    });
};

export async function retryAsync<T>(
    operation: () => Promise<T>,
    {
        attempts,
        initialDelayMs = 0,
        backoffMultiplier = 1,
        signal,
        shouldRetryError,
        shouldRetryResult,
    }: RetryOptions<T>
): Promise<T> {
    const maxAttempts = Math.max(1, Math.floor(attempts));
    let lastError: unknown;
    let delayMs = initialDelayMs;

    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
        if (signal?.aborted) {
            throw signal.reason ?? new Error("Aborted");
        }

        try {
            const result = await operation();
            const wantsRetry = attempt < maxAttempts && (shouldRetryResult?.(result, attempt) ?? false);
            if (!wantsRetry) {
                return result;
            }
        } catch (error) {
            lastError = error;
            if (signal?.aborted) {
                throw error;
            }

            const canRetry = attempt < maxAttempts && (shouldRetryError?.(error, attempt) ?? true);
            if (!canRetry) {
                throw error;
            }
        }

        await wait(delayMs, signal);
        delayMs = Math.max(delayMs, 0) * Math.max(backoffMultiplier, 1);
    }

    throw lastError ?? new Error("Retry operation failed");
}

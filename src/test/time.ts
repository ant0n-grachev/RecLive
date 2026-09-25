import {vi} from "vitest";

export function freezeTime(value = "2026-08-31T12:00:00.000Z"): void {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(value));
}

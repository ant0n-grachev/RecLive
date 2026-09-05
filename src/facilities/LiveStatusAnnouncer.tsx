import type {CSSProperties} from "react";

export type LiveStatus = "idle" | "refreshing" | "updated" | "refresh-error" | "alert-created";

const messages: Record<Exclude<LiveStatus, "idle">, string> = {
    refreshing: "Refreshing live occupancy.",
    updated: "Live occupancy updated.",
    "refresh-error": "Live occupancy could not be refreshed. Showing the last available data.",
    "alert-created": "Occupancy alert created.",
};

const visuallyHidden: CSSProperties = {
    position: "absolute",
    width: 1,
    height: 1,
    padding: 0,
    margin: -1,
    overflow: "hidden",
    clip: "rect(0 0 0 0)",
    whiteSpace: "nowrap",
    border: 0,
};

export function LiveStatusAnnouncer({status}: {status: LiveStatus}) {
    if (status === "idle") return null;

    const assertive = status === "refresh-error";
    return (
        <div
            role={assertive ? "alert" : "status"}
            aria-live={assertive ? "assertive" : "polite"}
            style={visuallyHidden}
        >
            {messages[status]}
        </div>
    );
}

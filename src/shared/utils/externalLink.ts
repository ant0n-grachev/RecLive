export const EXTERNAL_LINK_REL = "external noopener noreferrer";

export const openExternalInBrowser = (url: string): void => {
    if (typeof window === "undefined") return;

    const popup = window.open(url, "_blank", "noopener,noreferrer");
    if (popup) {
        try {
            popup.opener = null;
        } catch {
            // Ignore cross-origin assignment issues.
        }
        return;
    }

    window.location.assign(url);
};

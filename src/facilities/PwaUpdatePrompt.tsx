export interface PwaUpdatePromptProps {
    needRefresh: boolean;
    offlineReady: boolean;
    updateServiceWorker(reloadPage?: boolean): Promise<void>;
    onDismiss(): void;
}

export function PwaUpdatePrompt({
    needRefresh,
    offlineReady,
    updateServiceWorker,
    onDismiss,
}: PwaUpdatePromptProps) {
    if (!needRefresh && !offlineReady) return null;

    const message = needRefresh
        ? "A new version of RecLive is ready."
        : "RecLive is ready to use offline.";

    return (
        <div role="status" aria-live="polite">
            <p>{message}</p>
            {needRefresh && (
                <button
                    type="button"
                    onClick={async () => {
                        await updateServiceWorker(true);
                        onDismiss();
                    }}
                >
                    Update now
                </button>
            )}
            <button type="button" onClick={onDismiss}>Dismiss</button>
        </div>
    );
}

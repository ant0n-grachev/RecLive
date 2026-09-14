export interface PwaUpdatePromptProps {
    needRefresh: boolean;
    offlineReady: boolean;
    updateServiceWorker(reloadPage?: boolean): Promise<void>;
    onDismiss(): void;
}

export function PwaUpdatePrompt({
    needRefresh,
    updateServiceWorker,
    onDismiss,
}: PwaUpdatePromptProps) {
    if (!needRefresh) return null;

    return (
        <div role="status" aria-live="polite">
            <p>A new version of RecLive is ready.</p>
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

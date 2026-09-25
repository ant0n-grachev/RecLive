import {useEffect, useState} from "react";

export const useOnlineStatus = (): boolean => {
    const [isOffline, setIsOffline] = useState<boolean>(() => {
        if (typeof window === "undefined") return false;
        return !window.navigator.onLine;
    });

    useEffect(() => {
        if (typeof window === "undefined") return;

        const syncOnlineStatus = () => {
            setIsOffline(!window.navigator.onLine);
        };

        syncOnlineStatus();
        window.addEventListener("online", syncOnlineStatus);
        window.addEventListener("offline", syncOnlineStatus);
        return () => {
            window.removeEventListener("online", syncOnlineStatus);
            window.removeEventListener("offline", syncOnlineStatus);
        };
    }, []);

    return isOffline;
};

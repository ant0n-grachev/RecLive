import {useEffect, useState} from "react";

export interface StandalonePwaState {
    isStandalonePwa: boolean;
    isTouchCapable: boolean;
}

export const useStandalonePwa = (): StandalonePwaState => {
    const [isStandalonePwa, setIsStandalonePwa] = useState(false);
    const [isTouchCapable] = useState(
        () => typeof window !== "undefined" && ("ontouchstart" in window || window.navigator.maxTouchPoints > 0)
    );

    useEffect(() => {
        if (typeof window === "undefined") return;

        const mediaQueries = [
            window.matchMedia("(display-mode: standalone)"),
            window.matchMedia("(display-mode: fullscreen)"),
            window.matchMedia("(display-mode: minimal-ui)"),
        ];

        const updateStandaloneState = () => {
            const iosStandalone = (
                window.navigator as Navigator & {standalone?: boolean}
            ).standalone === true;
            const standalone = mediaQueries.some((mediaQuery) => mediaQuery.matches) || iosStandalone;
            setIsStandalonePwa(standalone);
        };

        updateStandaloneState();

        if (typeof mediaQueries[0].addEventListener === "function") {
            mediaQueries.forEach((mediaQuery) => mediaQuery.addEventListener("change", updateStandaloneState));
            return () => {
                mediaQueries.forEach((mediaQuery) => mediaQuery.removeEventListener("change", updateStandaloneState));
            };
        }

        mediaQueries.forEach((mediaQuery) => mediaQuery.addListener(updateStandaloneState));
        return () => {
            mediaQueries.forEach((mediaQuery) => mediaQuery.removeListener(updateStandaloneState));
        };
    }, []);

    return {isStandalonePwa, isTouchCapable};
};

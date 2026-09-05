import {createAppTheme} from "./theme";

describe("app theme occupancy colors", () => {
    it("keeps the established light colors and uses readable semantic colors on dark surfaces", () => {
        const light = createAppTheme("light");
        const dark = createAppTheme("dark");

        expect([
            light.palette.success.main,
            light.palette.warning.main,
            light.palette.error.main,
        ]).toEqual(["#2e7d32", "#ca8a04", "#d32f2f"]);
        expect([
            dark.palette.success.main,
            dark.palette.warning.main,
            dark.palette.error.main,
        ]).toEqual(["#66bb6a", "#facc15", "#ef5350"]);
    });

    it("uses dark foregrounds on the brighter dark-theme status colors", () => {
        const palette = createAppTheme("dark").palette;
        expect(palette.success.contrastText).toBe("#0f172a");
        expect(palette.warning.contrastText).toBe("#111827");
        expect(palette.error.contrastText).toBe("#0f172a");
    });
});

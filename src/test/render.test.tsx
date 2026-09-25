import {screen} from "@testing-library/react";
import {describe, expect, it} from "vitest";
import {renderWithApp} from "./render";

describe("renderWithApp", () => {
    it("renders with a memory route and clears persisted browser state after each test", () => {
        window.localStorage.setItem("reclive:test", "set");
        renderWithApp(<button type="button">Foundation ready</button>, {route: "/nick"});
        expect(screen.getByRole("button", {name: "Foundation ready"})).toBeVisible();
    });

    it("starts the next test with an empty localStorage", () => {
        expect(window.localStorage.getItem("reclive:test")).toBeNull();
    });
});

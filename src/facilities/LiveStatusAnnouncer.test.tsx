import {render, screen} from "@testing-library/react";
import {expect, it} from "vitest";
import {LiveStatusAnnouncer} from "./LiveStatusAnnouncer";

it("announces refresh completion after an in-progress state", () => {
    const view = render(<LiveStatusAnnouncer status="refreshing" />);
    expect(screen.getByRole("status")).toHaveTextContent("Refreshing live occupancy");
    view.rerender(<LiveStatusAnnouncer status="updated" />);
    expect(screen.getByRole("status")).toHaveTextContent("Live occupancy updated");
});

it("uses assertive announcement for a refresh failure", () => {
    render(<LiveStatusAnnouncer status="refresh-error" />);
    expect(screen.getByRole("alert")).toHaveTextContent("Live occupancy could not be refreshed");
});

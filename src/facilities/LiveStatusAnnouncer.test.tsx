import {render, screen} from "@testing-library/react";
import {expect, it} from "vitest";
import {LiveStatusAnnouncer} from "./LiveStatusAnnouncer";

it("announces refresh completion after an in-progress state", () => {
    const view = render(<LiveStatusAnnouncer status="refreshing" />);
    expect(screen.getByRole("status")).toHaveTextContent("Refreshing live occupancy");
    view.rerender(<LiveStatusAnnouncer status="updated" />);
    expect(screen.getByRole("status")).toHaveTextContent("Live occupancy updated");
});

it("reports a requested refresh failure briefly without a stale-data announcement", () => {
    render(<LiveStatusAnnouncer status="refresh-error" />);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Couldn't refresh. Try again.");
    expect(screen.getByRole("status")).not.toHaveTextContent(/last available|stale/i);
});

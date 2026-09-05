import {render, screen} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {expect, it, vi} from "vitest";
import {PwaUpdatePrompt} from "./PwaUpdatePrompt";

it("announces and applies an available update", async () => {
    const user = userEvent.setup();
    const updateServiceWorker = vi.fn(async () => undefined);
    const onDismiss = vi.fn();

    render(
        <PwaUpdatePrompt
            needRefresh
            offlineReady={false}
            updateServiceWorker={updateServiceWorker}
            onDismiss={onDismiss}
        />
    );

    expect(screen.getByRole("status")).toHaveTextContent("A new version of RecLive is ready");
    await user.click(screen.getByRole("button", {name: "Update now"}));
    expect(updateServiceWorker).toHaveBeenCalledWith(true);
    expect(onDismiss).toHaveBeenCalledOnce();
});

it("does not render when neither worker state is active", () => {
    const {container} = render(
        <PwaUpdatePrompt
            needRefresh={false}
            offlineReady={false}
            updateServiceWorker={async () => undefined}
            onDismiss={() => undefined}
        />
    );

    expect(container).toBeEmptyDOMElement();
});

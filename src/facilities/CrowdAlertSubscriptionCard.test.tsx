import {act, fireEvent, render, screen, waitFor, within} from "@testing-library/react";
import type {PushRule} from "../lib/api/pushNotifications";
import type {OccupancySummary} from "../shared/occupancy/computeOccupancySummary";
import {
    default as CrowdAlertSubscriptionCard,
    resolveInitialSectionKey,
    type AlertSectionOption,
} from "./CrowdAlertSubscriptionCard";

const pushApi = vi.hoisted(() => ({
    cancelAllPushRules: vi.fn(),
    cancelPushRule: vi.fn(),
    ensurePushSubscription: vi.fn(),
    getExistingPushSubscription: vi.fn(),
    getPushAvailability: vi.fn(),
    isWebPushSupported: vi.fn(),
    listPushRules: vi.fn(),
    subscribePushRule: vi.fn(),
}));

vi.mock("../lib/api/pushNotifications", () => pushApi);

const STORAGE_KEY = "reclive:crowd-alert-subscriptions";

const validSubscription: PushSubscriptionJSON = {
    endpoint: "https://push.reclive-notify.net/subscription-a",
    expirationTime: null,
    keys: {p256dh: "p256dh-fixture", auth: "auth-fixture"},
};

const secondSubscription: PushSubscriptionJSON = {
    endpoint: "https://push.reclive-notify.net/subscription-b",
    expirationTime: null,
    keys: {p256dh: "p256dh-fixture-b", auth: "auth-fixture-b"},
};

const managedRule: PushRule = {
    id: 7,
    facilityId: 1186,
    sectionKey: "overall",
    threshold: 40,
    createdAt: "2026-09-01T12:00:00Z",
    expiresAt: "2026-09-02T12:00:00Z",
    status: "pending",
};

const crossFacilityRule: PushRule = {
    id: 9,
    facilityId: 1656,
    sectionKey: "legacy cardio",
    threshold: 25,
    createdAt: "2026-09-01T08:00:00-04:00",
    expiresAt: "2026-09-03T08:00:00-04:00",
    status: "pending",
};

const browserSubscription = (
    json: PushSubscriptionJSON = validSubscription
): PushSubscription => ({
    toJSON: () => json,
}) as unknown as PushSubscription;

const deferred = <T,>() => {
    let resolve!: (value: T | PromiseLike<T>) => void;
    let reject!: (reason?: unknown) => void;
    const promise = new Promise<T>((resolvePromise, rejectPromise) => {
        resolve = resolvePromise;
        reject = rejectPromise;
    });
    return {promise, reject, resolve};
};

const setNotificationPermission = (permission: NotificationPermission): void => {
    Object.defineProperty(globalThis, "Notification", {
        configurable: true,
        value: {
            permission,
            requestPermission: vi.fn().mockResolvedValue(permission),
        },
    });
};

const summary = (
    status: OccupancySummary["status"],
    percent: number | null,
    overrides: Partial<OccupancySummary> = {}
): OccupancySummary => ({
    count: percent === null ? null : 25,
    observedCapacity: percent === null ? 0 : 100,
    expectedOpenCapacity: 100,
    coverage: status === "live" ? 1 : status === "partial" ? 0.6 : 0,
    percent,
    observedLocations: percent === null ? 0 : 1,
    expectedLocations: 1,
    latestFetchedAt: percent === null ? null : "2026-08-31T12:00:00Z",
    oldestFetchedAt: percent === null ? null : "2026-08-31T12:00:00Z",
    status,
    ...overrides,
});

const option = (
    key: string,
    occupancySummary: OccupancySummary
): AlertSectionOption => ({key, label: key, summary: occupancySummary});

const defaultSections = (): AlertSectionOption[] => [
    {key: "overall", label: "Entire Facility", summary: summary("live", 60)},
    {key: "fitness floors", label: "Fitness Floors", summary: summary("live", 45)},
];

const renderOpenCard = (
    overrides: Partial<React.ComponentProps<typeof CrowdAlertSubscriptionCard>> = {}
) => {
    const onClose = overrides.onClose ?? vi.fn();
    const view = render(
        <CrowdAlertSubscriptionCard
            facility={1186}
            isOpen
            onClose={onClose}
            sections={defaultSections()}
            {...overrides}
        />
    );
    return {onClose, ...view};
};

beforeEach(() => {
    setNotificationPermission("granted");
    pushApi.isWebPushSupported.mockReturnValue(true);
    pushApi.getExistingPushSubscription.mockResolvedValue(null);
    pushApi.getPushAvailability.mockResolvedValue({
        apiAvailable: true,
        dbAvailable: true,
        alertsAvailable: true,
        reason: null,
    });
    pushApi.listPushRules.mockResolvedValue([]);
    pushApi.cancelPushRule.mockResolvedValue(undefined);
    pushApi.cancelAllPushRules.mockResolvedValue(0);
    pushApi.ensurePushSubscription.mockResolvedValue(browserSubscription());
    pushApi.subscribePushRule.mockResolvedValue({created: true, rule: managedRule});
});

describe("resolveInitialSectionKey", () => {
    it("clears an insufficient saved selection and selects the first live option", () => {
        window.localStorage.setItem("reclive:crowd-alert-subscriptions", JSON.stringify({
            "1186": {sectionKey: "overall", threshold: 20},
        }));

        expect(resolveInitialSectionKey(1186, [
            option("overall", summary("insufficient", null)),
            option("weights", summary("live", 25)),
        ])).toBe("weights");
    });

    it("replaces a saved partial selection with the first live option", () => {
        window.localStorage.setItem("reclive:crowd-alert-subscriptions", JSON.stringify({
            "1186": {sectionKey: "courts", threshold: 20},
        }));

        expect(resolveInitialSectionKey(1186, [
            option("overall", summary("live", 25)),
            option("courts", summary("partial", 40)),
        ])).toBe("overall");
    });

    it("returns no selection when no option has a usable percentage", () => {
        expect(resolveInitialSectionKey(1186, [
            option("overall", summary("unknown", null)),
            option("weights", summary("insufficient", null)),
            option("closed", summary("closed", null)),
        ])).toBe("");
    });
});

describe("CrowdAlertSubscriptionCard", () => {
    it("keeps an invalid selection cleared after its summary becomes usable again", () => {
        const liveOverall = option("overall", summary("live", 25));
        const unavailableOverall = option("overall", summary("insufficient", null));
        const liveWeights = option("weights", summary("live", 40));
        const props = {
            facility: 1186 as const,
            isOpen: false,
            onClose: vi.fn(),
        };
        const {rerender} = render(
            <CrowdAlertSubscriptionCard {...props} sections={[liveOverall, liveWeights]} />
        );

        expect(screen.getByRole("combobox", {name: "Gym area"})).toHaveTextContent("overall");

        rerender(
            <CrowdAlertSubscriptionCard {...props} sections={[unavailableOverall, liveWeights]} />
        );
        expect(screen.getByRole("combobox", {name: "Gym area"})).toHaveTextContent("weights");

        rerender(
            <CrowdAlertSubscriptionCard {...props} sections={[liveOverall, liveWeights]} />
        );
        expect(screen.getByRole("combobox", {name: "Gym area"})).toHaveTextContent("weights");
    });

    it("caps an over-capacity alert threshold at 100 percent", () => {
        render(
            <CrowdAlertSubscriptionCard
                facility={1186}
                isOpen={false}
                onClose={vi.fn()}
                sections={[option("overall", summary("live", 150, {
                    count: 150,
                    observedCapacity: 100,
                }))]}
            />
        );

        const thresholdInput = screen.getByLabelText("Alert threshold (%)");
        const submitButton = screen.getByRole("button", {name: "Set alert"});
        expect(thresholdInput).toHaveAttribute("max", "100");
        expect(submitButton).toBeEnabled();

        fireEvent.change(thresholdInput, {target: {value: "101"}});
        expect(screen.getByText("Enter a number between 1 and 100.")).toBeInTheDocument();
        expect(submitButton).toBeDisabled();

        fireEvent.change(thresholdInput, {target: {value: "100"}});
        expect(submitButton).toBeEnabled();
    });
});

describe("CrowdAlertSubscriptionCard server-backed alert management", () => {
    it("loads an existing browser subscription on open and lists every server rule with Chicago expiry copy", async () => {
        pushApi.getExistingPushSubscription.mockResolvedValue(browserSubscription());
        pushApi.listPushRules.mockResolvedValue([managedRule, crossFacilityRule]);

        renderOpenCard();

        const list = await screen.findByRole("list", {name: "Active alerts"});
        expect(within(list).getAllByRole("listitem")).toHaveLength(2);
        expect(pushApi.getExistingPushSubscription).toHaveBeenCalledTimes(1);
        expect(pushApi.listPushRules).toHaveBeenCalledTimes(1);
        expect(pushApi.listPushRules).toHaveBeenCalledWith(validSubscription);

        const expiry = within(list).getByText("Expires Sep 2, 2026 at 7:00 AM CDT");
        expect(expiry.tagName).toBe("TIME");
        expect(expiry).toHaveAttribute("datetime", managedRule.expiresAt);
        expect(within(list).getByRole("listitem", {
            name: "Alert for Nick Entire Facility at 40%, expires Sep 2, 2026 at 7:00 AM CDT",
        })).toBeVisible();
    });

    it("shows a truthful empty state without listing when no browser subscription exists and can create an alert", async () => {
        renderOpenCard();

        expect(await screen.findByText("No active browser subscription was found.")).toBeVisible();
        expect(pushApi.listPushRules).not.toHaveBeenCalled();

        const setAlert = screen.getByRole("button", {name: "Set alert"});
        await waitFor(() => expect(setAlert).toBeEnabled());
        fireEvent.click(setAlert);

        await waitFor(() => expect(pushApi.subscribePushRule).toHaveBeenCalledTimes(1));
        expect(pushApi.subscribePushRule).toHaveBeenCalledWith({
            subscription: validSubscription,
            facilityId: 1186,
            sectionKey: "overall",
            threshold: 40,
        });
        await waitFor(() => expect(screen.getByText("Alert set successfully.")).toBeVisible());
        expect(screen.getByRole("status")).toHaveTextContent("Occupancy alert created.");
        expect(screen.getByRole("listitem", {name: /Alert for Nick Entire Facility at 40%/i})).toBeVisible();
    });

    it("still lists and cancels an existing subscription when notification permission is denied", async () => {
        setNotificationPermission("denied");
        pushApi.getExistingPushSubscription.mockResolvedValue(browserSubscription());
        pushApi.listPushRules.mockResolvedValue([managedRule]);

        renderOpenCard();

        const cancel = await screen.findByRole("button", {
            name: "Cancel alert for Nick Entire Facility at 40%",
        });
        expect(screen.getByRole("button", {name: "Set alert"})).toBeDisabled();
        expect(screen.getByText("Notifications are blocked in browser settings.")).toBeVisible();

        fireEvent.click(cancel);
        await waitFor(() => expect(pushApi.cancelPushRule).toHaveBeenCalledWith(7, validSubscription));
        await waitFor(() => expect(screen.queryByRole("listitem")).not.toBeInTheDocument());
    });

    it("still loads availability and manages existing rules when standalone gating blocks creation", async () => {
        pushApi.getExistingPushSubscription.mockResolvedValue(browserSubscription());
        pushApi.listPushRules.mockResolvedValue([managedRule]);

        renderOpenCard({requireStandalonePwaForAlerts: true});

        const cancel = await screen.findByRole("button", {
            name: "Cancel alert for Nick Entire Facility at 40%",
        });
        expect(pushApi.getPushAvailability).toHaveBeenCalledTimes(1);
        expect(screen.getByRole("button", {name: "Set alert"})).toBeDisabled();
        fireEvent.click(cancel);
        await waitFor(() => expect(pushApi.cancelPushRule).toHaveBeenCalledWith(7, validSubscription));
    });

    it("shows a fixed list failure and never turns it into an empty-list success", async () => {
        const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
        pushApi.getExistingPushSubscription.mockResolvedValue(browserSubscription());
        pushApi.listPushRules.mockRejectedValue(new Error("private-list-response"));

        renderOpenCard();

        expect(await screen.findByText("Could not load alerts right now.")).toBeVisible();
        expect(screen.queryByText("No active alerts for this browser.")).not.toBeInTheDocument();
        expect(screen.queryByText(/private-list-response/i)).not.toBeInTheDocument();
        expect(consoleError).not.toHaveBeenCalled();
        expect(screen.getByRole("region", {name: "Manage alerts"})).toHaveAccessibleDescription(
            "Could not load alerts right now."
        );
    });

    it("shows a distinct fixed existing-subscription lookup failure without listing or exposing exception text", async () => {
        const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
        pushApi.getExistingPushSubscription.mockRejectedValue(new Error("private-lookup-response"));

        renderOpenCard();

        expect(await screen.findByText("Could not access this browser's alerts right now.")).toBeVisible();
        expect(screen.queryByText("Could not load alerts right now.")).not.toBeInTheDocument();
        expect(screen.queryByText("No active alerts for this browser.")).not.toBeInTheDocument();
        expect(screen.queryByText("No active browser subscription was found.")).not.toBeInTheDocument();
        expect(screen.queryByText(/private-lookup-response/i)).not.toBeInTheDocument();
        expect(pushApi.listPushRules).not.toHaveBeenCalled();
        expect(consoleError).not.toHaveBeenCalled();
    });

    it("keeps loaded management data visible when a later open cannot refresh the list", async () => {
        pushApi.getExistingPushSubscription.mockResolvedValue(browserSubscription());
        pushApi.listPushRules
            .mockResolvedValueOnce([managedRule])
            .mockRejectedValueOnce(new Error("private-refresh-response"));
        const {rerender} = renderOpenCard();

        expect(await screen.findByRole("listitem", {name: /Alert for Nick Entire Facility at 40%/i})).toBeVisible();
        rerender(
            <CrowdAlertSubscriptionCard
                facility={1186}
                isOpen={false}
                onClose={vi.fn()}
                sections={defaultSections()}
            />
        );
        rerender(
            <CrowdAlertSubscriptionCard
                facility={1186}
                isOpen
                onClose={vi.fn()}
                sections={defaultSections()}
            />
        );

        expect(await screen.findByText("Could not load alerts right now.")).toBeVisible();
        expect(screen.getByRole("listitem", {name: /Alert for Nick Entire Facility at 40%/i})).toBeVisible();
    });

    it("recovers safe worker lookup failure by closing and reopening alerts", async () => {
        pushApi.getExistingPushSubscription
            .mockRejectedValueOnce(new Error("synthetic worker registration failure"))
            .mockResolvedValueOnce(browserSubscription());
        pushApi.listPushRules.mockResolvedValue([managedRule]);
        const {rerender} = renderOpenCard();
        expect(await screen.findByText("Could not access this browser's alerts right now.")).toBeVisible();
        expect(screen.queryByText(/synthetic worker registration failure/i)).not.toBeInTheDocument();
        rerender(<CrowdAlertSubscriptionCard facility={1186} isOpen={false} onClose={vi.fn()} sections={defaultSections()}/>);
        rerender(<CrowdAlertSubscriptionCard facility={1186} isOpen onClose={vi.fn()} sections={defaultSections()}/>);
        expect(await screen.findByRole("listitem", {name: /Alert for Nick Entire Facility at 40%/i})).toBeVisible();
        expect(screen.queryByText("Could not access this browser's alerts right now.")).not.toBeInTheDocument();
    });

    it("disables and handler-guards retained per-rule controls while owner lookup is loading", async () => {
        const refreshLookup = deferred<PushSubscription | null>();
        pushApi.getExistingPushSubscription
            .mockResolvedValueOnce(browserSubscription())
            .mockReturnValueOnce(refreshLookup.promise);
        pushApi.listPushRules.mockResolvedValue([managedRule]);
        const {rerender} = renderOpenCard();

        expect(await screen.findByRole("listitem", {name: /Alert for Nick Entire Facility at 40%/i})).toBeVisible();
        rerender(
            <CrowdAlertSubscriptionCard facility={1186} isOpen={false} onClose={vi.fn()} sections={defaultSections()} />
        );
        rerender(
            <CrowdAlertSubscriptionCard facility={1186} isOpen onClose={vi.fn()} sections={defaultSections()} />
        );
        await waitFor(() => expect(pushApi.getExistingPushSubscription).toHaveBeenCalledTimes(2));

        const cancel = screen.getByRole("button", {
            name: "Cancel alert for Nick Entire Facility at 40%",
        });
        expect(cancel).toBeDisabled();
        cancel.removeAttribute("disabled");
        fireEvent.click(cancel);
        expect(pushApi.cancelPushRule).not.toHaveBeenCalled();

        refreshLookup.resolve(browserSubscription());
        await waitFor(() => expect(cancel).toBeEnabled());
    });

    it("disables and handler-guards retained cancel-all while the refreshed rule list is loading", async () => {
        const refreshList = deferred<PushRule[]>();
        pushApi.getExistingPushSubscription.mockResolvedValue(browserSubscription());
        pushApi.listPushRules
            .mockResolvedValueOnce([managedRule])
            .mockReturnValueOnce(refreshList.promise);
        const {rerender} = renderOpenCard();

        expect(await screen.findByRole("listitem", {name: /Alert for Nick Entire Facility at 40%/i})).toBeVisible();
        rerender(
            <CrowdAlertSubscriptionCard facility={1186} isOpen={false} onClose={vi.fn()} sections={defaultSections()} />
        );
        rerender(
            <CrowdAlertSubscriptionCard facility={1186} isOpen onClose={vi.fn()} sections={defaultSections()} />
        );
        await waitFor(() => expect(pushApi.listPushRules).toHaveBeenCalledTimes(2));

        const cancelAll = screen.getByRole("button", {name: "Cancel all alerts"});
        expect(cancelAll).toBeDisabled();
        cancelAll.removeAttribute("disabled");
        fireEvent.click(cancelAll);
        expect(pushApi.cancelAllPushRules).not.toHaveBeenCalled();

        refreshList.resolve([managedRule]);
        await waitFor(() => expect(cancelAll).toBeEnabled());
    });

    it("hides old-owner rules immediately and retains the changed subscription for creation when its list fails", async () => {
        const changedOwnerList = deferred<PushRule[]>();
        pushApi.getExistingPushSubscription
            .mockResolvedValueOnce(browserSubscription())
            .mockResolvedValueOnce(browserSubscription(secondSubscription));
        pushApi.listPushRules
            .mockResolvedValueOnce([managedRule])
            .mockReturnValueOnce(changedOwnerList.promise);
        const {rerender} = renderOpenCard();
        expect(await screen.findByRole("listitem", {name: /Alert for Nick Entire Facility at 40%/i})).toBeVisible();

        rerender(
            <CrowdAlertSubscriptionCard facility={1186} isOpen={false} onClose={vi.fn()} sections={defaultSections()} />
        );
        rerender(
            <CrowdAlertSubscriptionCard facility={1186} isOpen onClose={vi.fn()} sections={defaultSections()} />
        );

        await waitFor(() => expect(pushApi.listPushRules).toHaveBeenCalledWith(secondSubscription));
        expect(screen.queryByRole("listitem", {name: /Alert for Nick Entire Facility at 40%/i})).not.toBeInTheDocument();
        expect(screen.queryByRole("button", {name: /Cancel alert for Nick Entire Facility/i})).not.toBeInTheDocument();

        changedOwnerList.reject(new Error("private-new-owner-list-response"));
        expect(await screen.findByText("Could not load alerts right now.")).toBeVisible();
        expect(screen.queryByRole("listitem", {name: /Alert for Nick Entire Facility at 40%/i})).not.toBeInTheDocument();
        expect(screen.queryByText("No active alerts for this browser.")).not.toBeInTheDocument();
        fireEvent.click(screen.getByRole("button", {name: "Set alert"}));
        await waitFor(() => expect(pushApi.subscribePushRule).toHaveBeenCalledWith({
            subscription: secondSubscription,
            facilityId: 1186,
            sectionKey: "overall",
            threshold: 40,
        }));
        expect(pushApi.ensurePushSubscription).not.toHaveBeenCalled();
    });

    it("keeps an availability failure distinct while preserving successfully loaded management data", async () => {
        pushApi.getPushAvailability.mockRejectedValue(new Error("private-availability-response"));
        pushApi.getExistingPushSubscription.mockResolvedValue(browserSubscription());
        pushApi.listPushRules.mockResolvedValue([managedRule]);

        renderOpenCard();

        expect(await screen.findByRole("listitem", {name: /Alert for Nick Entire Facility at 40%/i})).toBeVisible();
        expect(screen.getByText("Alerts aren’t available right now. Try again.")).toBeVisible();
        expect(screen.queryByText("Could not load alerts right now.")).not.toBeInTheDocument();
        expect(screen.getByRole("button", {name: "Set alert"})).toBeDisabled();
    });

    it.each(["push_rules_db_unavailable", "push_vapid_unconfigured"] as const)(
        "uses the same concise unavailable message for %s",
        async (reason) => {
            pushApi.getPushAvailability.mockResolvedValue({
                apiAvailable: true,
                dbAvailable: reason !== "push_rules_db_unavailable",
                alertsAvailable: false,
                reason,
            });

            renderOpenCard();

            expect(await screen.findByText("Alerts aren’t available right now. Try again.")).toBeVisible();
            expect(screen.getByRole("button", {name: "Set alert"})).toBeDisabled();
        }
    );

    it("shows a fixed associated subscribe failure without erasing loaded rules or defaults", async () => {
        const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify({
            "1186": {sectionKey: "fitness floors", threshold: 20},
        }));
        pushApi.getExistingPushSubscription.mockResolvedValue(browserSubscription());
        pushApi.listPushRules.mockResolvedValue([managedRule]);
        pushApi.subscribePushRule.mockRejectedValue(new Error("private-subscribe-response"));

        renderOpenCard();
        expect(await screen.findByRole("listitem", {name: /Alert for Nick Entire Facility at 40%/i})).toBeVisible();
        fireEvent.click(screen.getByRole("button", {name: "Set alert"}));

        const error = await screen.findByText("Could not save this alert right now.");
        const button = screen.getByRole("button", {name: "Set alert"});
        expect(button).toHaveAttribute("aria-describedby", error.id);
        expect(screen.getByRole("listitem", {name: /Alert for Nick Entire Facility at 40%/i})).toBeVisible();
        expect(window.localStorage.getItem(STORAGE_KEY)).toBe(JSON.stringify({
            "1186": {sectionKey: "fitness floors", threshold: 20},
        }));
        expect(screen.queryByText(/private-subscribe-response/i)).not.toBeInTheDocument();
        expect(consoleError).not.toHaveBeenCalled();
    });

    it("replaces incomplete management data with the authoritative full list after subscribe succeeds", async () => {
        const subscribe = deferred<{created: boolean; rule: PushRule}>();
        const retryList = deferred<PushRule[]>();
        pushApi.getExistingPushSubscription.mockResolvedValue(browserSubscription());
        pushApi.listPushRules
            .mockRejectedValueOnce(new Error("private-initial-list-response"))
            .mockReturnValueOnce(retryList.promise);
        pushApi.subscribePushRule.mockReturnValue(subscribe.promise);
        renderOpenCard();

        expect(await screen.findByText("Could not load alerts right now.")).toBeVisible();
        fireEvent.click(screen.getByRole("button", {name: "Set alert"}));
        await waitFor(() => expect(pushApi.subscribePushRule).toHaveBeenCalledTimes(1));
        expect(pushApi.listPushRules).toHaveBeenCalledTimes(1);

        subscribe.resolve({created: true, rule: managedRule});
        await waitFor(() => expect(pushApi.listPushRules).toHaveBeenCalledTimes(2));
        expect(pushApi.listPushRules).toHaveBeenLastCalledWith(validSubscription);
        expect(screen.getByText("Could not load alerts right now.")).toBeVisible();

        retryList.resolve([managedRule, crossFacilityRule]);
        const list = await screen.findByRole("list", {name: "Active alerts"});
        await waitFor(() => expect(within(list).getAllByRole("listitem")).toHaveLength(2));
        expect(screen.queryByText("Could not load alerts right now.")).not.toBeInTheDocument();
        expect(screen.getByText("Alert set successfully.")).toBeVisible();
    });

    it("keeps incomplete-list status and safe returned data when the post-subscribe list retry fails", async () => {
        const retryList = deferred<PushRule[]>();
        pushApi.getExistingPushSubscription.mockResolvedValue(browserSubscription());
        pushApi.listPushRules
            .mockRejectedValueOnce(new Error("private-initial-list-response"))
            .mockReturnValueOnce(retryList.promise);
        pushApi.subscribePushRule.mockResolvedValue({created: true, rule: managedRule});
        renderOpenCard();

        expect(await screen.findByText("Could not load alerts right now.")).toBeVisible();
        fireEvent.click(screen.getByRole("button", {name: "Set alert"}));
        await waitFor(() => expect(pushApi.listPushRules).toHaveBeenCalledTimes(2));

        retryList.reject(new Error("private-retry-list-response"));
        await waitFor(() => expect(screen.getByRole("button", {name: "Set alert"})).toBeEnabled());
        expect(screen.getByText("Could not load alerts right now.")).toBeVisible();
        expect(screen.getByRole("listitem", {name: /Alert for Nick Entire Facility at 40%/i})).toBeVisible();
        expect(screen.queryByText("No active alerts for this browser.")).not.toBeInTheDocument();
        expect(screen.queryByText("Could not save this alert right now.")).not.toBeInTheDocument();
        expect(screen.queryByText(/private-retry-list-response/i)).not.toBeInTheDocument();
        expect(screen.getByText("Alert set successfully.")).toBeVisible();
    });

    it("treats created true as success, upserts the returned rule, and writes a form default only after success", async () => {
        const subscribe = deferred<{created: boolean; rule: PushRule}>();
        pushApi.subscribePushRule.mockReturnValue(subscribe.promise);
        renderOpenCard();
        expect(await screen.findByText("No active browser subscription was found.")).toBeVisible();

        fireEvent.click(screen.getByRole("button", {name: "Set alert"}));
        await waitFor(() => expect(pushApi.subscribePushRule).toHaveBeenCalledTimes(1));
        expect(window.localStorage.getItem(STORAGE_KEY)).toBeNull();

        subscribe.resolve({created: true, rule: managedRule});
        await waitFor(() => expect(screen.getByText("Alert set successfully.")).toBeVisible());
        expect(screen.getByRole("status")).toHaveTextContent("Occupancy alert created.");
        expect(screen.getByRole("listitem", {name: /Alert for Nick Entire Facility at 40%/i})).toBeVisible();
        expect(JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "null")).toEqual({
            "1186": {sectionKey: "overall", threshold: 40},
        });
    });

    it("treats created false as an idempotent success without a second subscribe", async () => {
        pushApi.subscribePushRule.mockResolvedValue({created: false, rule: managedRule});
        renderOpenCard();
        expect(await screen.findByText("No active browser subscription was found.")).toBeVisible();

        fireEvent.click(screen.getByRole("button", {name: "Set alert"}));

        await waitFor(() => expect(screen.getByText("This alert was already active.")).toBeVisible());
        expect(screen.getByRole("status")).toHaveTextContent("Occupancy alert created.");
        expect(pushApi.subscribePushRule).toHaveBeenCalledTimes(1);
        expect(pushApi.listPushRules).not.toHaveBeenCalled();
        expect(screen.getByRole("listitem", {name: /Alert for Nick Entire Facility at 40%/i})).toBeVisible();
        expect(JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "null")).toEqual({
            "1186": {sectionKey: "overall", threshold: 40},
        });
    });

    it("never auto-closes after a successful subscribe even after timers advance", async () => {
        const onClose = vi.fn();
        renderOpenCard({onClose});
        expect(await screen.findByText("No active browser subscription was found.")).toBeVisible();
        vi.useFakeTimers();

        fireEvent.click(screen.getByRole("button", {name: "Set alert"}));
        await act(async () => {
            for (let index = 0; index < 6; index += 1) await Promise.resolve();
        });
        act(() => vi.advanceTimersByTime(10_000));

        expect(onClose).not.toHaveBeenCalled();
    });

    it("renders and manages a cross-facility unknown-section rule with a safe fallback label", async () => {
        pushApi.getExistingPushSubscription.mockResolvedValue(browserSubscription());
        pushApi.listPushRules.mockResolvedValue([crossFacilityRule]);
        renderOpenCard();

        const item = await screen.findByRole("listitem", {
            name: "Alert for Bakke Area: legacy cardio at 25%, expires Sep 3, 2026 at 7:00 AM CDT",
        });
        expect(item).toHaveTextContent("Bakke — Area: legacy cardio — 25%");
        const cancel = within(item).getByRole("button", {
            name: "Cancel alert for Bakke Area: legacy cardio at 25%",
        });
        fireEvent.click(cancel);
        await waitFor(() => expect(pushApi.cancelPushRule).toHaveBeenCalledWith(9, validSubscription));
    });

    it("cancels one rule exactly once, removes only that row/default, and announces success", async () => {
        const cancellation = deferred<void>();
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify({
            "1186": {sectionKey: "overall", threshold: 40},
            "1656": {sectionKey: "fitness floors", threshold: 30},
        }));
        pushApi.getExistingPushSubscription.mockResolvedValue(browserSubscription());
        pushApi.listPushRules.mockResolvedValue([managedRule, crossFacilityRule]);
        pushApi.cancelPushRule.mockReturnValue(cancellation.promise);
        renderOpenCard();

        const cancel = await screen.findByRole("button", {
            name: "Cancel alert for Nick Entire Facility at 40%",
        });
        fireEvent.click(cancel);
        fireEvent.click(cancel);
        expect(pushApi.cancelPushRule).toHaveBeenCalledTimes(1);
        expect(screen.getByRole("button", {
            name: "Cancelling alert for Nick Entire Facility at 40%",
        })).toBeDisabled();

        cancellation.resolve();
        await waitFor(() => expect(screen.queryByRole("listitem", {
            name: /Alert for Nick Entire Facility at 40%/i,
        })).not.toBeInTheDocument());
        expect(screen.getByRole("listitem", {name: /Alert for Bakke Area: legacy cardio at 25%/i})).toBeVisible();
        expect(screen.getByText("Alert cancelled.")).toBeVisible();
        expect(screen.getByRole("status")).toHaveTextContent("Alert cancelled.");
        expect(JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "null")).toEqual({
            "1656": {sectionKey: "fitness floors", threshold: 30},
        });
    });

    it("retains the rule/default and associates a fixed safe error when cancel-one fails", async () => {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify({
            "1186": {sectionKey: "overall", threshold: 40},
        }));
        pushApi.getExistingPushSubscription.mockResolvedValue(browserSubscription());
        pushApi.listPushRules.mockResolvedValue([managedRule]);
        pushApi.cancelPushRule.mockRejectedValue(new Error("private-cancel-response"));
        renderOpenCard();

        fireEvent.click(await screen.findByRole("button", {
            name: "Cancel alert for Nick Entire Facility at 40%",
        }));

        const error = await screen.findByText("Could not cancel this alert right now.");
        const button = screen.getByRole("button", {
            name: "Cancel alert for Nick Entire Facility at 40%",
        });
        expect(button).toHaveAttribute("aria-describedby", error.id);
        expect(screen.getByRole("listitem", {name: /Alert for Nick Entire Facility at 40%/i})).toBeVisible();
        expect(JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "null")).toEqual({
            "1186": {sectionKey: "overall", threshold: 40},
        });
        expect(screen.queryByText(/private-cancel-response/i)).not.toBeInTheDocument();
    });

    it("cancels all exactly once, clears server rows, and removes only matching convenience defaults", async () => {
        const cancellation = deferred<number>();
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify({
            "1186": {sectionKey: "overall", threshold: 40},
            "1656": {sectionKey: "fitness floors", threshold: 30},
        }));
        pushApi.getExistingPushSubscription.mockResolvedValue(browserSubscription());
        pushApi.listPushRules.mockResolvedValue([managedRule, crossFacilityRule]);
        pushApi.cancelAllPushRules.mockReturnValue(cancellation.promise);
        renderOpenCard();

        const cancelAll = await screen.findByRole("button", {name: "Cancel all alerts"});
        fireEvent.click(cancelAll);
        fireEvent.click(cancelAll);
        expect(pushApi.cancelAllPushRules).toHaveBeenCalledTimes(1);
        expect(pushApi.cancelAllPushRules).toHaveBeenCalledWith(validSubscription);
        expect(screen.getByRole("button", {name: "Cancelling all alerts"})).toBeDisabled();

        cancellation.resolve(2);
        await waitFor(() => expect(screen.queryByRole("listitem")).not.toBeInTheDocument());
        expect(screen.getByText("No active alerts for this browser.")).toBeVisible();
        expect(screen.getByText("All alerts cancelled.")).toBeVisible();
        expect(screen.getByRole("status")).toHaveTextContent("All alerts cancelled.");
        expect(JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "null")).toEqual({
            "1656": {sectionKey: "fitness floors", threshold: 30},
        });
    });

    it("prevents cancel-one and cancel-all from overlapping in either direction", async () => {
        const cancelOne = deferred<void>();
        const cancelAll = deferred<number>();
        pushApi.getExistingPushSubscription.mockResolvedValue(browserSubscription());
        pushApi.listPushRules.mockResolvedValue([managedRule, crossFacilityRule]);
        pushApi.cancelPushRule.mockReturnValue(cancelOne.promise);
        pushApi.cancelAllPushRules.mockReturnValue(cancelAll.promise);
        renderOpenCard();

        fireEvent.click(await screen.findByRole("button", {
            name: "Cancel alert for Nick Entire Facility at 40%",
        }));
        const cancelAllWhileOne = screen.getByRole("button", {name: "Cancel all alerts"});
        expect(cancelAllWhileOne).toBeDisabled();
        fireEvent.click(cancelAllWhileOne);
        expect(pushApi.cancelAllPushRules).not.toHaveBeenCalled();

        cancelOne.resolve();
        await waitFor(() => expect(screen.queryByRole("listitem", {
            name: /Alert for Nick Entire Facility at 40%/i,
        })).not.toBeInTheDocument());
        const remainingCancel = screen.getByRole("button", {
            name: "Cancel alert for Bakke Area: legacy cardio at 25%",
        });
        const enabledCancelAll = screen.getByRole("button", {name: "Cancel all alerts"});
        await waitFor(() => expect(enabledCancelAll).toBeEnabled());
        fireEvent.click(enabledCancelAll);

        expect(screen.getByRole("button", {name: "Cancelling all alerts"})).toBeDisabled();
        expect(remainingCancel).toBeDisabled();
        fireEvent.click(remainingCancel);
        expect(pushApi.cancelPushRule).toHaveBeenCalledTimes(1);
        expect(pushApi.cancelAllPushRules).toHaveBeenCalledTimes(1);

        cancelAll.resolve(1);
        await waitFor(() => expect(screen.queryByRole("listitem")).not.toBeInTheDocument());
    });

    it("prevents cancel-one and subscribe from overlapping in either direction", async () => {
        const cancelOne = deferred<void>();
        const subscribe = deferred<{created: boolean; rule: PushRule}>();
        pushApi.getExistingPushSubscription.mockResolvedValue(browserSubscription());
        pushApi.listPushRules.mockResolvedValue([managedRule, crossFacilityRule]);
        pushApi.cancelPushRule.mockReturnValue(cancelOne.promise);
        pushApi.subscribePushRule.mockReturnValue(subscribe.promise);
        renderOpenCard();

        fireEvent.click(await screen.findByRole("button", {
            name: "Cancel alert for Nick Entire Facility at 40%",
        }));
        const setAlertDuringCancel = screen.getByRole("button", {name: "Set alert"});
        expect(setAlertDuringCancel).toBeDisabled();
        setAlertDuringCancel.removeAttribute("disabled");
        fireEvent.click(setAlertDuringCancel);
        expect(pushApi.subscribePushRule).not.toHaveBeenCalled();

        cancelOne.resolve();
        await waitFor(() => expect(screen.queryByRole("listitem", {
            name: /Alert for Nick Entire Facility at 40%/i,
        })).not.toBeInTheDocument());
        const enabledSetAlert = screen.getByRole("button", {name: "Set alert"});
        await waitFor(() => expect(enabledSetAlert).toBeEnabled());
        fireEvent.click(enabledSetAlert);
        await waitFor(() => expect(pushApi.subscribePushRule).toHaveBeenCalledTimes(1));

        const cancelDuringSubscribe = screen.getByRole("button", {
            name: "Cancel alert for Bakke Area: legacy cardio at 25%",
        });
        expect(cancelDuringSubscribe).toBeDisabled();
        cancelDuringSubscribe.removeAttribute("disabled");
        fireEvent.click(cancelDuringSubscribe);
        expect(pushApi.cancelPushRule).toHaveBeenCalledTimes(1);

        subscribe.resolve({created: true, rule: managedRule});
        await waitFor(() => expect(screen.getByText("Alert set successfully.")).toBeVisible());
    });

    it("prevents subscribe and cancel-all from overlapping in either direction", async () => {
        const subscribe = deferred<{created: boolean; rule: PushRule}>();
        const cancelAll = deferred<number>();
        pushApi.getExistingPushSubscription.mockResolvedValue(browserSubscription());
        pushApi.listPushRules.mockResolvedValue([managedRule]);
        pushApi.subscribePushRule.mockReturnValue(subscribe.promise);
        pushApi.cancelAllPushRules.mockReturnValue(cancelAll.promise);
        renderOpenCard();

        const setAlert = await screen.findByRole("button", {name: "Set alert"});
        fireEvent.click(setAlert);
        await waitFor(() => expect(pushApi.subscribePushRule).toHaveBeenCalledTimes(1));
        const cancelAllDuringSubscribe = screen.getByRole("button", {name: "Cancel all alerts"});
        expect(cancelAllDuringSubscribe).toBeDisabled();
        fireEvent.click(cancelAllDuringSubscribe);
        expect(pushApi.cancelAllPushRules).not.toHaveBeenCalled();

        subscribe.resolve({created: false, rule: managedRule});
        await waitFor(() => expect(screen.getByRole("button", {name: "Set alert"})).toBeEnabled());
        const enabledCancelAll = screen.getByRole("button", {name: "Cancel all alerts"});
        await waitFor(() => expect(enabledCancelAll).toBeEnabled());
        fireEvent.click(enabledCancelAll);

        expect(screen.getByRole("button", {name: "Set alert"})).toBeDisabled();
        fireEvent.click(screen.getByRole("button", {name: "Set alert"}));
        expect(pushApi.subscribePushRule).toHaveBeenCalledTimes(1);
        expect(pushApi.cancelAllPushRules).toHaveBeenCalledTimes(1);

        cancelAll.resolve(1);
        await waitFor(() => expect(screen.queryByRole("listitem")).not.toBeInTheDocument());
    });

    it("normalizes the captured draft only when subscription succeeds", async () => {
        const subscribe = deferred<{created: boolean; rule: PushRule}>();
        pushApi.subscribePushRule.mockReturnValue(subscribe.promise);
        renderOpenCard();
        await screen.findByText("No active browser subscription was found.");
        const input = screen.getByLabelText("Alert threshold (%)");
        fireEvent.change(input, {target: {value: "040"}});
        fireEvent.click(screen.getByRole("button", {name: "Set alert"}));
        await waitFor(() => expect(pushApi.subscribePushRule).toHaveBeenCalledTimes(1));
        expect(input).toHaveProperty("value", "040");
        subscribe.resolve({created: true, rule: managedRule});
        await screen.findByText("Alert set successfully.");
        await waitFor(() => expect(input).toHaveProperty("value", "40"));
    });

    it("leaves an unnormalized draft unchanged after subscription failure", async () => {
        const subscribe = deferred<{created: boolean; rule: PushRule}>();
        pushApi.subscribePushRule.mockReturnValue(subscribe.promise);
        renderOpenCard();
        await screen.findByText("No active browser subscription was found.");
        const input = screen.getByLabelText("Alert threshold (%)");
        fireEvent.change(input, {target: {value: "040"}});
        fireEvent.click(screen.getByRole("button", {name: "Set alert"}));
        await waitFor(() => expect(pushApi.subscribePushRule).toHaveBeenCalledTimes(1));
        subscribe.reject(new Error("fixture failure"));
        await screen.findByText("Could not save this alert right now.");
        expect(input).toHaveProperty("value", "040");
    });

    it("normalizes repeated successful submissions with the same captured values", async () => {
        const first = deferred<{created: boolean; rule: PushRule}>();
        const second = deferred<{created: boolean; rule: PushRule}>();
        pushApi.subscribePushRule.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
        renderOpenCard();
        await screen.findByText("No active browser subscription was found.");
        const input = screen.getByLabelText("Alert threshold (%)");
        for (const [index, operation] of [first, second].entries()) {
            fireEvent.change(input, {target: {value: "040"}});
            fireEvent.click(screen.getByRole("button", {name: "Set alert"}));
            await waitFor(() => expect(pushApi.subscribePushRule).toHaveBeenCalledTimes(index + 1));
            expect(input).toHaveProperty("value", "040");
            operation.resolve({created: index === 0, rule: managedRule});
            await waitFor(() => expect(input).toHaveProperty("value", "40"));
            await waitFor(() => expect(screen.getByRole("button", {name: "Set alert"})).toBeEnabled());
        }
        expect(pushApi.ensurePushSubscription).toHaveBeenCalledTimes(1);
        expect(screen.getAllByRole("listitem")).toHaveLength(1);
    });

    it("does not normalize a newly selected section when the captured section succeeds", async () => {
        const subscribe = deferred<{created: boolean; rule: PushRule}>();
        pushApi.subscribePushRule.mockReturnValue(subscribe.promise);
        renderOpenCard();
        await screen.findByText("No active browser subscription was found.");
        fireEvent.change(screen.getByLabelText("Alert threshold (%)"), {target: {value: "040"}});
        fireEvent.click(screen.getByRole("button", {name: "Set alert"}));
        await waitFor(() => expect(pushApi.subscribePushRule).toHaveBeenCalledTimes(1));
        fireEvent.mouseDown(screen.getByRole("combobox", {name: "Gym area"}));
        fireEvent.click(screen.getByRole("option", {name: "Fitness Floors"}));
        fireEvent.change(screen.getByLabelText("Alert threshold (%)"), {target: {value: "031"}});
        subscribe.resolve({created: true, rule: managedRule});
        await screen.findByText("Alert set successfully.");
        expect(screen.getByRole("combobox", {name: "Gym area"})).toHaveTextContent("Fitness Floors");
        expect(screen.getByLabelText("Alert threshold (%)")).toHaveProperty("value", "031");
        expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!)).toEqual({1186: {sectionKey: "overall", threshold: 40}});
    });

    it("does not let a stale subscribe completion overwrite a newly opened owner's form default", async () => {
        const staleSubscribe = deferred<{created: boolean; rule: PushRule}>();
        const newestRule: PushRule = {
            ...managedRule,
            id: 12,
            sectionKey: "fitness floors",
            threshold: 30,
        };
        pushApi.getExistingPushSubscription
            .mockResolvedValueOnce(null)
            .mockResolvedValueOnce(browserSubscription(secondSubscription));
        pushApi.listPushRules.mockResolvedValueOnce([newestRule]);
        pushApi.subscribePushRule.mockReturnValue(staleSubscribe.promise);
        const {rerender} = renderOpenCard();

        expect(await screen.findByText("No active browser subscription was found.")).toBeVisible();
        fireEvent.click(screen.getByRole("button", {name: "Set alert"}));
        await waitFor(() => expect(pushApi.subscribePushRule).toHaveBeenCalledTimes(1));
        rerender(
            <CrowdAlertSubscriptionCard facility={1186} isOpen={false} onClose={vi.fn()} sections={defaultSections()} />
        );
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify({
            "1186": {sectionKey: "fitness floors", threshold: 30},
        }));
        rerender(
            <CrowdAlertSubscriptionCard facility={1186} isOpen onClose={vi.fn()} sections={defaultSections()} />
        );
        fireEvent.change(screen.getByLabelText("Alert threshold (%)"), {target: {value: "031"}});
        await act(async () => {
            await Promise.resolve();
            await Promise.resolve();
        });
        expect(pushApi.getExistingPushSubscription).toHaveBeenCalledTimes(1);

        staleSubscribe.resolve({created: true, rule: managedRule});
        await waitFor(() => expect(pushApi.getExistingPushSubscription).toHaveBeenCalledTimes(2));
        expect(await screen.findByRole("listitem", {name: /Alert for Nick Fitness Floors at 30%/i})).toBeVisible();
        await waitFor(() => expect(screen.getByRole("button", {name: "Set alert"})).toBeEnabled());
        expect(screen.getByLabelText("Alert threshold (%)")).toHaveProperty("value", "031");
        expect(screen.getByRole("listitem", {name: /Alert for Nick Fitness Floors at 30%/i})).toBeVisible();
        expect(JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "null")).toEqual({
            "1186": {sectionKey: "fitness floors", threshold: 30},
        });
    });

    it("does not let a stale cancel-all completion clear a newly opened owner's rules", async () => {
        const staleCancelAll = deferred<number>();
        const newestRule: PushRule = {
            ...managedRule,
            id: 12,
            sectionKey: "fitness floors",
            threshold: 30,
        };
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify({
            "1186": {sectionKey: "overall", threshold: 40},
        }));
        pushApi.getExistingPushSubscription
            .mockResolvedValueOnce(browserSubscription())
            .mockResolvedValueOnce(browserSubscription(secondSubscription));
        pushApi.listPushRules
            .mockResolvedValueOnce([managedRule])
            .mockResolvedValueOnce([newestRule]);
        pushApi.cancelAllPushRules.mockReturnValue(staleCancelAll.promise);
        const {rerender} = renderOpenCard();

        fireEvent.click(await screen.findByRole("button", {name: "Cancel all alerts"}));
        expect(pushApi.cancelAllPushRules).toHaveBeenCalledWith(validSubscription);
        rerender(
            <CrowdAlertSubscriptionCard facility={1186} isOpen={false} onClose={vi.fn()} sections={defaultSections()} />
        );
        rerender(
            <CrowdAlertSubscriptionCard facility={1186} isOpen onClose={vi.fn()} sections={defaultSections()} />
        );
        await act(async () => {
            await Promise.resolve();
            await Promise.resolve();
        });
        expect(pushApi.getExistingPushSubscription).toHaveBeenCalledTimes(1);

        staleCancelAll.resolve(1);
        await waitFor(() => expect(pushApi.getExistingPushSubscription).toHaveBeenCalledTimes(2));
        expect(await screen.findByRole("listitem", {name: /Alert for Nick Fitness Floors at 30%/i})).toBeVisible();
        await waitFor(() => expect(screen.getByRole("button", {name: "Cancel all alerts"})).toBeEnabled());
        expect(screen.getByRole("listitem", {name: /Alert for Nick Fitness Floors at 30%/i})).toBeVisible();
        expect(screen.queryByText("No active alerts for this browser.")).not.toBeInTheDocument();
        expect(JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "null")).toEqual({
            "1186": {sectionKey: "overall", threshold: 40},
        });
    });

    it("defers a same-owner reopen list until an in-flight mutation settles", async () => {
        const cancellation = deferred<void>();
        const refreshedList = deferred<PushRule[]>();
        let cancellationSettled = false;
        pushApi.getExistingPushSubscription.mockResolvedValue(browserSubscription());
        pushApi.listPushRules
            .mockResolvedValueOnce([managedRule, crossFacilityRule])
            .mockImplementationOnce(() => {
                const capturedSnapshot = cancellationSettled
                    ? [crossFacilityRule]
                    : [managedRule, crossFacilityRule];
                return refreshedList.promise.then(() => capturedSnapshot);
            });
        pushApi.cancelPushRule.mockImplementation(() => cancellation.promise.then(() => {
            cancellationSettled = true;
        }));
        const {rerender} = renderOpenCard();

        fireEvent.click(await screen.findByRole("button", {
            name: "Cancel alert for Nick Entire Facility at 40%",
        }));
        rerender(
            <CrowdAlertSubscriptionCard facility={1186} isOpen={false} onClose={vi.fn()} sections={defaultSections()} />
        );
        rerender(
            <CrowdAlertSubscriptionCard facility={1186} isOpen onClose={vi.fn()} sections={defaultSections()} />
        );
        await act(async () => {
            await Promise.resolve();
            await Promise.resolve();
        });
        expect(pushApi.getExistingPushSubscription).toHaveBeenCalledTimes(1);
        expect(pushApi.listPushRules).toHaveBeenCalledTimes(1);

        cancellation.resolve();
        await waitFor(() => expect(pushApi.listPushRules).toHaveBeenCalledTimes(2));
        refreshedList.resolve([]);

        expect(await screen.findByRole("listitem", {
            name: /Alert for Bakke Area: legacy cardio at 25%/i,
        })).toBeVisible();
        expect(screen.queryByRole("listitem", {
            name: /Alert for Nick Entire Facility at 40%/i,
        })).not.toBeInTheDocument();
        expect(pushApi.cancelPushRule).toHaveBeenCalledTimes(1);
    });

    it("retains all rules/defaults and associates a fixed safe error when cancel-all fails", async () => {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify({
            "1186": {sectionKey: "overall", threshold: 40},
        }));
        pushApi.getExistingPushSubscription.mockResolvedValue(browserSubscription());
        pushApi.listPushRules.mockResolvedValue([managedRule]);
        pushApi.cancelAllPushRules.mockRejectedValue(new Error("private-cancel-all-response"));
        renderOpenCard();

        fireEvent.click(await screen.findByRole("button", {name: "Cancel all alerts"}));

        const error = await screen.findByText("Could not cancel alerts right now.");
        const button = screen.getByRole("button", {name: "Cancel all alerts"});
        expect(button).toHaveAttribute("aria-describedby", error.id);
        expect(screen.getByRole("listitem", {name: /Alert for Nick Entire Facility at 40%/i})).toBeVisible();
        expect(JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "null")).toEqual({
            "1186": {sectionKey: "overall", threshold: 40},
        });
        expect(screen.queryByText(/private-cancel-all-response/i)).not.toBeInTheDocument();
    });

    it.each([
        ["malformed JSON", "{"],
        ["valid but stale convenience data", JSON.stringify({"1186": {sectionKey: "overall", threshold: 40}})],
    ])("never synthesizes an active rule from %s in localStorage", async (_name, storedValue) => {
        window.localStorage.setItem(STORAGE_KEY, storedValue);
        pushApi.getExistingPushSubscription.mockResolvedValue(browserSubscription());
        pushApi.listPushRules.mockResolvedValue([]);

        renderOpenCard();

        expect(await screen.findByText("No active alerts for this browser.")).toBeVisible();
        expect(screen.queryByRole("listitem")).not.toBeInTheDocument();
        expect(pushApi.subscribePushRule).not.toHaveBeenCalled();
    });

    it("strips extra stored fields before a later successful default write", async () => {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify({
            "1656": {
                sectionKey: "fitness floors",
                threshold: 30,
                endpoint: "private-stored-endpoint",
                ruleId: 999,
                keys: {auth: "private-stored-auth"},
            },
        }));
        renderOpenCard();
        expect(await screen.findByText("No active browser subscription was found.")).toBeVisible();

        fireEvent.click(screen.getByRole("button", {name: "Set alert"}));

        await waitFor(() => expect(screen.getByText("Alert set successfully.")).toBeVisible());
        expect(JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "null")).toEqual({
            "1186": {sectionKey: "overall", threshold: 40},
            "1656": {sectionKey: "fitness floors", threshold: 30},
        });
        expect(window.localStorage.getItem(STORAGE_KEY)).not.toContain("private-stored");
    });

    it("ignores a stale async open after close/reopen and keeps the newest server list", async () => {
        const firstLookup = deferred<PushSubscription | null>();
        const secondRule: PushRule = {
            ...managedRule,
            id: 11,
            sectionKey: "fitness floors",
            threshold: 30,
        };
        pushApi.getExistingPushSubscription
            .mockReturnValueOnce(firstLookup.promise)
            .mockResolvedValueOnce(browserSubscription(secondSubscription));
        pushApi.listPushRules.mockImplementation(async (subscription: PushSubscriptionJSON) => (
            subscription.endpoint === secondSubscription.endpoint ? [secondRule] : [managedRule]
        ));
        const {rerender} = renderOpenCard();
        await waitFor(() => expect(pushApi.getExistingPushSubscription).toHaveBeenCalledTimes(1));

        rerender(
            <CrowdAlertSubscriptionCard facility={1186} isOpen={false} onClose={vi.fn()} sections={defaultSections()} />
        );
        rerender(
            <CrowdAlertSubscriptionCard facility={1186} isOpen onClose={vi.fn()} sections={defaultSections()} />
        );

        await waitFor(() => expect(pushApi.getExistingPushSubscription).toHaveBeenCalledTimes(2));
        expect(await screen.findByRole("listitem", {name: /Alert for Nick Fitness Floors at 30%/i})).toBeVisible();
        firstLookup.resolve(browserSubscription());
        await act(async () => {
            await Promise.resolve();
            await Promise.resolve();
        });

        expect(screen.getByRole("listitem", {name: /Alert for Nick Fitness Floors at 30%/i})).toBeVisible();
        expect(screen.queryByRole("listitem", {name: /Alert for Nick Entire Facility at 40%/i})).not.toBeInTheDocument();
        expect(pushApi.listPushRules).not.toHaveBeenCalledWith(validSubscription);
    });

    it("uses a synchronous guard so rapid submit clicks call subscribe exactly once and keep an accessible loading name", async () => {
        const subscribe = deferred<{created: boolean; rule: PushRule}>();
        pushApi.subscribePushRule.mockReturnValue(subscribe.promise);
        renderOpenCard();
        expect(await screen.findByText("No active browser subscription was found.")).toBeVisible();

        const button = screen.getByRole("button", {name: "Set alert"});
        fireEvent.click(button);
        fireEvent.click(button);

        await waitFor(() => expect(pushApi.subscribePushRule).toHaveBeenCalledTimes(1));
        expect(screen.getByRole("button", {name: "Setting alert"})).toBeDisabled();
        subscribe.resolve({created: true, rule: managedRule});
        await waitFor(() => expect(screen.getByText("Alert set successfully.")).toBeVisible());
    });

    it("keeps only live sections selectable without exposing partial coverage", () => {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify({
            "1186": {sectionKey: "fitness floors", threshold: 20},
        }));
        render(
            <CrowdAlertSubscriptionCard
                facility={1186}
                isOpen={false}
                onClose={vi.fn()}
                sections={[
                    {key: "overall", label: "Entire Facility", summary: summary("live", 60)},
                    {key: "fitness floors", label: "Fitness Floors", summary: summary("partial", 45)},
                    {key: "unknown", label: "Unknown Area", summary: summary("unknown", null)},
                    {key: "insufficient", label: "Insufficient Area", summary: summary("insufficient", null)},
                    {key: "closed", label: "Closed Area", summary: summary("closed", null)},
                ]}
            />
        );

        expect(screen.getByRole("combobox", {name: "Gym area"})).toHaveTextContent("Entire Facility");
        expect(screen.getByText(/25 people \(60% full\)/)).toBeVisible();
        expect(screen.queryByText(/Coverage:/)).not.toBeInTheDocument();
        expect(screen.getByLabelText("Alert threshold (%)")).toBeEnabled();

        fireEvent.mouseDown(screen.getByRole("combobox", {name: "Gym area"}));
        expect(screen.getByRole("option", {name: "Entire Facility"})).not.toHaveAttribute("aria-disabled", "true");
        expect(screen.queryByRole("option", {name: "Fitness Floors"})).not.toBeInTheDocument();
        expect(screen.queryByRole("option", {name: "Unknown Area"})).not.toBeInTheDocument();
        expect(screen.queryByRole("option", {name: "Insufficient Area"})).not.toBeInTheDocument();
        expect(screen.getByRole("option", {name: "Closed Area"})).toHaveAttribute("aria-disabled", "true");
    });

    it("uses concise unavailable copy when no live section can set a threshold", () => {
        render(
            <CrowdAlertSubscriptionCard
                facility={1186}
                isOpen={false}
                onClose={vi.fn()}
                sections={[
                    {key: "partial", label: "Partial Area", summary: summary("partial", 45)},
                    {key: "unknown", label: "Unknown Area", summary: summary("unknown", null)},
                ]}
            />
        );

        expect(screen.getByText("Current occupancy unavailable.")).toBeVisible();
        expect(screen.getByRole("button", {name: "Set alert"})).toBeDisabled();
    });
});

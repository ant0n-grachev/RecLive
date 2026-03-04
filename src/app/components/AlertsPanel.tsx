import {Suspense, lazy} from "react";
import {Box, CircularProgress, Dialog, DialogContent, IconButton, SwipeableDrawer, Typography} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import type {FacilityId} from "../../lib/types/facility";
import type {AlertSectionOption} from "../../facilities/CrowdAlertSubscriptionCard";

const CrowdAlertSubscriptionCard = lazy(() => import("../../facilities/CrowdAlertSubscriptionCard"));

interface AlertsPanelProps {
    open: boolean;
    onClose: () => void;
    facility: FacilityId;
    sections: AlertSectionOption[];
    useDesktopModal: boolean;
    isStandalonePwa: boolean;
    isTouchCapable: boolean;
}

const content = (
    isOpen: boolean,
    facility: FacilityId,
    sections: AlertSectionOption[],
    onClose: () => void,
    useDesktopModal: boolean,
    isStandalonePwa: boolean,
    isTouchCapable: boolean
) => (
    <>
        <Box sx={{display: "flex", alignItems: "center", justifyContent: "space-between", pb: 1}}>
            <Typography variant="h6">
                Alerts
            </Typography>
            <IconButton
                size="small"
                onClick={onClose}
                aria-label="Close alerts"
                sx={{color: "text.secondary"}}
            >
                <CloseIcon fontSize="small"/>
            </IconButton>
        </Box>
        <Box sx={{overflowY: "auto", pb: 1, maxHeight: useDesktopModal ? "65vh" : undefined}}>
            <Suspense
                fallback={(
                    <Box sx={{display: "grid", placeItems: "center", py: 3}}>
                        <CircularProgress size={24} thickness={5}/>
                    </Box>
                )}
            >
                <CrowdAlertSubscriptionCard
                    key={facility}
                    onClose={onClose}
                    facility={facility}
                    sections={sections}
                    isOpen={isOpen}
                    requireStandalonePwaForAlerts={!useDesktopModal && isTouchCapable && !isStandalonePwa}
                />
            </Suspense>
        </Box>
    </>
);

export default function AlertsPanel({
    open,
    onClose,
    facility,
    sections,
    useDesktopModal,
    isStandalonePwa,
    isTouchCapable,
}: AlertsPanelProps) {
    if (useDesktopModal) {
        return (
            <Dialog
                open={open}
                onClose={onClose}
                fullWidth
                maxWidth="sm"
                PaperProps={{
                    sx: {
                        borderRadius: 3,
                        px: 0.25,
                    },
                }}
            >
                <DialogContent key={`alerts-${facility}`} sx={{pt: 2, px: 2.5, pb: 2}}>
                    {content(open, facility, sections, onClose, useDesktopModal, isStandalonePwa, isTouchCapable)}
                </DialogContent>
            </Dialog>
        );
    }

    return (
        <SwipeableDrawer
            anchor="bottom"
            open={open}
            onClose={onClose}
            onOpen={() => {
                // Required by SwipeableDrawer; open is controlled externally.
            }}
            disableSwipeToOpen
            ModalProps={{keepMounted: true}}
            PaperProps={{
                sx: {
                    borderTopLeftRadius: 20,
                    borderTopRightRadius: 20,
                    height: {xs: "75vh", sm: "60vh"},
                    maxHeight: "75vh",
                    px: 2,
                    pt: 1,
                    pb: 2,
                },
            }}
        >
            <Box sx={{display: "flex", justifyContent: "center", pb: 1}}>
                <Box
                    sx={{
                        width: 44,
                        height: 5,
                        borderRadius: 999,
                        bgcolor: "rgba(15, 23, 42, 0.2)",
                    }}
                />
            </Box>
            <Box key={`alerts-${facility}`}>
                {content(open, facility, sections, onClose, useDesktopModal, isStandalonePwa, isTouchCapable)}
            </Box>
        </SwipeableDrawer>
    );
}

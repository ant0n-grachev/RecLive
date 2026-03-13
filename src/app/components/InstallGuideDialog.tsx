import {Box, Button, Dialog, DialogContent, Stack, Typography} from "@mui/material";

interface InstallGuideDialogProps {
    open: boolean;
    onClose: () => void;
}

export default function InstallGuideDialog({open, onClose}: InstallGuideDialogProps) {
    return (
        <Dialog
            open={open}
            onClose={onClose}
            fullWidth
            maxWidth="xs"
            PaperProps={{sx: {borderRadius: 3}}}
        >
            <DialogContent sx={{pt: 2.5, px: 2.5, pb: 2}}>
                <Typography variant="h6" sx={{pb: 1}}>
                    Add RecLive to Home Screen
                </Typography>
                <Stack spacing={1.25}>
                    <Typography variant="body2" color="text.secondary">
                        <Box component="span" sx={{fontWeight: 700}}>iPhone (Safari): </Box>
                        tap Share, then choose Add to Home Screen.
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        <Box component="span" sx={{fontWeight: 700}}>Android (Chrome): </Box>
                        tap the menu, then Install app or Add to Home screen.
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Open it from your home screen next time for the full app experience.
                    </Typography>
                </Stack>
                <Box sx={{display: "flex", justifyContent: "flex-end", pt: 2}}>
                    <Button onClick={onClose} sx={{textTransform: "none"}}>
                        Close
                    </Button>
                </Box>
            </DialogContent>
        </Dialog>
    );
}

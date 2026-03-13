import {Box, Button, IconButton, Link, Stack, Typography} from "@mui/material";
import GitHubIcon from "@mui/icons-material/GitHub";
import {EXTERNAL_LINK_REL, openExternalInBrowser} from "../../shared/utils/externalLink";

interface AppFooterProps {
    isStandalonePwa: boolean;
    onOpenInstallGuide: () => void;
}

export default function AppFooter({isStandalonePwa, onOpenInstallGuide}: AppFooterProps) {
    return (
        <Box sx={{pt: {xs: 0.75, sm: 1.25}, pb: {xs: 0.5, sm: 0.75}}}>
            {!isStandalonePwa && (
                <Box sx={{display: "flex", justifyContent: "center", pb: 0.5}}>
                    <Button
                        variant="text"
                        size="small"
                        onClick={onOpenInstallGuide}
                        sx={{textTransform: "none", fontWeight: 700, fontSize: "0.78rem"}}
                    >
                        How to add RecLive to your home screen
                    </Button>
                </Box>
            )}
            <Stack direction="row" spacing={0.75} alignItems="center" justifyContent="center">
                <Typography variant="caption" color="text.secondary" sx={{fontSize: "0.78rem"}}>
                    Train smarter. Skip the crowd. Built by{" "}
                    <Link
                        href="https://anton.grachev.us"
                        target="_blank"
                        rel={EXTERNAL_LINK_REL}
                        underline="hover"
                        sx={{fontWeight: 700}}
                        onClick={(event) => {
                            event.preventDefault();
                            openExternalInBrowser("https://anton.grachev.us");
                        }}
                    >
                        Anton
                    </Link>
                    {" "}and{" "}
                    <Link
                        href="https://github.com/alexgabrichidze"
                        target="_blank"
                        rel={EXTERNAL_LINK_REL}
                        underline="hover"
                        sx={{fontWeight: 700}}
                        onClick={(event) => {
                            event.preventDefault();
                            openExternalInBrowser("https://github.com/alexgabrichidze");
                        }}
                    >
                        Alex
                    </Link>
                </Typography>
                <IconButton
                    component="a"
                    href="https://github.com/ant0n-grachev/RecLive"
                    target="_blank"
                    rel={EXTERNAL_LINK_REL}
                    aria-label="RecLive GitHub repository"
                    size="small"
                    sx={{color: "text.secondary"}}
                    onClick={(event) => {
                        event.preventDefault();
                        openExternalInBrowser("https://github.com/ant0n-grachev/RecLive");
                    }}
                >
                    <GitHubIcon fontSize="small"/>
                </IconButton>
            </Stack>
        </Box>
    );
}

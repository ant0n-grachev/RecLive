import {Card} from "@mui/material";
import type {ReactNode} from "react";
import {CARD_SHELL_SX} from "../utils/styles";

interface ModernCardProps {
    children: ReactNode;
    disableMinHeight?: boolean;
}

export default function ModernCard({children, disableMinHeight = false}: ModernCardProps) {
    return (
        <Card
            variant="outlined"
            sx={{
                ...CARD_SHELL_SX,
                p: {xs: 2, sm: 2.25},
                display: "flex",
                flexDirection: "column",
                gap: 1.25,
                minHeight: disableMinHeight ? 0 : 150,
                height: disableMinHeight ? "auto" : "100%",
            }}
        >
            {children}
        </Card>
    );
}

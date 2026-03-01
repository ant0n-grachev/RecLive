import {Card} from "@mui/material";
import type {ReactNode} from "react";

export default function ModernCard({children}: { children: ReactNode }) {
    return (
        <Card
            variant="outlined"
            sx={{
                p: 3,
                borderRadius: 3,
                borderColor: "divider",
                backgroundColor: "background.paper",
                display: "flex",
                flexDirection: "column",
                gap: 1.25,
                minHeight: 150,
                height: "100%",
            }}
        >
            {children}
        </Card>
    );
}

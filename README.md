# RecLive

RecLive is a lightweight Progressive Web App that surfaces real-time occupancy for the University of Wisconsin-Madison RecWell facilities (Nick and Bakke). The UI focuses on at-a-glance capacity signals, mobile-first ergonomics, and installability so you can pin the dashboard to your phone like a native app.

<img width="800" height="710" alt="Screenshot 2025-11-14 at 11 49 13 PM" src="https://github.com/user-attachments/assets/e4fa53d7-5582-450f-861b-d461a51a0937" />

## Key features

- **Live occupancy feed**: Fetches the public RecWell endpoint on demand with abort-safe requests so stale data never flashes.
- **Category breakdowns**: Fitness floors, courts, pools, track, ice, and "Other" sections automatically group and display their individual load plus overall totals.
- **Status cues**: Color-coded percentages and explicit CLOSED badges make capacity issues obvious.

## Tech Stack

- **React 19 + TypeScript** for the component architecture
- **Vite** tooling for fast dev/build cycles
- **Material UI 7** for layout primitives and theming

## Getting Started

```bash
# Install dependencies
npm install

# Start Vite dev server (http://localhost:5173)
npm run dev

# Type-check and build production assets
npm run build

# Preview the production build
npm run preview
```

## Data Sources
**Facility counts API** – `https://goboardapi.azurewebsites.net/api/FacilityCount/GetCountsByAccount` (public RecWell feed this dashboard uses for live numbers).

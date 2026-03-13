# RecLive

*Train smarter. Skip the crowd.*

RecLive is a live gym intelligence app for UW students.

It helps people answer one simple question before they walk over:
**"Is it worth going right now?"**

## The Idea

Campus gyms can feel random.
Sometimes they are perfect, sometimes they are packed.
RecLive gives students a fast read on current crowd levels and near-term trends so they can plan better workouts.

## What RecLive Does

- Shows real-time occupancy for Nick and Bakke
- Breaks crowd levels down by key gym areas
- Highlights daily forecast windows (low, medium, peak)
- Sends one-time alerts when occupancy drops below your threshold
- Works as a mobile-first Progressive Web App

## Why It Matters

- Less time wasted traveling to packed gyms
- Better workout consistency
- Better experience for both beginners and regulars

## Product Focus

RecLive is designed to be:

- Fast to read
- Simple to trust
- Useful in seconds

No dashboard overload. Just the info you need to decide when to go.

## Tech Stack

- Frontend: React, TypeScript, Vite, Material UI
- Backend API: FastAPI (Python)
- Data: MySQL + live occupancy feed ingestion
- Forecasting: XGBoost predictions
- Notifications: Web Push (VAPID)
- Platform: Progressive Web App (PWA)

## Built by

Built by Anton and [Alex](https://github.com/alexgabrichidze).

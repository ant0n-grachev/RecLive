from datetime import datetime, timezone


NOW = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
HEALTHY_OBSERVED_AT = datetime(2026, 8, 31, 11, 55, tzinfo=timezone.utc)
STALE_OBSERVED_AT = datetime(2026, 8, 31, 11, 0, tzinfo=timezone.utc)
BOUNDARY_OBSERVED_AT = datetime(2026, 8, 31, 11, 50, tzinfo=timezone.utc)
JUST_STALE_OBSERVED_AT = datetime(
    2026,
    8,
    31,
    11,
    49,
    59,
    999999,
    tzinfo=timezone.utc,
)
FUTURE_OBSERVED_AT = datetime(2026, 8, 31, 12, 0, 0, 1, tzinfo=timezone.utc)


def healthy_schedule_payload(
    *,
    generated_at: str = "2026-08-31T11:59:00Z",
    nick_observed_at: str = "2026-08-31T11:50:00Z",
    bakke_observed_at: str = "2026-08-31T11:55:00Z",
) -> dict[str, object]:
    def facility(
        facility_id: int,
        name: str,
        slug: str,
        observed_at: str,
    ) -> dict[str, object]:
        url = f"https://recwell.example.test/locations/{slug}/"
        return {
            "facilityId": facility_id,
            "facilityName": name,
            "slug": slug,
            "url": url,
            "resolvedUrl": url,
            "status": "ok",
            "source": "direct_html",
            "sourceModifiedGmt": None,
            "sections": [
                {
                    "title": "Building Hours",
                    "rows": [
                        {"label": "Mon-Fri", "hours": "6:00 am - 10:00 pm"}
                    ],
                    "note": None,
                }
            ],
            "sourceFetchedAt": observed_at,
            "lastSuccessfulAt": observed_at,
            "stale": False,
            "error": None,
            "errorCategory": None,
            "updatedAt": observed_at,
        }

    return {
        "generatedAt": generated_at,
        "sourceSite": "https://recwell.example.test",
        "facilities": [
            facility(1186, "Nick", "nick", nick_observed_at),
            facility(1656, "Bakke", "bakke", bakke_observed_at),
        ],
        "okCount": 2,
        "totalCount": 2,
    }

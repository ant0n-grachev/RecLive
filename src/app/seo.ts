import {useEffect} from "react";
import type {FacilityId} from "../lib/types/facility";
import {env} from "../lib/config/env";

const DEFAULT_SITE_NAME = "RecLive";
const DEFAULT_SITE_URL = "https://reclive.netlify.app";
const DEFAULT_IMAGE_PATH = "/icons/icon-512.png";

interface FacilitySeoConfig {
    slug: "nick" | "bakke";
    shortName: string;
    title: string;
    description: string;
}

const FACILITY_SEO: Record<FacilityId, FacilitySeoConfig> = {
    1186: {
        slug: "nick",
        shortName: "Nick",
        title: "Nick Gym Crowd | RecLive",
        description: "Live crowd levels, occupancy trends, and near-term forecasts for UW Nick Recreation Center.",
    },
    1656: {
        slug: "bakke",
        shortName: "Bakke",
        title: "Bakke Gym Crowd | RecLive",
        description: "Live crowd levels, occupancy trends, and near-term forecasts for UW Bakke Recreation & Wellbeing Center.",
    },
};

const sanitizeBaseUrl = (value: string): string => value.replace(/\/+$/, "");

const resolveBaseUrl = (): string => {
    const configured = env.siteUrl;
    if (configured) {
        return sanitizeBaseUrl(configured);
    }
    if (typeof window !== "undefined" && window.location.origin) {
        return sanitizeBaseUrl(window.location.origin);
    }
    return DEFAULT_SITE_URL;
};

const ensureMetaByName = (name: string): HTMLMetaElement => {
    let element = document.head.querySelector(`meta[name="${name}"]`) as HTMLMetaElement | null;
    if (!element) {
        element = document.createElement("meta");
        element.setAttribute("name", name);
        document.head.appendChild(element);
    }
    return element;
};

const ensureMetaByProperty = (property: string): HTMLMetaElement => {
    let element = document.head.querySelector(`meta[property="${property}"]`) as HTMLMetaElement | null;
    if (!element) {
        element = document.createElement("meta");
        element.setAttribute("property", property);
        document.head.appendChild(element);
    }
    return element;
};

const ensureCanonicalLink = (): HTMLLinkElement => {
    let element = document.head.querySelector("link[rel='canonical']") as HTMLLinkElement | null;
    if (!element) {
        element = document.createElement("link");
        element.setAttribute("rel", "canonical");
        document.head.appendChild(element);
    }
    return element;
};

const ensureJsonLdScript = (id: string): HTMLScriptElement => {
    let element = document.head.querySelector(`script#${id}`) as HTMLScriptElement | null;
    if (!element) {
        element = document.createElement("script");
        element.type = "application/ld+json";
        element.id = id;
        document.head.appendChild(element);
    }
    return element;
};

export const facilityPath = (facilityId: FacilityId): string => (
    FACILITY_SEO[facilityId].slug === "nick" ? "/nick" : "/bakke"
);

export const useFacilitySeo = (facilityId: FacilityId): void => {
    useEffect(() => {
        const config = FACILITY_SEO[facilityId];
        const baseUrl = resolveBaseUrl();
        const pagePath = facilityPath(facilityId);
        const pageUrl = `${baseUrl}${pagePath}`;
        const imageUrl = `${baseUrl}${DEFAULT_IMAGE_PATH}`;

        document.title = config.title;

        ensureMetaByName("description").setAttribute("content", config.description);
        ensureMetaByName("robots").setAttribute(
            "content",
            "index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1"
        );
        ensureMetaByName("application-name").setAttribute("content", DEFAULT_SITE_NAME);
        ensureMetaByName("twitter:card").setAttribute("content", "summary_large_image");
        ensureMetaByName("twitter:title").setAttribute("content", config.title);
        ensureMetaByName("twitter:description").setAttribute("content", config.description);
        ensureMetaByName("twitter:image").setAttribute("content", imageUrl);
        ensureMetaByName("twitter:image:alt").setAttribute("content", `${config.shortName} occupancy dashboard`);

        ensureMetaByProperty("og:type").setAttribute("content", "website");
        ensureMetaByProperty("og:site_name").setAttribute("content", DEFAULT_SITE_NAME);
        ensureMetaByProperty("og:locale").setAttribute("content", "en_US");
        ensureMetaByProperty("og:title").setAttribute("content", config.title);
        ensureMetaByProperty("og:description").setAttribute("content", config.description);
        ensureMetaByProperty("og:url").setAttribute("content", pageUrl);
        ensureMetaByProperty("og:image").setAttribute("content", imageUrl);
        ensureMetaByProperty("og:image:alt").setAttribute("content", `${config.shortName} occupancy dashboard`);

        ensureCanonicalLink().setAttribute("href", pageUrl);

        const webPageSchema = {
            "@context": "https://schema.org",
            "@type": "WebPage",
            name: config.title,
            description: config.description,
            url: pageUrl,
            isPartOf: {
                "@type": "WebSite",
                name: DEFAULT_SITE_NAME,
                url: baseUrl,
            },
            about: {
                "@type": "SportsActivityLocation",
                name: config.shortName,
                sameAs: config.slug === "nick"
                    ? "https://recwell.wisc.edu/locations/nick/"
                    : "https://recwell.wisc.edu/locations/bakke/",
            },
        };
        ensureJsonLdScript("reclive-route-jsonld").textContent = JSON.stringify(webPageSchema);
    }, [facilityId]);
};

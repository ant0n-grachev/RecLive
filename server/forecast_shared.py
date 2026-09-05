"""Compatibility export for the canonical section normalizer."""

if not __package__:
    import reclive  # noqa: F401 - initialize the canonical parent for direct scripts.

from server.reclive.sections import normalize_section_key as normalize_section_key

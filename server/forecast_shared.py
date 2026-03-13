def normalize_section_key(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())

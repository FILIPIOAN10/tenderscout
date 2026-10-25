TECH_ALIASES: dict[str, str] = {
    "k8s": "kubernetes",
    "azure kubernetes service": "aks",
    "postgres": "postgresql",
    "github actions": "github-actions",
    "azure openai": "azure-openai",
    "spring boot": "spring-boot",
    "python3": "python",
}

IMPLIES: dict[str, set[str]] = {
    "aks": {"kubernetes", "azure"},
    "azure-openai": {"azure"},
    "langgraph": {"langchain"},
}


def canonical_tech(value: str) -> str:
    key = value.strip().lower()
    return TECH_ALIASES.get(key, key.replace(" ", "-"))


def expand_tech(tags: list[str]) -> list[str]:
    """Used at INGEST time: a doc tagged 'aks' also matches 'kubernetes' and 'azure'."""
    normalized = {canonical_tech(t) for t in tags}
    out = set(normalized)
    for t in normalized:
        out |= IMPLIES.get(t, set())
    return sorted(out)

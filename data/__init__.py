"""Data module — re-exports all public helpers from license_corpus."""

from data.license_corpus import (
    LICENSE_CORPUS,
    CONFLICT_RULES,
    SAAS_FORBIDDEN,
    get_conflict_severity,
)

__all__ = [
    "LICENSE_CORPUS",
    "CONFLICT_RULES",
    "SAAS_FORBIDDEN",
    "get_conflict_severity",
]

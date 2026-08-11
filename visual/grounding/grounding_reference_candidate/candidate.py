"""Artifact entrypoint for the persistent operational reference learner."""

from grounding_kernel.unified_grounder import PersistentOperationalGrounder


def build() -> PersistentOperationalGrounder:
    return PersistentOperationalGrounder()


__all__ = ["build"]

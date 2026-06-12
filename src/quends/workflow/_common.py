"""Shared helpers for the workflow layer."""

from typing import Any, List

from ..base.data_stream import DataStream


def resolve_members(ensemble_or_members: Any) -> List[DataStream]:
    """Accept an Ensemble or a plain list of DataStreams; return the member list."""
    # Local import to avoid a circular import at module load time.
    from ..base.ensemble import Ensemble  # noqa: PLC0415

    if isinstance(ensemble_or_members, Ensemble):
        return ensemble_or_members.members()
    if isinstance(ensemble_or_members, list):
        return ensemble_or_members
    raise TypeError(
        "ensemble_or_members must be an Ensemble or a list of DataStream objects; "
        f"got {type(ensemble_or_members).__name__!r}."
    )

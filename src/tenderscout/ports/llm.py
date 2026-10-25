from typing import Protocol

from tenderscout.domain.models import (
    Evidence,
    ExtractionResult,
    GradeResult,
    MatchResult,
    ProposalDraft,
    Requirement,
)


class LLMPort(Protocol):
    """Everything the graph asks of a language model; nodes depend on this, not on a vendor SDK."""

    async def extract_requirements(self, rfp_text: str) -> ExtractionResult: ...

    async def grade_candidates(
        self, requirement: Requirement, candidates: list[Evidence]
    ) -> GradeResult: ...

    async def write_draft(
        self,
        requirements: list[Requirement],
        matches: dict[str, MatchResult],
        feedback: str | None = None,
    ) -> ProposalDraft: ...

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from tenderscout.domain.vocab import canonical_tech

TechTag = Annotated[str, AfterValidator(canonical_tech)]
DocType = Literal["project", "cv"]
GradeStatus = Literal["covered", "partial", "gap"]


class Priority(StrEnum):
    MUST = "must"  # MUSS
    SHOULD = "should"  # SOLL
    NICE = "nice"  # KANN


class Category(StrEnum):
    FUNCTIONAL = "functional"
    TECHNICAL = "technical"
    COMPLIANCE = "compliance"
    STAFFING = "staffing"
    DOMAIN = "domain"


class Industry(StrEnum):
    ENERGY = "energy"
    INSURANCE = "insurance"
    HEALTHCARE = "healthcare"
    LOGISTICS = "logistics"
    PUBLIC_SECTOR = "public_sector"


class Requirement(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^R\d{1,3}$")
    source_section: str = Field(description="Section number in the RFP, e.g. '2.3'")
    source_quote: str = Field(description="Verbatim sentence from the RFP, original language")
    language: Literal["de", "en"]
    summary_en: str = Field(max_length=300)
    category: Category
    priority: Priority
    # --- become Qdrant filters ---
    technologies: list[TechTag] = Field(default_factory=list)
    industry: Industry | None = None
    certifications: list[str] = Field(default_factory=list)
    min_years_experience: int | None = Field(default=None, ge=0, le=30)
    target_doc_types: list[DocType] = Field(default_factory=list)
    # --- becomes the retrieval query ---
    search_query_en: str = Field(description="Short English query for hybrid search")


class ExtractionResult(BaseModel):
    requirements: list[Requirement]
    reference_window_years: int | None = Field(
        default=None, ge=1, le=30, description="'aus den letzten fünf Jahren' -> 5"
    )


class Evidence(BaseModel):
    doc_id: str
    doc_type: DocType
    title: str
    snippet: str


class GradeResult(BaseModel):
    """What the grading LLM decides for one requirement over the retrieved candidates."""

    status: GradeStatus
    selected_doc_ids: list[str] = Field(
        default_factory=list, description="Only ids taken from the candidates"
    )
    rationale: str = ""


class MatchResult(BaseModel):
    requirement_id: str
    status: Literal["covered", "partial", "gap", "error"]
    evidence: list[Evidence] = Field(default_factory=list)
    rationale: str = ""
    filters_relaxed: list[str] = Field(default_factory=list)

    @classmethod
    def error(cls, requirement_id: str) -> "MatchResult":
        return cls(requirement_id=requirement_id, status="error", rationale="matching failed")


class ProposalDraft(BaseModel):
    markdown: str
    cited_doc_ids: list[str] = Field(default_factory=list)


class ReviewDecision(BaseModel):
    action: Literal["approve", "revise"]
    comments: str = ""

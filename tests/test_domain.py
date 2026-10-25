import pytest
from pydantic import ValidationError

from tenderscout.adapters.fake_llm import FakeLLM
from tenderscout.domain.models import (
    Category,
    Evidence,
    Priority,
    Requirement,
)
from tenderscout.domain.vocab import canonical_tech, expand_tech
from tenderscout.ports.llm import LLMPort


def test_canonical_tech_normalizes_aliases() -> None:
    assert canonical_tech("K8s") == "kubernetes"
    assert canonical_tech("  Azure Kubernetes Service ") == "aks"
    assert canonical_tech("Postgres") == "postgresql"
    assert canonical_tech("GitHub Actions") == "github-actions"
    assert canonical_tech("Spring Boot") == "spring-boot"


def test_expand_tech_adds_implied_tags() -> None:
    expanded = expand_tech(["AKS", "LangGraph"])
    assert expanded == ["aks", "azure", "kubernetes", "langchain", "langgraph"]


def test_requirement_normalizes_tech_and_validates_id() -> None:
    req = Requirement(
        id="R1",
        source_section="2.1",
        source_quote="Must use K8s and Postgres",
        language="en",
        summary_en="Use Kubernetes and PostgreSQL",
        category=Category.TECHNICAL,
        priority=Priority.MUST,
        technologies=["K8s", "Postgres", "Python3"],
        search_query_en="kubernetes postgresql python",
    )
    assert req.technologies == ["kubernetes", "postgresql", "python"]

    with pytest.raises(ValidationError):
        Requirement(
            id="INVALID_ID",
            source_section="2.1",
            source_quote="quote",
            language="en",
            summary_en="summary",
            category=Category.TECHNICAL,
            priority=Priority.MUST,
            search_query_en="query",
        )


async def test_fake_llm_satisfies_protocol_and_rejects_cv03_on_section_2_4() -> None:
    llm: LLMPort = FakeLLM()
    extraction = await llm.extract_requirements("Sample RFP text")
    assert extraction.reference_window_years == 5
    assert len(extraction.requirements) == 7

    req_2_4 = next(r for r in extraction.requirements if r.source_section == "2.4")
    candidates = [
        Evidence(
            doc_id="CV-01",
            doc_type="cv",
            title="Anna",
            snippet="Seven years of Python and Kubernetes experience.",
        ),
        Evidence(
            doc_id="CV-02",
            doc_type="cv",
            title="Mihai",
            snippet="AKS, Kubernetes and Python automation.",
        ),
        Evidence(
            doc_id="CV-03",
            doc_type="cv",
            title="Laura",
            snippet="Four years of Java, Spring Boot and Kubernetes.",
        ),
    ]
    grade = await llm.grade_candidates(req_2_4, candidates)
    assert grade.status == "covered"
    assert grade.selected_doc_ids == ["CV-01", "CV-02"]
    assert "CV-03" not in grade.selected_doc_ids

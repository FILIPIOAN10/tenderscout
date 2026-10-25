"""Deterministic stand-in for the LLM, so tests and CI never need an API key.

Extraction returns the requirements of data/rfp/stadtwerke-mittelfeld.md whatever the input text.
Grading is a keyword rule: a candidate is selected when its text names every technology of the
requirement; that is enough to reject CV-03 (Java, no Python) on section 2.4.
"""

from tenderscout.domain.models import (
    Category,
    Evidence,
    ExtractionResult,
    GradeResult,
    Industry,
    MatchResult,
    Priority,
    ProposalDraft,
    Requirement,
)

STADTWERKE_REQUIREMENTS: list[Requirement] = [
    Requirement(
        id="R1",
        source_section="2.1",
        source_quote="Entwicklung eines RAG-basierten Wissensassistenten in Python (LangChain oder "
        "vergleichbar) mit Vektordatenbank. Jede Antwort muss Quellenangaben enthalten.",
        language="de",
        summary_en="Build a RAG knowledge assistant in Python with LangChain, a vector database "
        "and source citations.",
        category=Category.TECHNICAL,
        priority=Priority.MUST,
        technologies=["python", "langchain"],
        target_doc_types=["project", "cv"],
        search_query_en="RAG knowledge assistant python langchain vector database citations",
    ),
    Requirement(
        id="R2",
        source_section="2.2",
        source_quote="Betrieb in Microsoft Azure, bevorzugt auf AKS. Die gesamte Infrastruktur ist "
        "als Code (Terraform) bereitzustellen, inklusive CI/CD-Pipeline.",
        language="de",
        summary_en="Run on Azure (preferably AKS), all infrastructure as Terraform code, CI/CD.",
        category=Category.TECHNICAL,
        priority=Priority.MUST,
        technologies=["azure", "terraform"],
        target_doc_types=["project", "cv"],
        search_query_en="Azure AKS Kubernetes Terraform infrastructure as code CI/CD pipeline",
    ),
    Requirement(
        id="R3",
        source_section="2.3",
        source_quote="DSGVO-konforme Verarbeitung: Daten verbleiben ausschließlich in "
        "EU-Rechenzentren, personenbezogene Daten werden vor der Indexierung maskiert.",
        language="de",
        summary_en="GDPR-compliant processing: data stays in EU data centres, personal data is "
        "masked before indexing.",
        category=Category.COMPLIANCE,
        priority=Priority.MUST,
        target_doc_types=["project"],
        search_query_en="GDPR EU data residency personal data masking before indexing",
    ),
    Requirement(
        id="R4",
        source_section="2.4",
        source_quote="Das Projektteam umfasst mindestens zwei Entwickler mit jeweils mindestens "
        "drei Jahren Erfahrung mit Python und Kubernetes.",
        language="de",
        summary_en="At least two developers with 3+ years of Python and Kubernetes each.",
        category=Category.STAFFING,
        priority=Priority.SHOULD,
        technologies=["python", "kubernetes"],
        min_years_experience=3,
        target_doc_types=["cv"],
        search_query_en="developer python kubernetes experience",
    ),
    Requirement(
        id="R5",
        source_section="2.5",
        source_quote="Nachweisbare Projekterfahrung in der Energiewirtschaft, idealerweise bei "
        "einem Stadtwerk.",
        language="de",
        summary_en="Proven project experience in the energy sector, ideally a municipal utility.",
        category=Category.DOMAIN,
        priority=Priority.SHOULD,
        industry=Industry.ENERGY,
        target_doc_types=["project"],
        search_query_en="energy sector municipal utility Stadtwerk project",
    ),
    Requirement(
        id="R6",
        source_section="2.6",
        source_quote="Anbindung an das Abrechnungssystem SAP IS-U.",
        language="de",
        summary_en="Integration with the SAP IS-U billing system.",
        category=Category.TECHNICAL,
        priority=Priority.NICE,
        technologies=["sap"],
        target_doc_types=["project", "cv"],
        search_query_en="SAP IS-U billing system integration",
    ),
    Requirement(
        id="R7",
        source_section="4.1",
        source_quote="The bidder has experience evaluating and monitoring LLM applications in "
        "production (tracing, automated quality metrics such as faithfulness).",
        language="en",
        summary_en="Experience evaluating and monitoring LLM applications in production.",
        category=Category.TECHNICAL,
        priority=Priority.SHOULD,
        target_doc_types=["project", "cv"],
        search_query_en="LLM evaluation monitoring tracing faithfulness metrics production",
    ),
]


def _mentions(text: str, tech: str) -> bool:
    return tech in text or tech.replace("-", " ") in text


class FakeLLM:
    async def extract_requirements(self, rfp_text: str) -> ExtractionResult:
        return ExtractionResult(
            requirements=list(STADTWERKE_REQUIREMENTS), reference_window_years=5
        )

    async def grade_candidates(
        self, requirement: Requirement, candidates: list[Evidence]
    ) -> GradeResult:
        if not candidates:
            return GradeResult(status="gap", rationale="no candidates retrieved")
        techs = requirement.technologies
        if not techs:
            return GradeResult(
                status="covered",
                selected_doc_ids=[c.doc_id for c in candidates],
                rationale="no technology constraint; all candidates accepted",
            )

        def text(c: Evidence) -> str:
            return f"{c.title} {c.snippet}".lower()

        full = [c.doc_id for c in candidates if all(_mentions(text(c), t) for t in techs)]
        if full:
            return GradeResult(
                status="covered", selected_doc_ids=full, rationale=f"mentions {', '.join(techs)}"
            )
        some = [c.doc_id for c in candidates if any(_mentions(text(c), t) for t in techs)]
        if some:
            return GradeResult(
                status="partial", selected_doc_ids=some, rationale="only some technologies"
            )
        return GradeResult(status="gap", rationale=f"no candidate mentions {', '.join(techs)}")

    async def write_draft(
        self,
        requirements: list[Requirement],
        matches: dict[str, MatchResult],
        feedback: str | None = None,
    ) -> ProposalDraft:
        lines = ["# Proposal draft", ""]
        cited: list[str] = []
        for req in requirements:
            match = matches.get(req.id)
            status = match.status if match else "gap"
            ids = [e.doc_id for e in match.evidence] if match else []
            cited.extend(i for i in ids if i not in cited)
            refs = " ".join(f"[{i}]" for i in ids)
            lines.append(f"- {req.source_section} ({req.priority}): {status} {refs}".rstrip())
        if feedback:
            lines += ["", f"_Revised after review: {feedback}_"]
        return ProposalDraft(markdown="\n".join(lines), cited_doc_ids=cited)

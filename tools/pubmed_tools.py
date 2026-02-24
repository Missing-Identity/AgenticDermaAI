# PubMedSearchTool: searches PubMed for peer-reviewed medical literature
# using the NCBI Entrez API via Biopython.

import os
import time
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from dotenv import load_dotenv

load_dotenv()

# Biopython's NCBI Entrez interface
from Bio import Entrez

# Configure Entrez with your credentials
Entrez.email = os.getenv("NCBI_EMAIL", "researcher@dermaai.local")
Entrez.api_key = os.getenv("NCBI_API_KEY", "")

# Enforce max 2 searches per research task — LLM often ignores prompt limits
_MAX_PUBMED_CALLS = 2
_pubmed_call_count = 0


def reset_pubmed_call_count() -> None:
    """Reset before each crew run so the research agent gets a fresh limit."""
    global _pubmed_call_count
    _pubmed_call_count = 0


class PubMedSearchInput(BaseModel):
    query: str = Field(
        description=(
            "A short PubMed search query of 2-4 clinical terms. "
            "Lead with the diagnosis or lesion morphology, then add 1-2 modifiers. "
            "Good: 'tinea corporis hand pruritus', 'granuloma annulare', "
            "'erythema annulare centrifugum'. "
            "Bad: long descriptive phrases with 6+ words."
        )
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of articles to retrieve (default 5, max 10)",
        ge=1,
        le=10,
    )
    exclude_pmids: str = Field(
        default="",
        description=(
            "Optional. Comma-separated PMIDs to exclude (e.g. '37433389,18801146'). "
            "Use when doing a secondary search to avoid returning articles already found."
        )
    )


class PubMedSearchTool(BaseTool):
    name: str = "pubmed_search"
    description: str = (
        "Searches PubMed for peer-reviewed dermatology and medical literature. "
        "Returns titles, abstracts, and publication years of relevant articles. "
        "Use this to find evidence-based information about skin conditions, "
        "treatments, and differential diagnoses. "
        "Provide a specific medical query. For a secondary search, pass exclude_pmids "
        "(comma-separated PMIDs from your first search) to avoid duplicates."
    )
    args_schema: type[BaseModel] = PubMedSearchInput

    def _run(self, query: str, max_results: int = 5, exclude_pmids: str = "") -> str:
        """
        Search PubMed and return formatted article summaries.
        Returns a string — the Research Agent will read and interpret this.
        """
        global _pubmed_call_count
        _pubmed_call_count += 1
        if _pubmed_call_count > _MAX_PUBMED_CALLS:
            return (
                f"STOP: Maximum of {_MAX_PUBMED_CALLS} pubmed_search calls reached. "
                "Do not call this tool again. Write your research summary now using the "
                "articles you have already retrieved."
            )

        # Cap at 4 words — PubMed returns no results for over-specified queries
        words = query.split()
        if len(words) > 4:
            query = " ".join(words[:4])

        exclude_set = {
            p.strip() for p in exclude_pmids.split(",") if p and p.strip().isdigit()
        }
        retmax = min(max_results + len(exclude_set), 50)  # fetch extra to allow filtering

        try:
            # Step 1: Search for article IDs
            search_handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=retmax,
                sort="relevance",
                datetype="pdat",
                mindate="2015",   # only articles from 2015 onward for recency
            )
            search_result = Entrez.read(search_handle)
            search_handle.close()

            raw_ids = search_result.get("IdList", [])
            ids = [pid for pid in raw_ids if pid not in exclude_set][:max_results]
            total_found = search_result.get("Count", 0)

            if not ids:
                if raw_ids and exclude_set:
                    return (
                        f"All matching articles for query '{query}' were already retrieved. "
                        f"No new PMIDs to add (excluded: {len(exclude_set)}).\n"
                        "Proceed with your summary using the articles from your first search."
                    )
                return (
                    f"No PubMed articles found for query: '{query}'\n"
                    "Consider broadening the search terms."
                )

            # Step 2: Small delay to respect rate limits
            time.sleep(0.5)

            # Step 3: Fetch article details
            fetch_handle = Entrez.efetch(
                db="pubmed",
                id=",".join(ids),
                rettype="abstract",
                retmode="xml",
            )
            articles = Entrez.read(fetch_handle)
            fetch_handle.close()

            # Step 4: Format results into readable text
            results = [
                f"PubMed Search Results for: '{query}'",
                f"Total articles found: {total_found} (showing {len(ids)})\n",
                "=" * 60,
            ]

            for i, article in enumerate(articles.get("PubmedArticle", []), 1):
                try:
                    citation = article["MedlineCitation"]
                    art_data = citation["Article"]

                    title = str(art_data.get("ArticleTitle", "No title"))
                    pmid = str(citation.get("PMID", "Unknown"))

                    # Get publication year
                    pub_date = art_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
                    year = pub_date.get("Year", pub_date.get("MedlineDate", "Unknown year"))

                    # Get abstract (may be structured with multiple sections)
                    abstract_data = art_data.get("Abstract", {}).get("AbstractText", [])
                    if isinstance(abstract_data, list):
                        abstract = " ".join(str(t) for t in abstract_data)
                    else:
                        abstract = str(abstract_data)

                    # Truncate very long abstracts — key findings are in the first ~500 chars
                    if len(abstract) > 500:
                        abstract = abstract[:500] + "... [truncated]"

                    # Get first author
                    authors = art_data.get("AuthorList", [])
                    first_author = "Unknown author"
                    if authors:
                        a = authors[0]
                        last = a.get("LastName", "")
                        first = a.get("ForeName", "")
                        first_author = f"{last} {first}".strip() or "Unknown author"

                    results.append(f"\n[{i}] PMID: {pmid}")
                    results.append(f"    Title: {title}")
                    results.append(f"    Author: {first_author} et al. ({year})")
                    results.append(f"    Abstract: {abstract}")
                    results.append("-" * 40)

                except (KeyError, IndexError, TypeError) as e:
                    results.append(f"\n[{i}] Error parsing article: {e}")
                    continue

            return "\n".join(results)

        except Exception as e:
            return (
                f"PubMed search error: {str(e)}\n"
                "Check your internet connection and NCBI credentials in .env"
            )
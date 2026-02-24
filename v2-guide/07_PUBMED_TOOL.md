# Chapter 07 — The PubMed Search Tool

**Goal:** Build a custom CrewAI tool that searches PubMed for peer-reviewed medical literature. Test it standalone with real queries before attaching it to an agent. Understand rate limiting and API key setup.

**Time estimate:** 40–50 minutes

---

## What Is PubMed and Why Are We Using It?

PubMed is the US National Library of Medicine's database of over 35 million biomedical citations. It is:
- Peer-reviewed (not Reddit, not social media)
- Free to access via the NCBI Entrez API
- Authoritative for clinical diagnosis support

NCBI provides a Python-friendly API via **Biopython's Entrez module**, which we already installed.

---

## Step 1 — Get Your NCBI API Key

Without an API key: 3 requests per second limit  
With an API key: 10 requests per second limit

For development, both are fine. For production, get the key.

1. Go to: https://www.ncbi.nlm.nih.gov/account/  
2. Create a free account  
3. Go to your account settings → API Key Management  
4. Generate a key  
5. Add it to your `.env` file:
   ```
   NCBI_API_KEY=your_key_here
   NCBI_EMAIL=your@email.com
   ```

> **NCBI requires a real email** in the `Entrez.email` field. They use it to contact you if your queries are causing issues. It is not shared publicly.

---

## Step 2 — Understand the Entrez API Flow

Before writing the tool, understand what Biopython's Entrez does:

```
Your query string
      │
      ▼
  Entrez.esearch()      ← search PubMed, returns a list of article IDs
      │ returns IDs
      ▼
  Entrez.efetch()       ← fetch full records for those IDs
      │ returns XML records
      ▼
  Entrez.read()         ← parse XML into Python dict
      │
      ▼
  Your code extracts:
    - Title
    - Abstract
    - Authors
    - Publication year
    - PubMed ID (PMID)
```

You never download PDFs — just titles and abstracts, which is enough for the research agent to cite findings.

---

## Step 3 — Test Entrez Directly (Before Building the Tool)

Open a Python REPL (`python` in your terminal):

```python
from Bio import Entrez

# Always set your email
Entrez.email = "your@email.com"
Entrez.api_key = "your_key_here"   # or leave out if you don't have one yet

# Search for articles about contact dermatitis
handle = Entrez.esearch(
    db="pubmed",
    term="contact dermatitis forearm pruritus[Title/Abstract]",
    retmax=3,           # only get 3 results for testing
    sort="relevance",
)
record = Entrez.read(handle)
handle.close()

print("Total results found:", record["Count"])
print("IDs returned:", record["IdList"])
```

You should see something like:
```
Total results found: 847
IDs returned: ['38421234', '37892341', '37123456']
```

Now fetch the actual article details:
```python
ids = ",".join(record["IdList"])
fetch_handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
articles = Entrez.read(fetch_handle)
fetch_handle.close()

# Look at the first article
first = articles["PubmedArticle"][0]
print("Title:", first["MedlineCitation"]["Article"]["ArticleTitle"])
print("Abstract:", str(first["MedlineCitation"]["Article"]["Abstract"]["AbstractText"])[:200])
```

**This is the exact flow you're wrapping into a Tool.** Understand it here before abstracting it.

---

## Step 4 — Build the PubMedSearchTool

**`tools/pubmed_tools.py`**

```python
# tools/pubmed_tools.py
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


class PubMedSearchInput(BaseModel):
    query: str = Field(
        description=(
            "A targeted medical search query. "
            "Include condition name, key symptoms, and relevant patient factors. "
            "Example: 'contact dermatitis occupational painter solvent forearm'"
        )
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of articles to retrieve (default 5, max 10)",
        ge=1,
        le=10,
    )


class PubMedSearchTool(BaseTool):
    name: str = "pubmed_search"
    description: str = (
        "Searches PubMed for peer-reviewed dermatology and medical literature. "
        "Returns titles, abstracts, and publication years of relevant articles. "
        "Use this to find evidence-based information about skin conditions, "
        "treatments, and differential diagnoses. "
        "Provide a specific medical query combining diagnosis, symptoms, and patient factors."
    )
    args_schema: type[BaseModel] = PubMedSearchInput

    def _run(self, query: str, max_results: int = 5) -> str:
        """
        Search PubMed and return formatted article summaries.
        Returns a string — the Research Agent will read and interpret this.
        """
        try:
            # Step 1: Search for article IDs
            search_handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=min(max_results, 10),
                sort="relevance",
                datetype="pdat",
                mindate="2015",   # only articles from 2015 onward for recency
            )
            search_result = Entrez.read(search_handle)
            search_handle.close()

            ids = search_result.get("IdList", [])
            total_found = search_result.get("Count", 0)

            if not ids:
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

                    # Truncate very long abstracts
                    if len(abstract) > 800:
                        abstract = abstract[:800] + "... [truncated]"

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
```

---

## Step 5 — Test the Tool Directly

Create **`test_pubmed_tool.py`**:

```python
# test_pubmed_tool.py
from tools.pubmed_tools import PubMedSearchTool

tool = PubMedSearchTool()

# Test 1: Specific condition query
print("\n" + "="*60)
print("TEST 1: Contact dermatitis query")
print("="*60)
result = tool._run(
    query="contact dermatitis occupational chemical exposure forearm",
    max_results=3,
)
print(result)

# Test 2: Broad dermatology query
print("\n" + "="*60)
print("TEST 2: Psoriasis with joint involvement")
print("="*60)
result = tool._run(
    query="psoriasis plaque elbow joint pain psoriatic arthritis",
    max_results=3,
)
print(result)

# Test 3: No results query (should handle gracefully)
print("\n" + "="*60)
print("TEST 3: Nonsense query (should return no results gracefully)")
print("="*60)
result = tool._run(
    query="xyzzy frobozz skin condition 99999",
    max_results=3,
)
print(result)
```

Run it:
```powershell
python test_pubmed_tool.py
```

**What to verify:**

Test 1:
- You get 3 actual article titles and abstracts about contact dermatitis
- PMID numbers are real (you can verify at pubmed.ncbi.nlm.nih.gov/PMID)
- No errors

Test 2:
- Articles discuss psoriasis and joint involvement
- Years are 2015 or later

Test 3:
- Returns the "no results" message gracefully (no crash, no exception)

**If you get a rate limit error:** Add your NCBI API key to `.env`. Without it, 3 queries burst can trigger a limit.

**If you get an SSL error:** Your network may block NCBI. Try on a different connection.

---

## Step 6 — Understanding Query Construction (Critical for the Research Agent)

The quality of PubMed results depends entirely on query construction. Here's how to think about it:

**Bad query:** `"rash"`  
→ Returns 200,000 results, most irrelevant

**Better query:** `"contact dermatitis"`  
→ Returns 15,000 results, mostly relevant

**Good query:** `"contact dermatitis[MeSH] forearm painter solvent"`  
→ Returns focused, relevant results

**Best query from our structured data:**
```python
# The Research Agent will build queries from Decomposition + Lesion outputs:
diagnosis_hint = "contact dermatitis"
symptoms = "pruritus erythema papular"
location = "forearm"
patient_factor = "painter solvent exposure"
query = f"{diagnosis_hint} {symptoms} {location} {patient_factor}"
```

The Research Agent learns this pattern from its backstory and task description (Chapter 08).

---

## Step 7 — Verify Real Articles

Pick one PMID from your Test 1 output. Go to:
```
https://pubmed.ncbi.nlm.nih.gov/YOUR_PMID_HERE/
```

Confirm:
- The title matches what your tool returned
- The abstract matches (or at least the first few sentences)

This verifies your tool is fetching real, accurate data — not hallucinating citations.

---

## Checkpoint ✅

- [ ] `tools/pubmed_tools.py` exists with `PubMedSearchTool`
- [ ] NCBI email is set in `.env` (API key optional but recommended)
- [ ] Test 1 returns real dermatology articles with titles and abstracts
- [ ] Test 2 returns real psoriasis articles
- [ ] Test 3 handles no-results gracefully without crashing
- [ ] You've verified at least one PMID is real on the PubMed website
- [ ] `tools/__init__.py` exists (needed for imports)

Delete the test file:
```powershell
Remove-Item test_pubmed_tool.py
```

---

*Next → `08_RESEARCH_AGENT.md`*

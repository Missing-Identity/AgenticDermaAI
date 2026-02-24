# Chapter 11 — PDF Report Generation

**Goal:** Generate three PDF documents from a completed diagnosis run. The **doctor audit PDF** is generated immediately after every crew run (before approval) and shows the complete agent workflow. The **doctor clinical report** and **patient summary** are generated only after doctor approval.

**Time estimate:** 60–80 minutes

---

## Three PDFs, Three Purposes

| PDF | Generated | Audience | Content |
|-----|-----------|----------|---------|
| `doctor_audit_*.pdf` | After every run (pre-approval) | Doctor reviewing AI work | Full agent-by-agent workflow, raw vision outputs, every intermediate result, all reasoning, feedback history |
| `doctor_report_*.pdf` | Post-approval | Treating physician | Structured clinical report: lesion profile, differentials, treatment protocol, literature, doctor notes |
| `patient_summary_*.pdf` | Post-approval | Patient | 1–2 pages: what they have, what to do, when to worry, disclaimer |

---

## Design Philosophy

**Doctor Audit PDF (the transparency document):**
The doctor must be able to trace *every* inference the AI made. This PDF reads like a scientific lab notebook — each agent's input and output is documented in order. The doctor can spot exactly where an error occurred and give targeted feedback.

**Doctor Clinical Report:**
A clean, structured clinical document. The doctor uses this alongside (not instead of) the audit PDF. Formatted for easy reference during consultation.

**Patient Summary:**
Maximum two pages. No medical jargon that isn't immediately explained. The goal is that a patient reads this once and knows what to do.

---

## Step 1 — Create the PDF Service

Create **`pdf_service.py`** in the project root. This file contains all three generators plus shared utilities.

### Part A — Shared Utilities and Colour Palette

```python
# pdf_service.py
# Generates three PDF reports:
#   1. doctor_audit_*.pdf   — full agent workflow (pre-approval, generated every run)
#   2. doctor_report_*.pdf  — clinical report (post-approval)
#   3. patient_summary_*.pdf — patient-facing (post-approval)

import os
from datetime import datetime
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
    Table, TableStyle, PageBreak,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# ── Colour palette ─────────────────────────────────────────────────────────────
NAVY       = colors.HexColor("#1A2B4A")
TEAL       = colors.HexColor("#0D7377")
AMBER      = colors.HexColor("#E8A838")
SOFT_RED   = colors.HexColor("#D64045")
LIGHT_GREY = colors.HexColor("#F5F5F5")
MID_GREY   = colors.HexColor("#888888")
DARK_GREY  = colors.HexColor("#333333")
WHITE      = colors.white
PALE_BLUE  = colors.HexColor("#EBF5FB")
PALE_AMBER = colors.HexColor("#FEF9E7")


def _severity_colour(severity: str) -> colors.Color:
    return {"Mild": TEAL, "Moderate": AMBER, "Severe": SOFT_RED}.get(severity, TEAL)


def _make_styles() -> dict:
    """Return a dictionary of ParagraphStyles for consistent formatting."""
    return {
        "title":       ParagraphStyle("Title",      fontName="Helvetica-Bold",    fontSize=20, textColor=NAVY,       spaceAfter=4),
        "subtitle":    ParagraphStyle("Subtitle",   fontName="Helvetica",         fontSize=11, textColor=MID_GREY,   spaceAfter=14),
        "section":     ParagraphStyle("Section",    fontName="Helvetica-Bold",    fontSize=12, textColor=NAVY,       spaceBefore=14, spaceAfter=5),
        "subsection":  ParagraphStyle("Subsection", fontName="Helvetica-Bold",    fontSize=10, textColor=TEAL,       spaceBefore=8,  spaceAfter=3),
        "body":        ParagraphStyle("Body",       fontName="Helvetica",         fontSize=9,  textColor=DARK_GREY,  spaceAfter=5,   leading=14),
        "mono":        ParagraphStyle("Mono",       fontName="Courier",           fontSize=8,  textColor=DARK_GREY,  spaceAfter=3,   leading=12),
        "label":       ParagraphStyle("Label",      fontName="Helvetica-Bold",    fontSize=9,  textColor=NAVY),
        "small":       ParagraphStyle("Small",      fontName="Helvetica",         fontSize=7,  textColor=MID_GREY,   spaceAfter=3),
        "disclaimer":  ParagraphStyle("Disclaimer", fontName="Helvetica-Oblique", fontSize=7,  textColor=MID_GREY,   spaceBefore=16),
        "red_flag":    ParagraphStyle("RedFlag",    fontName="Helvetica-Bold",    fontSize=9,  textColor=SOFT_RED),
        "patient_h":   ParagraphStyle("PatientH",   fontName="Helvetica-Bold",    fontSize=14, textColor=NAVY,       spaceBefore=10, spaceAfter=6),
        "patient_body":ParagraphStyle("PatientBody",fontName="Helvetica",         fontSize=11, textColor=DARK_GREY,  spaceAfter=8,   leading=17),
    }


def _header_banner(title_line1: str, title_line2: str, subtitle: str,
                   severity: str, styles: dict) -> Table:
    """Returns a colour-coded two-row header banner as a Table."""
    sev_col = _severity_colour(severity)

    rows = [
        [
            Paragraph(title_line1, ParagraphStyle("H1", fontName="Helvetica-Bold", fontSize=9, textColor=WHITE)),
            Paragraph(f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}", ParagraphStyle("HR", fontName="Helvetica", fontSize=8, textColor=WHITE, alignment=2)),
        ],
        [
            Paragraph(title_line2, ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=17, textColor=WHITE)),
            Paragraph(subtitle, ParagraphStyle("HS", fontName="Helvetica", fontSize=9, textColor=WHITE, alignment=2)),
        ],
        [
            Paragraph(f"Severity: {severity}", ParagraphStyle("HSev", fontName="Helvetica-Bold", fontSize=10, textColor=WHITE)),
            Paragraph("", ParagraphStyle("Empty")),
        ],
    ]

    t = Table(rows, colWidths=[12 * cm, 6 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 1), NAVY),
        ("BACKGROUND",    (0, 2), (-1, 2), sev_col),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
    ]))
    return t


def _two_col_table(data: list[tuple[str, str]], styles: dict) -> Table:
    """Helper to render a two-column label:value table."""
    rows = [
        [Paragraph(label, styles["label"]), Paragraph(str(value), styles["body"])]
        for label, value in data
    ]
    t = Table(rows, colWidths=[5 * cm, 12 * cm])
    t.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, LIGHT_GREY]),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
        ("GRID",           (0, 0), (-1, -1), 0.3, MID_GREY),
    ]))
    return t
```

---

### Part B — Doctor Audit PDF

This is the transparency document. It documents every agent step in sequence.

```python
# ── Doctor Audit PDF ───────────────────────────────────────────────────────────

def generate_doctor_audit_pdf(audit) -> bytes:
    """
    Generate the complete agent workflow audit PDF.
    Called after every crew run, BEFORE doctor approval.

    Sections:
      0. Cover / Run info
      1. Patient Input (text + image path)
      2. Vision Analysis (raw model outputs per dimension)
      3. Lesion Agents (each pydantic output + reason)
      4. Symptom Decomposition
      5. Research Findings (queries, articles, key findings, PMIDs)
      6. Differential Diagnosis (primary + each differential with FOR/AGAINST)
      7. Treatment Plan (full tiered protocol)
      8. Orchestrator Synthesis (final diagnosis + reasoning + re-diagnosis info)
      9. Doctor Feedback History (all previous rounds)
    """
    buffer = BytesIO()
    styles = _make_styles()

    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    story = []

    diag_name = audit.final_diagnosis.primary_diagnosis if audit.final_diagnosis else "Pending"
    severity  = audit.final_diagnosis.severity if audit.final_diagnosis else "Unknown"

    # ── Cover ──────────────────────────────────────────────────────────────────
    story.append(_header_banner(
        "DermaAI v2 — Doctor Audit Report",
        diag_name,
        f"Run #{audit.run_count}  |  PENDING DOCTOR APPROVAL",
        severity, styles,
    ))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph(
        "This document shows the complete step-by-step reasoning of every AI agent "
        "that contributed to this diagnosis. Review each section carefully before "
        "approving or rejecting the diagnosis.",
        styles["body"],
    ))
    story.append(Spacer(1, 0.3*cm))

    # ── Section 1: Patient Input ───────────────────────────────────────────────
    story.append(HRFlowable(color=NAVY, thickness=1.5))
    story.append(Paragraph("1. Patient Input", styles["section"]))

    story.append(_two_col_table([
        ("Image Path",  audit.image_path or "None provided"),
        ("Symptom Text", audit.patient_text),
    ], styles))

    # ── Section 2: Vision Analysis ─────────────────────────────────────────────
    if audit.image_path:
        story.append(Spacer(1, 0.3*cm))
        story.append(HRFlowable(color=TEAL, thickness=1))
        story.append(Paragraph("2. Vision Analysis — Raw Model Outputs", styles["section"]))
        story.append(Paragraph(
            "The vision model (medgemma) was called 4 times with focused clinical prompts. "
            "These are its verbatim outputs before any clinical agent processed them.",
            styles["body"],
        ))

        for label, raw_text in [
            ("Colour",    audit.vision_colour_raw),
            ("Texture",   audit.vision_texture_raw),
            ("Levelling", audit.vision_levelling_raw),
            ("Shape/Border", audit.vision_shape_raw),
        ]:
            story.append(Paragraph(label, styles["subsection"]))
            story.append(Paragraph(raw_text or "No output", styles["mono"]))
            story.append(Spacer(1, 0.1*cm))

    # ── Section 3: Lesion Agent Outputs ───────────────────────────────────────
    story.append(HRFlowable(color=TEAL, thickness=1))
    story.append(Paragraph("3. Lesion Agent Clinical Interpretations", styles["section"]))

    lesion_rows = [["Dimension", "Assessment", "Clinical Reasoning"]]
    if audit.colour_output:
        lesion_rows.append(["Colour", audit.colour_output.lesion_colour, audit.colour_output.reason])
    if audit.texture_output:
        lesion_rows.append(["Surface", audit.texture_output.surface, audit.texture_output.reason])
    if audit.levelling_output:
        lesion_rows.append(["Elevation", audit.levelling_output.levelling, audit.levelling_output.reason])
    if audit.shape_output:
        shape_val = getattr(audit.shape_output, "shape_border", getattr(audit.shape_output, "shape_border", "N/A"))
        lesion_rows.append(["Border", str(shape_val), audit.shape_output.reason])

    if len(lesion_rows) > 1:
        lt = Table(lesion_rows, colWidths=[3.5*cm, 4*cm, 9.5*cm])
        lt.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), TEAL),
            ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("GRID",          (0, 0), (-1, -1), 0.3, MID_GREY),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(lt)

    # ── Section 4: Symptom Decomposition ─────────────────────────────────────
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(color=TEAL, thickness=1))
    story.append(Paragraph("4. Symptom Decomposition", styles["section"]))

    if audit.decomposition_output:
        d = audit.decomposition_output
        story.append(_two_col_table([
            ("Symptoms",     ", ".join(d.symptoms) if d.symptoms else "None identified"),
            ("Duration",     f"{d.time_days} days" if d.time_days else "Not specified"),
            ("Onset",        d.onset or "Unknown"),
            ("Progression",  d.progression or "Unknown"),
            ("Location",     d.body_location or "Not specified"),
            ("Aggravating",  ", ".join(d.aggravating_factors) if d.aggravating_factors else "None"),
            ("Relieving",    ", ".join(d.relieving_factors) if d.relieving_factors else "None"),
            ("Occupational", d.occupational_exposure or "None"),
            ("Associated",   ", ".join(d.associated_symptoms) if d.associated_symptoms else "None"),
            ("Prior Tx",     ", ".join(d.prior_treatments) if d.prior_treatments else "None"),
            ("Patient words", d.patient_description),
        ], styles))
    else:
        story.append(Paragraph("No decomposition data available.", styles["body"]))

    # ── Section 5: Research Findings ──────────────────────────────────────────
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(color=TEAL, thickness=1))
    story.append(Paragraph("5. Research Agent Findings", styles["section"]))

    if audit.research_output:
        r = audit.research_output
        story.append(_two_col_table([
            ("Primary Query",   r.primary_search_query),
            ("Secondary Query", r.secondary_search_query or "None"),
            ("Articles Found",  str(r.articles_found)),
            ("Evidence Level",  r.evidence_strength),
            ("PMIDs",          ", ".join(r.cited_pmids)),
        ], styles))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("Key Findings from Literature:", styles["subsection"]))
        for finding in r.key_findings:
            story.append(Paragraph(f"• {finding}", styles["body"]))
        if r.contradicted_findings:
            story.append(Paragraph("Contradictions / Flags:", styles["subsection"]))
            for flag in r.contradicted_findings:
                story.append(Paragraph(f"⚠ {flag}", styles["red_flag"]))
        if r.research_notes:
            story.append(Paragraph(f"Research notes: {r.research_notes}", styles["body"]))

    # ── Section 6: Differential Diagnosis ─────────────────────────────────────
    story.append(PageBreak())
    story.append(HRFlowable(color=NAVY, thickness=1.5))
    story.append(Paragraph("6. Differential Diagnosis", styles["section"]))

    if audit.differential_output:
        diff = audit.differential_output
        story.append(_two_col_table([
            ("Primary Diagnosis",  diff.primary_diagnosis),
            ("Confidence",         diff.confidence_in_primary),
        ], styles))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("Primary Reasoning:", styles["subsection"]))
        story.append(Paragraph(diff.primary_reasoning, styles["body"]))

        if diff.red_flags:
            story.append(Spacer(1, 0.2*cm))
            story.append(Paragraph("⚠ Red Flags Present:", styles["red_flag"]))
            for flag in diff.red_flags:
                story.append(Paragraph(f"  • {flag}", styles["red_flag"]))
            story.append(Paragraph(
                f"Urgent referral required: {'YES' if diff.requires_urgent_referral else 'No'}",
                styles["red_flag"] if diff.requires_urgent_referral else styles["body"],
            ))

        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph(f"Differential Diagnoses ({len(diff.differentials)}):", styles["subsection"]))

        for entry in diff.differentials:
            prob_col = {"high": AMBER, "moderate": TEAL, "low": MID_GREY}.get(entry.probability, TEAL)
            entry_data = [
                [Paragraph(f"[{entry.probability.upper()}] {entry.condition}", ParagraphStyle("EH", fontName="Helvetica-Bold", fontSize=9, textColor=prob_col)), ""],
                [Paragraph("Features FOR:", styles["label"]),
                 Paragraph(", ".join(entry.key_features_matching), styles["body"])],
                [Paragraph("Features AGAINST:", styles["label"]),
                 Paragraph(", ".join(entry.key_features_against), styles["body"])],
                [Paragraph("Distinguishing test:", styles["label"]),
                 Paragraph(entry.distinguishing_test, styles["body"])],
                [Paragraph("Reasoning:", styles["label"]),
                 Paragraph(entry.clinical_reasoning, styles["body"])],
            ]
            et = Table(entry_data, colWidths=[4.5*cm, 12.5*cm])
            et.setStyle(TableStyle([
                ("SPAN",          (0, 0), (-1, 0)),
                ("BACKGROUND",    (0, 0), (-1, 0), PALE_BLUE),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
                ("FONTSIZE",      (0, 0), (-1, -1), 8),
                ("TOPPADDING",    (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING",   (0, 0), (-1, -1), 6),
                ("GRID",          (0, 0), (-1, -1), 0.3, MID_GREY),
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(et)
            story.append(Spacer(1, 0.2*cm))

    # ── Section 7: Treatment Plan ──────────────────────────────────────────────
    story.append(HRFlowable(color=NAVY, thickness=1.5))
    story.append(Paragraph("7. Treatment Plan", styles["section"]))

    if audit.treatment_output:
        t = audit.treatment_output
        story.append(_two_col_table([
            ("For Diagnosis",    t.for_diagnosis),
            ("Evidence Level",   t.evidence_level),
            ("Referral Needed",  "YES — " + t.referral_to if t.referral_needed else "No"),
            ("Follow-up",        t.follow_up),
        ], styles))

        if t.immediate_actions:
            story.append(Spacer(1, 0.2*cm))
            story.append(Paragraph("Immediate Actions:", styles["subsection"]))
            for action in t.immediate_actions:
                story.append(Paragraph(f"• {action}", styles["body"]))

        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("Medication Protocol:", styles["subsection"]))
        med_rows = [["Line", "Treatment", "Dose / Protocol", "Duration", "Monitoring"]]
        for m in t.medications:
            med_rows.append([m.line.upper(), m.treatment_name, m.dose_or_protocol, m.duration, m.monitoring or "—"])
        mt = Table(med_rows, colWidths=[1.5*cm, 4*cm, 5*cm, 3*cm, 3.5*cm])
        mt.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
            ("GRID",          (0, 0), (-1, -1), 0.3, MID_GREY),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(mt)

        if t.non_pharmacological:
            story.append(Spacer(1, 0.2*cm))
            story.append(Paragraph("Non-Pharmacological Interventions:", styles["subsection"]))
            for item in t.non_pharmacological:
                story.append(Paragraph(f"• {item}", styles["body"]))

        if t.contraindications:
            story.append(Spacer(1, 0.15*cm))
            story.append(Paragraph("Patient-Specific Contraindications:", styles["subsection"]))
            for c in t.contraindications:
                story.append(Paragraph(f"⚠ {c}", styles["red_flag"]))

    # ── Section 8: Orchestrator Synthesis ─────────────────────────────────────
    story.append(PageBreak())
    story.append(HRFlowable(color=NAVY, thickness=1.5))
    story.append(Paragraph("8. Orchestrator Final Synthesis", styles["section"]))

    if audit.final_diagnosis:
        fd = audit.final_diagnosis
        story.append(_two_col_table([
            ("Primary Diagnosis",  fd.primary_diagnosis),
            ("Confidence",         fd.confidence),
            ("Severity",           fd.severity),
            ("Re-diagnosis",       "YES — " + fd.re_diagnosis_reason if fd.re_diagnosis_applied else "No revision required"),
        ], styles))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("Clinical Reasoning:", styles["subsection"]))
        story.append(Paragraph(fd.clinical_reasoning, styles["body"]))
        story.append(Spacer(1, 0.1*cm))
        story.append(Paragraph("Doctor Notes:", styles["subsection"]))
        story.append(Paragraph(fd.doctor_notes, styles["body"]))
        story.append(Spacer(1, 0.1*cm))
        story.append(Paragraph("Literature Support:", styles["subsection"]))
        story.append(Paragraph(fd.literature_support, styles["body"]))
        story.append(Paragraph(f"Cited PMIDs: {', '.join(fd.cited_pmids)}", styles["small"]))

    # ── Section 9: Doctor Feedback History ────────────────────────────────────
    if audit.feedback_history:
        story.append(Spacer(1, 0.3*cm))
        story.append(HRFlowable(color=AMBER, thickness=1.5))
        story.append(Paragraph("9. Doctor Feedback History", styles["section"]))

        for entry in audit.feedback_history:
            action = entry.get("action", "rejected")
            if action == "approved":
                story.append(Paragraph(
                    f"Round {entry['round']}: APPROVED",
                    ParagraphStyle("Approved", fontName="Helvetica-Bold", fontSize=9, textColor=TEAL),
                ))
            else:
                story.append(Paragraph(
                    f"Round {entry['round']}: REJECTED — Scope: {entry.get('rerun_scope', 'unknown')}",
                    ParagraphStyle("Rejected", fontName="Helvetica-Bold", fontSize=9, textColor=AMBER),
                ))
                story.append(Paragraph(f"Feedback: {entry.get('feedback', '')}", styles["body"]))
            story.append(Spacer(1, 0.1*cm))

    doc.build(story)
    return buffer.getvalue()


def save_doctor_audit_pdf(audit, patient_name: str = "Patient") -> str:
    """Save the audit PDF and return its path."""
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = patient_name.replace(" ", "_")
    path = f"reports/doctor_audit_{safe_name}_{timestamp}.pdf"
    with open(path, "wb") as f:
        f.write(generate_doctor_audit_pdf(audit))
    return path
```

---

### Part C — Doctor Clinical Report (Post-Approval)

This is a clean, structured document for use during consultation.

```python
# ── Doctor Clinical Report (post-approval) ─────────────────────────────────────

def generate_doctor_pdf(result, audit=None, patient_name: str = "Patient") -> bytes:
    """
    Structured clinical report for the treating physician.
    Generated only after doctor approval.
    """
    buffer = BytesIO()
    styles = _make_styles()

    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    story = []

    story.append(_header_banner(
        "DermaAI v2 — Doctor Clinical Report  ✓ APPROVED",
        result.primary_diagnosis,
        f"Patient: {patient_name}",
        result.severity, styles,
    ))
    story.append(Spacer(1, 0.4*cm))

    # Lesion Profile
    story.append(Paragraph("Lesion Profile", styles["section"]))
    story.append(_two_col_table([
        ("Colour",    result.lesion_profile.get("colour", "N/A")),
        ("Texture",   result.lesion_profile.get("texture", "N/A")),
        ("Levelling", result.lesion_profile.get("levelling", "N/A")),
        ("Border",    result.lesion_profile.get("border", "N/A")),
    ], styles))

    # Differentials from the audit trail if available, else from FinalDiagnosis
    story.append(Paragraph("Differential Diagnoses", styles["section"]))
    if audit and audit.differential_output:
        for entry in audit.differential_output.differentials:
            story.append(Paragraph(
                f"[{entry.probability.upper()}] {entry.condition} — {entry.distinguishing_test}",
                styles["body"],
            ))
    else:
        for dx in result.differential_diagnoses:
            story.append(Paragraph(f"• {dx}", styles["body"]))

    if result.re_diagnosis_applied:
        story.append(Spacer(1, 0.2*cm))
        story.append(HRFlowable(color=AMBER, thickness=1))
        story.append(Paragraph("Re-Diagnosis Applied", styles["subsection"]))
        story.append(Paragraph(result.re_diagnosis_reason, styles["body"]))
        story.append(HRFlowable(color=AMBER, thickness=1))

    # Clinical Reasoning
    story.append(Paragraph("Clinical Reasoning", styles["section"]))
    story.append(Paragraph(result.clinical_reasoning, styles["body"]))

    # Doctor Notes
    story.append(Paragraph("Clinical Notes", styles["section"]))
    story.append(Paragraph(result.doctor_notes, styles["body"]))

    # Investigations + Treatment
    story.append(Paragraph("Suggested Investigations", styles["section"]))
    for inv in result.suggested_investigations:
        story.append(Paragraph(f"• {inv}", styles["body"]))

    story.append(Paragraph("Treatment Protocol", styles["section"]))
    if audit and audit.treatment_output:
        for m in audit.treatment_output.medications:
            story.append(Paragraph(
                f"[{m.line.upper()}] {m.treatment_name} — {m.dose_or_protocol} for {m.duration}",
                styles["body"],
            ))
    else:
        for i, tx in enumerate(result.treatment_suggestions, 1):
            story.append(Paragraph(f"{i}. {tx}", styles["body"]))

    # Evidence
    story.append(Paragraph("Evidence Base", styles["section"]))
    story.append(Paragraph(result.literature_support, styles["body"]))
    story.append(Paragraph(f"PMIDs: {', '.join(result.cited_pmids)}", styles["small"]))

    # Disclaimer
    story.append(HRFlowable(color=MID_GREY, thickness=0.5))
    story.append(Paragraph(result.disclaimer, styles["disclaimer"]))

    doc.build(story)
    return buffer.getvalue()
```

---

### Part D — Patient Summary PDF (Post-Approval, 1–2 Pages)

```python
# ── Patient Summary PDF (post-approval) ────────────────────────────────────────

def generate_patient_pdf(result, patient_name: str = "Patient") -> bytes:
    """
    1–2 page plain-language summary for the patient.
    Generated only after doctor approval.
    No medical jargon without immediate explanation.
    """
    buffer = BytesIO()
    styles = _make_styles()

    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    story = []

    # Clean header — no clinical codes, no confidence percentages
    story.append(_header_banner(
        "DermaAI v2 — Your Skin Health Summary",
        result.primary_diagnosis,
        f"For: {patient_name}",
        result.severity, styles,
    ))
    story.append(Spacer(1, 0.5*cm))

    # What does this mean?
    story.append(Paragraph("What Does This Mean?", styles["patient_h"]))
    story.append(Paragraph(result.patient_summary, styles["patient_body"]))
    story.append(Spacer(1, 0.3*cm))

    # How serious is it?
    sev_icons = {"Mild": "🟢 MILD", "Moderate": "🟡 MODERATE", "Severe": "🔴 SEVERE"}
    sev_text = sev_icons.get(result.severity, result.severity)
    sev_col = _severity_colour(result.severity)
    story.append(HRFlowable(color=sev_col, thickness=2.5))
    story.append(Spacer(1, 0.1*cm))
    story.append(Paragraph(
        f"Assessed Severity: {sev_text}",
        ParagraphStyle("SevLabel", fontName="Helvetica-Bold", fontSize=13, textColor=sev_col, spaceAfter=6),
    ))
    story.append(HRFlowable(color=sev_col, thickness=2.5))
    story.append(Spacer(1, 0.4*cm))

    # What you should do
    story.append(Paragraph("What Should You Do?", styles["patient_h"]))
    for i, rec in enumerate(result.patient_recommendations, 1):
        story.append(Paragraph(
            f"{i}.  {rec}",
            ParagraphStyle("Rec", fontName="Helvetica", fontSize=11, textColor=DARK_GREY, spaceAfter=7, leading=16, leftIndent=10),
        ))
    story.append(Spacer(1, 0.4*cm))

    # When to seek care — always prominently placed
    story.append(HRFlowable(color=SOFT_RED, thickness=2.5))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph("When to Seek Urgent Medical Care", styles["patient_h"]))
    story.append(Paragraph(result.when_to_seek_care, styles["patient_body"]))
    story.append(HRFlowable(color=SOFT_RED, thickness=2.5))
    story.append(Spacer(1, 0.4*cm))

    # Disclaimer — smaller, at the bottom
    story.append(HRFlowable(color=MID_GREY, thickness=0.5))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(result.disclaimer, styles["disclaimer"]))
    story.append(Spacer(1, 0.1*cm))
    story.append(Paragraph(
        f"Doctor-approved on {datetime.now().strftime('%d %B %Y')}   |   Generated by DermaAI v2",
        styles["small"],
    ))

    doc.build(story)
    return buffer.getvalue()


# ── Convenience save functions ─────────────────────────────────────────────────

def save_reports(result, audit=None, patient_name: str = "Patient") -> tuple[str, str]:
    """
    Generate and save the doctor clinical report + patient summary PDF.
    Call this ONLY after doctor approval.
    Returns: (doctor_pdf_path, patient_pdf_path)
    """
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = patient_name.replace(" ", "_")

    doctor_path  = f"reports/doctor_report_{safe_name}_{timestamp}.pdf"
    patient_path = f"reports/patient_summary_{safe_name}_{timestamp}.pdf"

    with open(doctor_path, "wb") as f:
        f.write(generate_doctor_pdf(result, audit, patient_name))

    with open(patient_path, "wb") as f:
        f.write(generate_patient_pdf(result, patient_name))

    return doctor_path, patient_path
```

---

## Step 2 — Test the PDF Generator

Create **`test_pdf.py`** to test all three PDFs without running the full pipeline:

```python
# test_pdf.py
# Test PDF generation using mock data — no agents, no LLM required.

from agents.orchestrator_agent import FinalDiagnosis
from agents.clinical_agents import DifferentialDiagnosisOutput, DifferentialEntry, TreatmentPlanOutput, TreatmentEntry
from agents.decomposition_agent import DecompositionOutput
from agents.research_agent import ResearchSummary
from agents.lesion_agents import ColourOutput, SurfaceOutput, LevellingOutput, ShapeBorderOutput
from audit_trail import AuditTrail
from pdf_service import save_doctor_audit_pdf, save_reports

# Build a mock AuditTrail
audit = AuditTrail(
    patient_text="I have a red, itchy, bumpy rash on my left forearm for 4 days. I'm a painter.",
    image_path=r"C:\path\to\your\test_image.jpg",
    vision_colour_raw="The lesion appears erythematous with a slightly violaceous tinge along the borders. The surrounding skin is medium-toned and unaffected.",
    vision_texture_raw="Surface shows multiple small papules with fine peripheral scaling. The central area appears slightly crusted.",
    vision_levelling_raw="Shadows at the lesion margin suggest elevation above surrounding skin. Central dome shape visible.",
    vision_shape_raw="The border is irregular with notched edges in the superior aspect. The boundary with surrounding skin is not sharply defined.",
    biodata_summary="Name: Ravi Kumar | Age: 34 | Sex: Male | Skin Tone: medium | Occupation: Painter",
    run_count=1,
)

audit.colour_output    = ColourOutput(lesion_colour="Erythematous with violaceous borders", reason="On medium skin tone, the inflammation appears as darker red-violet tones.")
audit.texture_output   = SurfaceOutput(surface="scaly", reason="Fine peripheral scaling with central papular eruption, consistent with chronic inflammation.")
audit.levelling_output = LevellingOutput(levelling="raised", reason="Shadow at margins and dome shape confirm elevation above surrounding skin.")

audit.decomposition_output = DecompositionOutput(
    symptoms=["pruritus", "erythema", "papular eruption"],
    time_days=4, onset="sudden", progression="spreading",
    body_location="left forearm", aggravating_factors=["evenings", "chemical exposure"],
    occupational_exposure="painter — new solvent brand",
    patient_description="Red bumpy rash on my arm",
)

audit.research_output = ResearchSummary(
    primary_search_query="contact dermatitis occupational painter solvent forearm",
    articles_found=47, evidence_strength="strong",
    key_findings=[
        "Solvent-based paints are a leading cause of allergic contact dermatitis in construction workers.",
        "Patch testing identifies the causal allergen in 84% of occupational ACD cases.",
        "Mid-potency topical corticosteroids are first-line treatment for moderate ACD.",
    ],
    supported_diagnoses=["Allergic Contact Dermatitis", "Irritant Contact Dermatitis"],
    cited_pmids=["38421234", "37892341"],
    research_notes="Strong evidence base for occupational ACD in painters.",
)

audit.differential_output = DifferentialDiagnosisOutput(
    primary_diagnosis="Allergic Contact Dermatitis",
    confidence_in_primary="high",
    primary_reasoning="Erythematous papular rash with irregular borders on forearm of painter following solvent brand change is a classic type IV hypersensitivity presentation.",
    differentials=[
        DifferentialEntry(
            condition="Irritant Contact Dermatitis",
            probability="moderate",
            key_features_matching=["chemical exposure", "forearm location", "erythema"],
            key_features_against=["delayed onset 4 days suggests sensitisation not direct irritation", "papular morphology favours ACD"],
            distinguishing_test="Patch testing — positive result confirms ACD over ICD",
            clinical_reasoning="ICD typically presents within hours of exposure; the 4-day onset favours ACD.",
        ),
        DifferentialEntry(
            condition="Atopic Dermatitis",
            probability="low",
            key_features_matching=["pruritus", "papular eruption"],
            key_features_against=["no personal or family atopic history", "clear occupational trigger"],
            distinguishing_test="Serum IgE and RAST panel",
            clinical_reasoning="Atopic dermatitis is less likely given the clear occupational trigger and no atopic history.",
        ),
    ],
    red_flags=[],
    requires_urgent_referral=False,
)

audit.treatment_output = TreatmentPlanOutput(
    for_diagnosis="Allergic Contact Dermatitis",
    immediate_actions=["Stop using the new solvent brand", "Wash affected area with mild soap and water", "Wear nitrile gloves at work"],
    medications=[
        TreatmentEntry(line="first", treatment_name="Hydrocortisone 1% cream", dose_or_protocol="Apply thin layer to affected area", duration="7 days", monitoring="Check for skin thinning"),
        TreatmentEntry(line="second", treatment_name="Betamethasone 0.1% cream", dose_or_protocol="Apply twice daily", duration="7-14 days", monitoring="Monitor for skin atrophy"),
        TreatmentEntry(line="adjunct", treatment_name="Cetirizine 10mg", dose_or_protocol="Once daily orally", duration="Until itch resolves", monitoring="Sedation in elderly"),
    ],
    non_pharmacological=["Fragrance-free emollient twice daily", "Avoid known chemical triggers", "Occupational health referral"],
    patient_instructions="Avoid the solvent, apply the cream as directed, take the antihistamine tablet once a day for the itch. Keep the area moisturised. See your doctor if no better in a week.",
    follow_up="Review in 1 week. If no improvement: patch testing + referral to dermatologist.",
    referral_needed=False, referral_to="",
    contraindications=["No known allergies noted in biodata"],
    evidence_level="strong",
)

mock_result = FinalDiagnosis(
    primary_diagnosis="Allergic Contact Dermatitis",
    confidence="high",
    differential_diagnoses=["Irritant Contact Dermatitis", "Atopic Dermatitis"],
    severity="Moderate",
    lesion_profile={"colour": "Erythematous with violaceous borders", "texture": "Papular with fine scaling", "levelling": "Raised", "border": "Irregular, not well-defined"},
    clinical_reasoning="Erythematous papular rash on forearm of painter following solvent brand change with 4-day delayed onset strongly supports allergic contact dermatitis.",
    re_diagnosis_applied=False, re_diagnosis_reason="",
    patient_summary="You have an allergic skin reaction caused by a chemical in your workplace. Your skin is reacting to the new solvent you started using. Once you avoid it, your skin should heal in 1-2 weeks.",
    patient_recommendations=["Avoid the new solvent brand immediately.", "Apply hydrocortisone cream twice daily for 7 days.", "Take cetirizine tablet once daily for itch.", "See a dermatologist in 1 week if no improvement."],
    doctor_notes="Occupational ACD. Recommend patch testing to identify allergen. Consider mid-potency topical corticosteroid if OTC hydrocortisone insufficient.",
    suggested_investigations=["Patch testing", "Skin swab if secondary infection suspected"],
    treatment_suggestions=["Topical corticosteroid (hydrocortisone 1%)", "Oral antihistamine", "Emollient"],
    literature_support="Strong PubMed evidence for ACD in painters (PMIDs: 38421234, 37892341).",
    cited_pmids=["38421234", "37892341"],
    when_to_seek_care="Go to A&E if rash spreads to face or you develop breathing difficulty. See GP within 2 days if rash spreads or fever develops.",
)
audit.final_diagnosis = mock_result

# Add some mock feedback history
audit.feedback_history = [
    {"round": 1, "feedback": "The texture assessment should mention crusting more prominently.", "rerun_scope": "orchestrator_only"},
    {"round": 2, "action": "approved", "feedback": ""},
]
audit.run_count = 2

# Generate all three PDFs
audit_path = save_doctor_audit_pdf(audit, "Test Patient")
doctor_path, patient_path = save_reports(mock_result, audit, "Test Patient")

print(f"✅ Doctor audit:   {audit_path}")
print(f"✅ Doctor report:  {doctor_path}")
print(f"✅ Patient summary:{patient_path}")
print("\nOpen the three PDFs and review them.")
```

Run it:
```powershell
python test_pdf.py
```

**Doctor Audit PDF checklist:**
- [ ] 9 numbered sections, each clearly labelled
- [ ] Raw vision outputs visible verbatim (Section 2)
- [ ] Lesion agent table shows assessment + reasoning per dimension (Section 3)
- [ ] Each differential has FOR / AGAINST / distinguishing test (Section 6)
- [ ] Medication table shows all three lines (Section 7)
- [ ] Feedback history shows the two rounds (Section 9)

**Doctor Clinical Report checklist:**
- [ ] Single structured document, clean layout
- [ ] Differential diagnoses list with probability tags
- [ ] Treatment protocol from the specialist agent

**Patient Summary checklist:**
- [ ] Fits comfortably on 1–2 pages
- [ ] No unexplained medical jargon
- [ ] "When to Seek Care" section is visually prominent (red rule)
- [ ] "Doctor-approved" stamp at the bottom

**Delete `test_pdf.py` when all three PDFs look correct.**

---

## Checkpoint ✅

- [ ] `pdf_service.py` exists with `generate_doctor_audit_pdf`, `generate_doctor_pdf`, `generate_patient_pdf`, `save_doctor_audit_pdf`, `save_reports`
- [ ] `save_reports()` takes both `result` and `audit` parameters
- [ ] Doctor audit PDF has 9 sections showing every agent's work
- [ ] Doctor audit PDF includes raw vision model output (not just interpreted results)
- [ ] Patient PDF is ≤ 2 pages with no medical jargon requiring a dictionary
- [ ] All three PDFs open and display correctly in a PDF viewer
- [ ] `main.py` calls `save_doctor_audit_pdf(audit)` after each run (pre-approval) and `save_reports(result, audit)` after approval

---

*Next → `12_DIFFERENTIAL_TREATMENT_AGENTS.md`*

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
            ("Border",    getattr(audit, "vision_border_raw", None) or audit.vision_shape_raw),
            ("Shape",     audit.vision_shape_raw),
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
    border_output = getattr(audit, "border_output", None)
    if border_output:
        lesion_rows.append(["Border", border_output.border, border_output.reason])
    if audit.shape_output:
        lesion_rows.append(["Shape", audit.shape_output.shape, audit.shape_output.reason])

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

    if getattr(audit, "adapter_status", None):
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("Schema Adapter Status", styles["subsection"]))
        for key, status in audit.adapter_status.items():
            err = getattr(audit, "adapter_errors", {}).get(key, "")
            text = f"{key}: {status}"
            if err:
                text += f" | error: {err[:160]}"
            story.append(Paragraph(text, styles["small"]))

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
            ("Location",     ", ".join(d.body_location) if d.body_location else "Not specified"),
            ("Aggravating",  ", ".join(d.aggravating_factors) if d.aggravating_factors else "None"),
            ("Relieving",    ", ".join(d.relieving_factors) if d.relieving_factors else "None"),
            ("Occupational", ", ".join(d.occupational_exposure) if d.occupational_exposure else "None"),
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

    # ── Section 6.5: Mimic Resolution ─────────────────────────────────────────
    if audit.mimic_resolution_output:
        mimic = audit.mimic_resolution_output
        story.append(Spacer(1, 0.3*cm))
        story.append(HRFlowable(color=TEAL, thickness=1))
        story.append(Paragraph("6.5. Mimic Resolution Analysis", styles["section"]))
        
        story.append(_two_col_table([
            ("Confirmed Primary", mimic.primary_diagnosis_confirmed),
            ("Rejected Mimic", mimic.rejected_mimic),
            ("Distinguishing Factor", mimic.distinguishing_factor),
        ], styles))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("Mimic Resolution Reasoning:", styles["subsection"]))
        story.append(Paragraph(mimic.mimic_reasoning, styles["body"]))
        story.append(Spacer(1, 0.3*cm))

    # ── Section 6.6: Visual Debate Resolver ───────────────────────────────────
    if getattr(audit, "visual_differential_review_output", None):
        vdr = audit.visual_differential_review_output
        story.append(Spacer(1, 0.3*cm))
        story.append(HRFlowable(color=TEAL, thickness=1))
        story.append(Paragraph("6.6. Visual Debate Resolver (MedGemma Image Arbitration)", styles["section"]))

        # Support both DebateResolverOutput (new) and VisualDifferentialReviewOutput (legacy)
        confirmed = getattr(vdr, "confirmed_diagnosis", None) or getattr(vdr, "visual_winner", "")
        visual_conf = getattr(vdr, "visual_confidence", "N/A")
        visual_reason = getattr(vdr, "visual_reasoning", None) or getattr(vdr, "visual_reasoning_summary", "")
        candidates = getattr(vdr, "candidates_considered", [])

        table_data = [("Confirmed Diagnosis", confirmed)]
        if visual_conf and visual_conf != "N/A":
            table_data.append(("Confidence", visual_conf))
        if candidates:
            table_data.append(("Candidates Considered", ", ".join(candidates)))
        story.append(_two_col_table(table_data, styles))

        if visual_reason:
            story.append(Spacer(1, 0.2*cm))
            story.append(Paragraph("Visual Reasoning:", styles["subsection"]))
            story.append(Paragraph(visual_reason, styles["body"]))

        # Legacy fields (VisualDifferentialReviewOutput only)
        decisive_features = getattr(vdr, "decisive_features", [])
        if decisive_features:
            story.append(Spacer(1, 0.15*cm))
            story.append(Paragraph("Decisive Visual Features:", styles["subsection"]))
            for feat in decisive_features:
                story.append(Paragraph(f"• {feat}", styles["body"]))

        votes = getattr(vdr, "votes", [])
        if votes:
            story.append(Spacer(1, 0.2*cm))
            story.append(Paragraph("Per-Candidate Visual Votes:", styles["subsection"]))
            vote_rows = [["Candidate", "Consistent?", "Confidence", "Visual Reasoning"]]
            for vote in votes:
                vote_rows.append([
                    vote.condition,
                    "YES" if vote.visually_consistent else "NO",
                    vote.confidence,
                    vote.visual_reasoning,
                ])
            vt = Table(vote_rows, colWidths=[4*cm, 2*cm, 2.5*cm, 8.5*cm])
            vt.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0), TEAL),
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
            story.append(vt)
        story.append(Spacer(1, 0.3*cm))

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

    # ── Section 8: CMO Synthesis & Scribe Output ──────────────────────────────
    story.append(PageBreak())
    story.append(HRFlowable(color=NAVY, thickness=1.5))
    story.append(Paragraph("8. CMO Synthesis & Scribe Output", styles["section"]))

    if audit.cmo_output:
        cmo = audit.cmo_output
        story.append(_two_col_table([
            ("CMO Confirmed Diagnosis",  cmo.primary_diagnosis),
            ("Confidence",         cmo.confidence),
            ("Severity",           cmo.severity),
            ("Re-diagnosis",       "YES — " + cmo.re_diagnosis_reason if cmo.re_diagnosis_applied else "No revision required"),
        ], styles))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("CMO Clinical Reasoning:", styles["subsection"]))
        story.append(Paragraph(cmo.clinical_reasoning, styles["body"]))

    if audit.final_diagnosis:
        fd = audit.final_diagnosis
        story.append(Spacer(1, 0.1*cm))
        story.append(Paragraph("Doctor Notes (Scribe):", styles["subsection"]))
        story.append(Paragraph(fd.doctor_notes, styles["body"]))
        story.append(Spacer(1, 0.1*cm))
        story.append(Paragraph("Literature Support (Scribe):", styles["subsection"]))
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

    # Differentials from the audit trail if available
    story.append(Paragraph("Differential Diagnoses", styles["section"]))
    if audit and audit.differential_output:
        for entry in audit.differential_output.differentials:
            story.append(Paragraph(
                f"[{entry.probability.upper()}] {entry.condition} — {entry.distinguishing_test}",
                styles["body"],
            ))
    else:
        fallback_diffs = getattr(result, "differential_diagnoses", []) or getattr(result, "treatment_suggestions", [])
        if fallback_diffs:
            for dx in fallback_diffs:
                story.append(Paragraph(f"• {dx}", styles["body"]))
        else:
            story.append(Paragraph("Differential data not available for this run.", styles["body"]))

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
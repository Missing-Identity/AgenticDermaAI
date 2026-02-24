# Chapter 15 — Voice Output with ElevenLabs

**Goal:** After the doctor approves the final diagnosis, convert the patient report and doctor report to speech using ElevenLabs. Each report gets a different voice tuned to its audience. A simple prompt lets the user play either report aloud.

**Time estimate:** 25–35 minutes

---

## Design Philosophy

Two reports, two voices, two purposes:

| Report | Voice Character | Tone | Goal |
|--------|----------------|------|------|
| Patient summary | Warm, calm, reassuring | Conversational, empathetic, slightly slow | Patient feels heard and understands their condition |
| Doctor report | Professional, clear, precise | Direct, measured, emphasis on clinical detail | Doctor can listen while reviewing notes, key points are unmissable |

ElevenLabs offers voice cloning and a library of pre-built voices. We use pre-built voices for simplicity.

**This feature is optional** — the system works without it. It activates only when:
1. The doctor has approved the diagnosis
2. The user explicitly requests voice output
3. An `ELEVENLABS_API_KEY` is set in `.env`

---

## Step 1 — ElevenLabs API Key Setup

1. Go to: https://elevenlabs.io  
2. Create a free account (free tier: 10,000 characters/month — enough for testing)
3. Go to **Profile Settings → API Key**
4. Copy your API key
5. Add to **`.env`**:

```
ELEVENLABS_API_KEY=your_key_here
```

> **Character budget:** A typical patient summary is ~600–800 characters. A doctor summary narration is ~1,200–1,500 characters. At ~2,000 characters per run, the free tier covers ~5 full sessions per month. The Starter plan ($5/month) gives 30,000 characters — about 15 full sessions.

---

## Step 2 — Install the ElevenLabs SDK

```powershell
pip install elevenlabs
```

Add to `requirements.txt`:
```
elevenlabs
```

---

## Step 3 — Choose Your Voices

ElevenLabs has a voice library. To browse it programmatically, you can run this in a Python REPL:

```python
from elevenlabs.client import ElevenLabs
import os
from dotenv import load_dotenv
load_dotenv()

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
voices = client.voices.get_all()
for v in voices.voices:
    print(v.voice_id, "|", v.name, "|", v.labels)
```

Or browse at: https://elevenlabs.io/voice-library

**Recommended voice selections:**

| Report | Voice Name | Voice ID | Why |
|--------|-----------|---------|-----|
| Patient | `Rachel` | `21m00Tcm4TlvDq8ikWAM` | Warm, clear American English, widely trusted for healthcare |
| Doctor | `Adam` | `pNInz6obpgDQGcFmaJgB` | Deep, professional, authoritative — emphasises key terms naturally |

You can substitute any voices from the library. To find `voice_id` for a specific voice, look it up at: https://api.elevenlabs.io/v1/voices (returns JSON with all voice IDs).

Add the voice IDs to `.env`:
```
ELEVENLABS_PATIENT_VOICE_ID=21m00Tcm4TlvDq8ikWAM
ELEVENLABS_DOCTOR_VOICE_ID=pNInz6obpgDQGcFmaJgB
```

---

## Step 4 — Create the Voice Output Module

Create **`voice_output.py`** in the project root:

```python
# voice_output.py
# Converts text reports to speech using ElevenLabs API.
# Activated post-doctor-approval only.
# Saves MP3 files to reports/ and optionally auto-plays them.

import os
import subprocess
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

ELEVENLABS_API_KEY       = os.getenv("ELEVENLABS_API_KEY", "")
PATIENT_VOICE_ID         = os.getenv("ELEVENLABS_PATIENT_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
DOCTOR_VOICE_ID          = os.getenv("ELEVENLABS_DOCTOR_VOICE_ID", "pNInz6obpgDQGcFmaJgB")


def _is_available() -> bool:
    """Check whether ElevenLabs is configured."""
    return bool(ELEVENLABS_API_KEY)


def _get_client():
    """Return an authenticated ElevenLabs client."""
    if not _is_available():
        raise RuntimeError(
            "ELEVENLABS_API_KEY is not set in .env. "
            "Add your key to enable voice output."
        )
    from elevenlabs.client import ElevenLabs
    return ElevenLabs(api_key=ELEVENLABS_API_KEY)


def _build_patient_narration(result) -> str:
    """
    Build the patient narration script from the FinalDiagnosis.
    This is plain English — conversational, not clinical.
    Pauses are added using ElevenLabs' SSML-like break tags.
    """
    name_part = ""   # Could be personalised if patient name is passed

    return (
        f"Hello{name_part}. I want to explain what our analysis found, and what you should do next. "
        f"Please listen carefully, as this contains important health information. "
        f"\n\n"
        f"{result.patient_summary} "
        f"\n\n"
        f"Here is what we recommend you do. "
        + " ".join(
            f"Step {i+1}: {rec}"
            for i, rec in enumerate(result.patient_recommendations)
        )
        + f"\n\n"
        f"Regarding when to seek urgent medical care: {result.when_to_seek_care} "
        f"\n\n"
        f"Important: {result.disclaimer}"
    )


def _build_doctor_narration(result, audit=None) -> str:
    """
    Build the doctor narration script.
    Direct, precise, clinical. Key findings are emphasised via sentence structure.
    """
    parts = [
        f"Clinical summary for the treating physician. "
        f"Primary diagnosis: {result.primary_diagnosis}. "
        f"Confidence: {result.confidence}. Severity: {result.severity}. ",

        f"Lesion profile — "
        f"Colour: {result.lesion_profile.get('colour', 'not assessed')}. "
        f"Texture: {result.lesion_profile.get('texture', 'not assessed')}. "
        f"Elevation: {result.lesion_profile.get('levelling', 'not assessed')}. "
        f"Border: {result.lesion_profile.get('border', 'not assessed')}. ",

        f"Differential diagnoses, in order of likelihood: "
        + ". ".join(result.differential_diagnoses[:4])
        + ". ",
    ]

    if result.re_diagnosis_applied:
        parts.append(
            f"Note: A re-diagnosis was applied. {result.re_diagnosis_reason}. "
        )

    parts.append(
        f"Suggested investigations: "
        + ". ".join(result.suggested_investigations[:4])
        + ". "
    )

    parts.append(
        f"Treatment protocol: "
        + ". ".join(result.treatment_suggestions[:3])
        + ". "
    )

    parts.append(
        f"Evidence: {result.literature_support[:400]}. "
        f"Cited PMIDs: {', '.join(result.cited_pmids[:5])}. "
    )

    if audit and audit.feedback_history:
        rounds = len([h for h in audit.feedback_history if h.get("action") != "approved"])
        if rounds > 0:
            parts.append(
                f"Note: This diagnosis required {rounds} revision round{'s' if rounds > 1 else ''} "
                f"based on your feedback before approval. "
            )

    parts.append(f"End of clinical summary.")

    return " ".join(parts)


def _generate_and_save(
    text: str,
    voice_id: str,
    output_path: str,
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    style: float = 0.3,
    use_speaker_boost: bool = True,
) -> str:
    """
    Generate TTS audio and save as MP3.

    Voice settings:
        stability:        0 = expressive/variable, 1 = monotone/stable
        similarity_boost: How closely to match the original voice
        style:            0 = neutral, 1 = maximum style expression
        use_speaker_boost: Enhances voice clarity (costs slightly more characters)
    """
    client = _get_client()

    from elevenlabs import VoiceSettings

    print(f"[Voice Output] Generating audio for: {os.path.basename(output_path)}...")

    audio_generator = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id="eleven_turbo_v2_5",   # Fastest model; use "eleven_multilingual_v2" for non-English
        voice_settings=VoiceSettings(
            stability=stability,
            similarity_boost=similarity_boost,
            style=style,
            use_speaker_boost=use_speaker_boost,
        ),
        output_format="mp3_44100_128",  # 128kbps MP3 — good quality, reasonable file size
    )

    # The API returns a generator; consume it to get all bytes
    audio_bytes = b"".join(chunk for chunk in audio_generator if chunk)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(audio_bytes)

    print(f"[Voice Output] Saved: {output_path} ({len(audio_bytes) // 1024} KB)")
    return output_path


def generate_patient_audio(result, patient_name: str = "Patient") -> str:
    """
    Generate the patient summary as warm, reassuring speech.
    Returns the path to the saved MP3 file.
    """
    text = _build_patient_narration(result)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = patient_name.replace(" ", "_")
    output_path = f"reports/patient_audio_{safe_name}_{timestamp}.mp3"

    return _generate_and_save(
        text=text,
        voice_id=PATIENT_VOICE_ID,
        output_path=output_path,
        stability=0.45,          # Slightly expressive — warm, natural
        similarity_boost=0.80,
        style=0.25,              # Mild style — reassuring not dramatic
    )


def generate_doctor_audio(result, audit=None, patient_name: str = "Patient") -> str:
    """
    Generate the doctor report as clear, professional speech.
    Returns the path to the saved MP3 file.
    """
    text = _build_doctor_narration(result, audit)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = patient_name.replace(" ", "_")
    output_path = f"reports/doctor_audio_{safe_name}_{timestamp}.mp3"

    return _generate_and_save(
        text=text,
        voice_id=DOCTOR_VOICE_ID,
        output_path=output_path,
        stability=0.65,          # More stable — professional, not overly expressive
        similarity_boost=0.75,
        style=0.15,              # Minimal style — focus is on clarity
    )


def play_audio(file_path: str) -> None:
    """
    Play an MP3 file using the system default player.
    Cross-platform: Windows uses 'start', macOS uses 'open', Linux uses 'xdg-open'.
    """
    if not os.path.exists(file_path):
        print(f"  ⚠️  Audio file not found: {file_path}")
        return

    print(f"[Voice Output] Playing: {os.path.basename(file_path)}")

    if sys.platform == "win32":
        os.startfile(file_path)
    elif sys.platform == "darwin":
        subprocess.run(["open", file_path])
    else:
        subprocess.run(["xdg-open", file_path])


def offer_voice_output(result, audit=None, patient_name: str = "Patient") -> None:
    """
    After doctor approval, offer to generate and play voice output.
    Shows a menu — does nothing if the user declines or ElevenLabs is not configured.
    """
    if not _is_available():
        print("\n[Voice Output] ElevenLabs not configured (ELEVENLABS_API_KEY not set). Skipping.")
        return

    print("\n" + "─"*60)
    print("VOICE OUTPUT (ElevenLabs)")
    print("─"*60)
    print("Generate spoken versions of the reports?\n")
    print("  [1] Generate patient summary audio (warm, reassuring voice)")
    print("  [2] Generate doctor report audio (professional, precise voice)")
    print("  [3] Generate both")
    print("  [4] Skip voice output")

    choice = input("\nChoice (1/2/3/4): ").strip()

    if choice in ("1", "3"):
        try:
            patient_mp3 = generate_patient_audio(result, patient_name)
            print(f"  ✅ Patient audio: {patient_mp3}")
            play_now = input("  Play patient audio now? (y/n): ").strip().lower()
            if play_now == "y":
                play_audio(patient_mp3)
        except Exception as e:
            print(f"  ⚠️  Patient audio failed: {e}")

    if choice in ("2", "3"):
        try:
            doctor_mp3 = generate_doctor_audio(result, audit, patient_name)
            print(f"  ✅ Doctor audio: {doctor_mp3}")
            play_now = input("  Play doctor audio now? (y/n): ").strip().lower()
            if play_now == "y":
                play_audio(doctor_mp3)
        except Exception as e:
            print(f"  ⚠️  Doctor audio failed: {e}")
```

---

## Step 5 — Test Voice Output

Create **`test_voice_output.py`** using the mock FinalDiagnosis from Chapter 11:

```python
# test_voice_output.py
from agents.orchestrator_agent import FinalDiagnosis
from voice_output import generate_patient_audio, generate_doctor_audio, play_audio

mock_result = FinalDiagnosis(
    primary_diagnosis="Allergic Contact Dermatitis",
    confidence="high",
    differential_diagnoses=["Irritant Contact Dermatitis", "Atopic Dermatitis", "Nummular Eczema"],
    severity="Moderate",
    lesion_profile={
        "colour": "Erythematous with violaceous tinge on medium skin tone",
        "texture": "Papular with fine scaling",
        "levelling": "Raised",
        "border": "Irregular, not well-defined",
    },
    clinical_reasoning=(
        "The presentation is consistent with occupational allergic contact dermatitis "
        "secondary to solvent exposure. "
        "Erythematous papular rash with irregular borders on forearm of a painter "
        "following solvent brand change is a classic presentation."
    ),
    re_diagnosis_applied=False,
    re_diagnosis_reason="",
    patient_summary=(
        "You appear to have an allergic skin reaction caused by contact with a chemical "
        "in your workplace. Your skin is red and itchy because your immune system "
        "is reacting to the new solvent. Once you avoid the trigger, "
        "your skin should start healing within one to two weeks."
    ),
    patient_recommendations=[
        "Avoid the new solvent brand immediately. Wear protective gloves at work.",
        "Apply hydrocortisone 1% cream twice daily for 7 days.",
        "Take cetirizine 10mg once daily for itch relief.",
        "See a dermatologist within one week if no improvement.",
    ],
    doctor_notes=(
        "Clinically consistent with occupational ACD secondary to solvent exposure. "
        "Recommend patch testing. Prescribe mid-potency topical corticosteroid if OTC "
        "hydrocortisone is insufficient. Refer to occupational health if exposure continues."
    ),
    suggested_investigations=["Patch testing", "Skin swab if infection suspected"],
    treatment_suggestions=[
        "Topical hydrocortisone 1% (OTC) or betamethasone 0.1% (prescription)",
        "Oral antihistamine for itch",
        "Emollient for barrier repair",
    ],
    literature_support=(
        "Strong literature support for ACD in occupational painter settings. "
        "Multiple systematic reviews cite solvent exposure as leading cause."
    ),
    cited_pmids=["38421234", "37892341"],
    when_to_seek_care=(
        "Go to A&E immediately if rash spreads to face or you develop breathing difficulty. "
        "See a GP within 2 days if rash spreads significantly or you develop fever."
    ),
)

print("Generating patient audio...")
patient_mp3 = generate_patient_audio(mock_result, patient_name="Test Patient")
print(f"✅ Patient audio: {patient_mp3}")

print("\nGenerating doctor audio...")
doctor_mp3 = generate_doctor_audio(mock_result, patient_name="Test Patient")
print(f"✅ Doctor audio: {doctor_mp3}")

# Play both
play_audio(patient_mp3)
input("Press Enter to play doctor audio...")
play_audio(doctor_mp3)
```

Run it:
```powershell
python test_voice_output.py
```

**What to verify:**

**Patient audio:**
- Voice is warm and unhurried — sounds like a caring nurse
- The summary is easy to understand when heard (not read)
- Recommendations are spoken clearly and distinctly
- The disclaimer doesn't sound alarming

**Doctor audio:**
- Voice is clear and professional — sounds like a clinical summary
- Diagnosis, confidence, and severity are stated early and clearly
- PMIDs are read out (good for dictation scenarios)
- The tone is direct without being cold

**Common issues:**

`401 Unauthorized` — Your API key is wrong or not loaded. Check `.env` and run `load_dotenv()` before calling.

`400 Bad Request / text too long` — ElevenLabs has a per-request character limit (~5,000 chars for most plans). If your narration is long, split it into chunks:
```python
# In _generate_and_save, add before the API call:
MAX_CHARS = 4500
if len(text) > MAX_CHARS:
    text = text[:MAX_CHARS] + "... Report truncated. Please see the full PDF."
```

`Voice ID not found` — The voice IDs above are correct as of early 2026 but ElevenLabs occasionally changes their library. Use the voice browser script in Step 3 to get current IDs.

**Delete `test_voice_output.py` when it passes.**

---

## Step 6 — Wire into main.py

Open **`main.py`**. After the patient PDF is generated (post-approval), add:

```python
# ── Voice output (optional, post-approval) ────────────────────────────────────
from voice_output import offer_voice_output
offer_voice_output(result, audit=audit, patient_name=pdf_patient_name)
```

`offer_voice_output()` shows the menu and handles everything. If `ELEVENLABS_API_KEY` is not set, it silently skips.

---

## Checkpoint ✅

- [ ] `elevenlabs` installed and in `requirements.txt`
- [ ] `ELEVENLABS_API_KEY`, `ELEVENLABS_PATIENT_VOICE_ID`, `ELEVENLABS_DOCTOR_VOICE_ID` in `.env`
- [ ] `voice_output.py` exists with both narration builders and `offer_voice_output()`
- [ ] Patient audio sounds warm and reassuring — appropriate for a worried patient
- [ ] Doctor audio sounds professional and clinical — key findings are clearly audible
- [ ] Both MP3 files saved to `reports/` with timestamped filenames
- [ ] `play_audio()` works on Windows (uses `os.startfile`)
- [ ] `main.py` calls `offer_voice_output()` only after doctor approval
- [ ] Feature degrades gracefully when `ELEVENLABS_API_KEY` is missing

---

## Final Project Structure (Chapters 12–15 additions)

```
dermaai-v2/
├── agents/
│   ├── clinical_agents.py          ← CH 12 (Differential + Treatment)
│   └── ... (existing agents)
│
├── audit_trail.py                  ← CH 13
├── doctor_approval.py              ← CH 13
├── voice_input.py                  ← CH 14 (faster-whisper recording + transcription)
├── voice_output.py                 ← CH 15 (ElevenLabs TTS)
│
└── reports/
    ├── doctor_audit_TIMESTAMP.pdf  ← Full agent workflow (generated pre-approval)
    ├── doctor_report_TIMESTAMP.pdf ← Doctor clinical report (generated post-approval)
    ├── patient_summary_TIMESTAMP.pdf
    ├── patient_audio_TIMESTAMP.mp3 ← CH 15
    └── doctor_audio_TIMESTAMP.mp3  ← CH 15
```

---

*Next → `11_PDF_REPORTS.md` (updated — now generates the full audit trail PDF)*

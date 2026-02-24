# Chapter 14 — Voice Input with faster-whisper

**Goal:** Let the patient describe their symptoms by speaking instead of typing. Audio is recorded locally, transcribed locally using `faster-whisper`, and the resulting text feeds into the pipeline exactly as typed input would. No audio ever leaves the machine.

**Time estimate:** 30–40 minutes

---

## Why Local Transcription?

Patient symptom descriptions contain sensitive health information. Using a cloud transcription service (OpenAI Whisper API, Google Speech-to-Text, etc.) means sending that audio to a third-party server. For a clinical tool, this is a privacy violation.

**`faster-whisper`** is a highly optimised re-implementation of OpenAI's Whisper model that runs entirely on your local machine (CPU or GPU). It is:
- 2–4x faster than the original `openai-whisper` package on the same hardware
- More memory efficient (uses CTranslate2 under the hood)
- No internet required after the initial model download
- Accurate on accented speech and medical terminology

---

## Step 1 — Install Dependencies

Add to your virtual environment:

```powershell
pip install faster-whisper sounddevice scipy
```

- **faster-whisper** — the transcription model
- **sounddevice** — captures audio from the microphone (cross-platform, no C++ build required)
- **scipy** — saves audio as a `.wav` file that Whisper can read

Also add to **`requirements.txt`**:
```
faster-whisper
sounddevice
scipy
```

---

## Step 2 — Choose Your Whisper Model Size

faster-whisper downloads models from Hugging Face on first use and caches them locally. The trade-off between speed and accuracy:

| Model  | VRAM / RAM | Speed (RTX 4050) | Accuracy | Recommended for |
|--------|-----------|-----------------|----------|-----------------|
| `tiny` | ~75MB | < 1s | Basic | Testing only |
| `base` | ~150MB | ~2s | Good | Fast machines |
| `small` | ~250MB | ~4s | Very good | **Recommended default** |
| `medium` | ~800MB | ~8s | Excellent | If accuracy is critical |
| `large-v3` | ~3GB | ~20s | Best | High-end machines |

For a clinical tool where medical terminology must be transcribed accurately, **`small`** is the recommended starting point. If patients frequently use complex medical history terms, try `medium`.

> **GPU acceleration:** If you have CUDA set up (you do — you have an RTX 4050), faster-whisper will automatically use it. Set `device="cuda"` and `compute_type="float16"` for maximum speed.

---

## Step 3 — Create the Voice Input Module

Create **`voice_input.py`** in the project root:

```python
# voice_input.py
# Records audio from the microphone and transcribes it locally
# using faster-whisper. No audio leaves the machine.

import os
import tempfile
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel


# ── Model configuration ────────────────────────────────────────────────────────
# Change model_size to "medium" if you need higher accuracy.
# Change device to "cpu" if you're not using CUDA.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = "cuda"        # "cuda" for GPU, "cpu" for CPU-only
WHISPER_COMPUTE = "float16"    # "float16" for GPU, "int8" for CPU


def _load_model() -> WhisperModel:
    """
    Load the faster-whisper model.
    First call downloads the model to ~/.cache/huggingface/hub/ (~250MB for small).
    Subsequent calls load from cache (fast).
    """
    print(f"[Voice] Loading Whisper model '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE}...")
    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
    )
    print("[Voice] Model ready.")
    return model


# Load model once at module level (cached across calls within a session)
_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def record_audio(duration_seconds: int = 60, sample_rate: int = 16000) -> np.ndarray:
    """
    Record audio from the default microphone.

    Args:
        duration_seconds: Maximum recording time. Recording stops early if
                          the user presses Enter.
        sample_rate: 16000 Hz is what Whisper expects. Do not change.

    Returns:
        NumPy array of audio samples (mono, float32).
    """
    print(f"\n[Voice] Recording for up to {duration_seconds} seconds.")
    print("        Speak clearly. Press ENTER at any time to stop early.\n")

    # Start recording in a thread so we can listen for Enter simultaneously
    import threading

    audio_chunks = []
    stop_event = threading.Event()

    def _record():
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
            while not stop_event.is_set():
                chunk, _ = stream.read(sample_rate // 10)  # 100ms chunks
                audio_chunks.append(chunk)

    record_thread = threading.Thread(target=_record, daemon=True)
    record_thread.start()

    # Wait for Enter or duration timeout
    import time
    start = time.time()
    try:
        while time.time() - start < duration_seconds:
            if input() == "" or stop_event.is_set():
                break
    except EOFError:
        pass  # non-interactive mode

    stop_event.set()
    record_thread.join(timeout=1.0)

    if not audio_chunks:
        raise RuntimeError("No audio recorded. Check microphone connection.")

    audio = np.concatenate(audio_chunks, axis=0).flatten()
    print(f"[Voice] Recorded {len(audio) / sample_rate:.1f} seconds of audio.")
    return audio


def transcribe_audio(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """
    Transcribe a NumPy audio array using faster-whisper.

    Returns the transcription as a single string.
    """
    # Save to a temporary WAV file (faster-whisper reads from file, not memory)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        wav.write(tmp_path, sample_rate, (audio * 32767).astype(np.int16))

    try:
        print("[Voice] Transcribing...")
        model = get_model()

        segments, info = model.transcribe(
            tmp_path,
            language="en",               # Force English; remove for auto-detection
            beam_size=5,                 # Higher beam_size = more accurate, slower
            vad_filter=True,             # Remove silence at start/end
            vad_parameters={
                "min_silence_duration_ms": 500,   # Silences under 0.5s are kept
            },
        )

        # Collect all segments into a single string
        full_text = " ".join(segment.text.strip() for segment in segments)
        full_text = full_text.strip()

        print(f"[Voice] Transcription complete ({info.language}, "
              f"confidence: {info.language_probability:.0%})")
        return full_text

    finally:
        os.unlink(tmp_path)   # Always clean up the temp file


def record_and_transcribe(duration_seconds: int = 60) -> str:
    """
    Convenience function: record then immediately transcribe.
    Returns the transcription string.
    """
    audio = record_audio(duration_seconds=duration_seconds)
    return transcribe_audio(audio)
```

---

## Step 4 — Test Voice Input in Isolation

Create **`test_voice_input.py`**:

```python
# test_voice_input.py
from voice_input import record_and_transcribe

print("=" * 50)
print("VOICE INPUT TEST")
print("=" * 50)
print("This will record your voice and transcribe it locally.")
print("Describe a fake skin condition for testing purposes.\n")

transcription = record_and_transcribe(duration_seconds=30)

print("\n" + "=" * 50)
print("TRANSCRIPTION RESULT:")
print("=" * 50)
print(transcription)
print("\nIf this looks correct, voice input is working.")
```

Run it:
```powershell
python test_voice_input.py
```

Speak something like: *"I have a red, itchy rash on my left arm that started three days ago. It's spreading and gets worse in the evening."*

**What to verify:**
- The transcription captures the key words accurately (rash, itchy, left arm, three days)
- Medical terms like "erythematous" or "pruritic" are transcribed correctly if you use them
- The recording stops when you press Enter
- No audio files remain after the test (`tempfile` is cleaned up)

**Common issues:**

`No audio recorded` or `PortAudio not found`:
```powershell
# On Windows, sounddevice usually works out of the box.
# If not, install PortAudio:
pip install sounddevice --upgrade
```

`CUDA out of memory` during model load:
- Change `WHISPER_COMPUTE = "int8"` and `WHISPER_DEVICE = "cuda"` — int8 uses much less VRAM
- Or change `WHISPER_DEVICE = "cpu"` — slower but always works

Model downloads to the wrong location:
- Set `HF_HOME` in your `.env` to control the Hugging Face cache directory

**Delete `test_voice_input.py` when it passes.**

---

## Step 5 — Integrate Voice Input into main.py

Open **`main.py`** and update `get_patient_text()` to offer voice as an option:

```python
def get_patient_text() -> str:
    """Prompt patient for symptom description — text or voice."""
    print("\nHow would you like to describe your symptoms?")
    print("  [1] Type your description")
    print("  [2] Speak your description (recorded locally, no cloud)")

    while True:
        choice = input("\nChoice (1/2): ").strip()
        if choice == "1":
            return _get_text_input()
        elif choice == "2":
            return _get_voice_input()
        print("  Please enter 1 or 2.")


def _get_text_input() -> str:
    """Original text input flow."""
    print("\nDescribe your symptoms in detail:")
    print("  - What does the skin look like?")
    print("  - Where is it? How long have you had it?")
    print("  - What makes it better or worse?\n")
    text = input("Your description: ").strip()
    if not text:
        print("  Please describe your symptoms.")
        return _get_text_input()
    return text


def _get_voice_input() -> str:
    """Voice recording + transcription flow."""
    from voice_input import record_and_transcribe

    print("\nVoice recording mode.")
    print("You have up to 2 minutes to describe your symptoms.")
    print("Speak clearly. Press ENTER when you are finished.\n")

    try:
        transcription = record_and_transcribe(duration_seconds=120)

        if not transcription:
            print("  ⚠️  Nothing was transcribed. Please try again.")
            return _get_voice_input()

        print(f"\nTranscribed: \"{transcription}\"")
        confirm = input("\nIs this correct? (y/n): ").strip().lower()

        if confirm == "y":
            return transcription
        else:
            print("  Let's try again.")
            return _get_voice_input()

    except Exception as e:
        print(f"  ⚠️  Voice input failed: {e}")
        print("  Falling back to text input.")
        return _get_text_input()
```

---

## Step 6 — Optional: Voice Input for Doctor Feedback

You can also allow the doctor to give rejection feedback by voice. In `doctor_approval.py`, update the rejection feedback prompt:

```python
# In get_doctor_decision(), replace the feedback input with:
print("Provide your feedback:")
print("  [1] Type feedback")
print("  [2] Speak feedback (recorded locally)")
feedback_mode = input("Choice (1/2): ").strip()

if feedback_mode == "2":
    from voice_input import record_and_transcribe
    print("\nSpeak your feedback. Press ENTER when done.\n")
    feedback = record_and_transcribe(duration_seconds=60)
    print(f"\nTranscribed: \"{feedback}\"")
    confirm = input("Correct? (y/n): ").strip().lower()
    if confirm != "y":
        feedback = input("Type your feedback instead: ").strip()
else:
    feedback = input("Your feedback: ").strip()
```

---

## Checkpoint ✅

- [ ] `faster-whisper`, `sounddevice`, `scipy` installed and in `requirements.txt`
- [ ] `voice_input.py` exists with `record_audio()`, `transcribe_audio()`, `record_and_transcribe()`
- [ ] Model loads on first call and is cached for subsequent calls within a session
- [ ] Voice recording stops when Enter is pressed (does not wait for full `duration_seconds`)
- [ ] Temp WAV file is deleted after transcription
- [ ] Transcription test passes with reasonable accuracy on natural speech
- [ ] `main.py` offers text/voice choice and falls back to text if voice fails
- [ ] No audio data is sent to any external service

---

*Next → `15_VOICE_OUTPUT.md`*

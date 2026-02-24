from agents.biodata_agent import PatientProfile, save_profile


def prompt(label: str, hint: str = "") -> str:
    """Helper to ask a question and return stripped input."""
    if hint:
        label = f"{label} ({hint})"
    return input(f"{label}: ").strip()


def prompt_list(label: str) -> list[str]:
    """Ask for comma-separated values, return as a list."""
    raw = input(f"{label} (comma-separated, or press Enter to skip): ").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def collect_profile() -> PatientProfile:
    print("\n" + "="*55)
    print("  DermaAI v2 — Patient Profile Setup")
    print("  All fields are optional. Press Enter to skip any.")
    print("="*55 + "\n")

    profile = PatientProfile(
        name=prompt("Patient Name", "or press Enter for Anonymous") or "Anonymous",
        age=int(a) if (a := prompt("Age", "years")) else None,
        sex=prompt("Biological Sex", "Male / Female / Other") or None,
        gender=prompt("Gender Identity", "if different from sex, else Enter") or None,
        skin_tone=prompt(
            "Skin Tone",
            "very light / light / medium / medium-dark / dark / very dark"
        ) or None,
        occupation=prompt("Occupation") or None,
        caste=prompt("Ethnicity / Caste") or None,
        pincode=prompt("Area Pincode") or None,
        known_allergies=prompt_list("Known Allergies"),
        current_medications=prompt_list("Current Medications"),
        past_skin_conditions=prompt_list("Past Skin Conditions"),
        family_skin_history=prompt("Family History of Skin Conditions") or None,
        notes=prompt("Any other relevant notes") or None,
    )

    return profile


def main():
    profile = collect_profile()

    print("\n--- Profile collected ---")
    print(profile.model_dump_json(indent=2))

    confirm = input("\nSave this profile? (y/n): ").strip().lower()
    if confirm == "y":
        save_profile(profile)
        print("Ready. Run your diagnosis session now.")
    else:
        print("Profile discarded.")


if __name__ == "__main__":
    main()
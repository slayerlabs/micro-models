"""Upload paczki modelu na Hugging Face Hub.
Wymaga wcześniejszego `huggingface-cli login`. Uruchom z katalogu repo:
    python src/upload_hf.py Maggio33/slay-piano-gpt
"""
import sys
from huggingface_hub import HfApi, create_repo

IGNORE = [
    "data/tunes.csv", "data/jigs.abc", "data/gpt_ckpt_v1_dirty.pt",
    "data/_test_chord.mid", "data/generated*.abc",
    "data/demo_*.mid", "data/clean_*.mid", "data/gpt_jig_*.mid",
    "data/jig_*.mid", "data/out_*.*",
    "**/__pycache__/**", "*.pyc",
]

def main():
    repo_id = sys.argv[1] if len(sys.argv) > 1 else "Maggio33/slay-piano-gpt"
    api = HfApi()
    who = api.whoami()
    print("zalogowany jako:", who.get("name"))
    create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    print(f"repo gotowe: {repo_id} — wysyłam pliki…")
    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=IGNORE,
        commit_message="Initial: slay-piano-gpt (0.82M char-level GPT, jigs ABC)",
    )
    print(f"\nGOTOWE -> https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()

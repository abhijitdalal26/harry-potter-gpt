"""
Orchestrates the full Harry Potter GPT pipeline: pretrain -> SFT -> HF convert -> DPO -> showcase.

Meant to be run from inside the nanoGPT/ directory (relative paths assume that cwd),
typically on a Colab GPU runtime with Google Drive mounted for durable backups.

Each stage:
  1. Checks Drive for an existing backup zip of that stage's output.
     - If present and large enough to plausibly be complete, restores it and skips training.
     - If present but suspiciously small, treats it as bad and retrains.
     - If the zip fails to extract (corrupted), treats it as bad and retrains.
  2. Otherwise runs the stage fresh, entirely on local disk (never trains directly onto
     a Drive-mounted path -- Drive's FUSE mount chokes on frequent large writes).
  3. Zips the result and copies the single zip to Drive as a backup.

This means the pipeline is safe to interrupt and re-run from scratch at any point --
completed stages are auto-skipped, and only a genuinely half-finished stage gets redone.

Usage:
    python run_pipeline.py --drive-backup /content/drive/MyDrive/harry-potter-gpt --stage all
    python run_pipeline.py --drive-backup /content/drive/MyDrive/harry-potter-gpt --stage dpo
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# rough minimum zip sizes for a genuinely complete checkpoint. nanoGPT checkpoints include
# optimizer state (~1.3-1.5GB zipped); HF exports are weights-only (~450-500MB zipped).
# anything well under this almost certainly means an interrupted/wrong run got zipped by mistake.
MIN_ZIP_SIZE = {
    "out-harry-potter": 800_000_000,
    "out-harry-potter-sft": 800_000_000,
    "harry-potter-hf": 300_000_000,
    "harry-potter-hf-dpo": 300_000_000,
}


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def restore_from_drive(name: str, drive_backup: Path) -> bool:
    """Try to restore `name/` from `drive_backup/{name}.zip`. Returns True if restored."""
    local_dir = Path(name)
    if local_dir.is_dir():
        print(f"[{name}] already present locally, skipping restore check")
        return True

    zip_path = drive_backup / f"{name}.zip"
    if not zip_path.exists():
        print(f"[{name}] no backup in Drive yet")
        return False

    zip_size = zip_path.stat().st_size
    min_size = MIN_ZIP_SIZE[name]
    if zip_size < min_size:
        print(
            f"[{name}] WARNING: backup zip is only {zip_size / 1e6:.0f}MB "
            f"(expected {min_size / 1e6:.0f}MB+) -- looks incomplete, not restoring"
        )
        return False

    print(f"[{name}] found valid-sized backup ({zip_size / 1e6:.0f}MB), restoring...")
    result = subprocess.run(["unzip", "-oq", str(zip_path), "-d", "."])
    if result.returncode != 0:
        print(f"[{name}] WARNING: unzip failed (exit {result.returncode}) -- backup is corrupted, not restoring")
        shutil.rmtree(local_dir, ignore_errors=True)
        return False

    print(f"[{name}] restored from Drive")
    return True


def backup_to_drive(name: str, drive_backup: Path) -> None:
    zip_path = Path(f"/tmp/{name}.zip") if sys.platform != "win32" else Path(f"{name}.zip")
    run(["zip", "-rq", str(zip_path), name])
    drive_backup.mkdir(parents=True, exist_ok=True)
    dest = drive_backup / f"{name}.zip"
    shutil.copy(zip_path, dest)
    size_mb = dest.stat().st_size / 1e6
    print(f"[{name}] backed up to {dest} ({size_mb:.0f}MB)")


def stage_pretrain(drive_backup: Path) -> None:
    print("\n=== Stage 1: continued pretraining ===")
    if restore_from_drive("out-harry-potter", drive_backup):
        return
    if not Path("data/harry_potter/train.bin").exists():
        run(["python", "data/harry_potter/prepare.py"])
    run(["python", "train.py", "config/finetune_harry_potter.py", "--compile=False"])
    run(["python", "sample.py", "--out_dir=out-harry-potter", "--start=Harry Potter",
         "--num_samples=3", "--max_new_tokens=200"])
    backup_to_drive("out-harry-potter", drive_backup)


def stage_sft(drive_backup: Path) -> None:
    print("\n=== Stage 2: supervised fine-tuning ===")
    if restore_from_drive("out-harry-potter-sft", drive_backup):
        return
    if not Path("out-harry-potter/ckpt.pt").exists():
        stage_pretrain(drive_backup)
    if not Path("out-harry-potter-sft").exists():
        shutil.copytree("out-harry-potter", "out-harry-potter-sft")
    if not Path("data/harry_potter_sft/train.bin").exists():
        run(["python", "data/harry_potter_sft/prepare_sft.py"])
    run(["python", "train.py", "config/harry_potter_sft.py",
         "--out_dir=out-harry-potter-sft", "--compile=False"])
    sft_sample_prompt = "<|user|> Who is Harry Potter?\n<|assistant|>"
    run(["python", "sample.py", "--out_dir=out-harry-potter-sft",
         f"--start={sft_sample_prompt}", "--num_samples=3", "--max_new_tokens=200"])
    backup_to_drive("out-harry-potter-sft", drive_backup)


def stage_convert(drive_backup: Path) -> None:
    print("\n=== Stage 3: convert nanoGPT checkpoint to HuggingFace format ===")
    if restore_from_drive("harry-potter-hf", drive_backup):
        return
    if not Path("out-harry-potter-sft/ckpt.pt").exists():
        stage_sft(drive_backup)
    run(["python", "convert_to_hf.py",
         "--ckpt_path=out-harry-potter-sft/ckpt.pt", "--output_dir=harry-potter-hf"])
    backup_to_drive("harry-potter-hf", drive_backup)


def stage_dpo(drive_backup: Path) -> None:
    print("\n=== Stage 4: DPO preference alignment ===")
    if restore_from_drive("harry-potter-hf-dpo", drive_backup):
        return
    if not Path("harry-potter-hf").exists():
        stage_convert(drive_backup)
    if not Path("data/harry_potter_dpo/train").exists():
        run(["python", "data/harry_potter_dpo/prepare_dpo.py"])
    run(["python", "train_dpo.py"])
    backup_to_drive("harry-potter-hf-dpo", drive_backup)


def stage_showcase(drive_backup: Path) -> None:
    print("\n=== Final: stage comparison showcase ===")
    for name in ["out-harry-potter", "harry-potter-hf", "harry-potter-hf-dpo"]:
        if not Path(name).exists():
            restore_from_drive(name, drive_backup)
    if not Path("out-harry-potter/ckpt.pt").exists():
        stage_pretrain(drive_backup)
    if not Path("harry-potter-hf").exists():
        stage_convert(drive_backup)
    if not Path("harry-potter-hf-dpo").exists():
        stage_dpo(drive_backup)

    run(["python", "showcase.py",
         "--base_dir=out-harry-potter", "--sft_dir=harry-potter-hf",
         "--dpo_dir=harry-potter-hf-dpo", "--output=results/stage_comparison.md"])

    drive_backup.mkdir(parents=True, exist_ok=True)
    shutil.copy("results/stage_comparison.md", drive_backup / "stage_comparison.md")
    print(f"Copied transcript to {drive_backup / 'stage_comparison.md'}")

    full_zip = Path("/tmp/harry-potter-gpt-full.zip") if sys.platform != "win32" else Path("harry-potter-gpt-full.zip")
    run(["zip", "-rq", str(full_zip), "out-harry-potter", "out-harry-potter-sft",
         "harry-potter-hf", "harry-potter-hf-dpo", "results"])
    shutil.copy(full_zip, drive_backup / "harry-potter-gpt-full.zip")
    print(f"Copied full bundle to {drive_backup / 'harry-potter-gpt-full.zip'}")


STAGES = {
    "pretrain": stage_pretrain,
    "sft": stage_sft,
    "convert": stage_convert,
    "dpo": stage_dpo,
    "showcase": stage_showcase,
}
STAGE_ORDER = ["pretrain", "sft", "convert", "dpo", "showcase"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-backup", required=True, help="Directory to read/write stage backups")
    parser.add_argument("--stage", choices=["all", *STAGE_ORDER], default="all")
    args = parser.parse_args()

    drive_backup = Path(args.drive_backup)
    to_run = STAGE_ORDER if args.stage == "all" else [args.stage]

    for stage_name in to_run:
        STAGES[stage_name](drive_backup)

    print("\nDone.")

"""
Orchestrates the full Harry Potter GPT pipeline: pretrain -> SFT -> HF convert -> DPO -> showcase.

Meant to be run from inside the nanoGPT/ directory (relative paths assume that cwd),
typically on a Colab GPU runtime with Google Drive mounted for durable backups.

Each stage:
  1. Checks Drive for an existing backup of that stage's output, verified two ways:
     - a sha256 checksum (written into a manifest alongside the zip at backup time)
       catches corruption/truncation -- a zip that changed since it was written.
     - for nanoGPT checkpoints, the `iter_num` stored inside ckpt.pt itself is checked
       against the config's expected iteration count -- a checksum-valid checkpoint
       can still be a valid file that just wasn't trained far enough (this is what
       actually happened: Stage 1 was silently retrained from scratch and produced a
       structurally fine but undertrained checkpoint). File size never told us that;
       reading the checkpoint's own training metadata does.
     If both checks pass, the backup is restored and that stage is skipped.
  2. Otherwise runs the stage fresh, entirely on local disk (never trains directly onto
     a Drive-mounted path -- Drive's FUSE mount chokes on frequent large writes).
  3. Zips the result, writes a manifest (checksum + training metadata), and copies both
     to Drive as the backup.

This means the pipeline is safe to interrupt and re-run from scratch at any point --
completed stages are auto-skipped, and only a genuinely half-finished or undertrained
stage gets redone.

Usage:
    python run_pipeline.py --drive-backup /content/drive/MyDrive/harry-potter-gpt --stage all
    python run_pipeline.py --drive-backup /content/drive/MyDrive/harry-potter-gpt --stage dpo
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

# nanoGPT checkpoints store their own iter_num. Compare against each config's max_iters
# (with slack for early-stopping via always_save_checkpoint) to catch a checkpoint that's
# structurally valid but wasn't actually trained to completion.
EXPECTED_MIN_ITERS = {
    "out-harry-potter": 200,      # config/finetune_harry_potter.py: max_iters = 250
    "out-harry-potter-sft": 500,  # config/harry_potter_sft.py: max_iters = 600
}


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_nanogpt_iter_num(local_dir: Path) -> int | None:
    ckpt_file = local_dir / "ckpt.pt"
    if not ckpt_file.exists():
        return None
    ckpt = torch.load(ckpt_file, map_location="cpu")
    return ckpt.get("iter_num")


def local_checkpoint_is_valid(name: str) -> bool:
    """A local dir existing isn't proof it's complete -- e.g. an interrupted training run
    leaves a partial (or missing) checkpoint behind. For nanoGPT stages, check iter_num same
    as we would for a Drive restore; for everything else, just check it looks non-empty."""
    local_dir = Path(name)
    if name in EXPECTED_MIN_ITERS:
        iter_num = read_nanogpt_iter_num(local_dir)
        if iter_num is None or iter_num < EXPECTED_MIN_ITERS[name]:
            print(f"[{name}] local copy is incomplete (iter_num={iter_num}, expected >= {EXPECTED_MIN_ITERS[name]}) "
                  f"-- likely from an interrupted run, discarding it")
            return False
        return True
    return any(local_dir.iterdir())


def restore_from_drive(name: str, drive_backup: Path) -> bool:
    """Try to restore `name/` from `drive_backup/{name}.zip`, verified against its manifest."""
    local_dir = Path(name)
    if local_dir.is_dir():
        if local_checkpoint_is_valid(name):
            print(f"[{name}] already present locally and verified, skipping restore check")
            return True
        shutil.rmtree(local_dir, ignore_errors=True)
        # fall through to try restoring from Drive instead

    zip_path = drive_backup / f"{name}.zip"
    manifest_path = drive_backup / f"{name}.manifest.json"
    if not zip_path.exists():
        print(f"[{name}] no backup in Drive yet")
        return False
    if not manifest_path.exists():
        print(f"[{name}] WARNING: backup zip exists but has no manifest (older run, or written by hand) "
              f"-- can't verify it, not restoring")
        return False

    manifest = json.loads(manifest_path.read_text())
    actual_sha256 = sha256_file(zip_path)
    if actual_sha256 != manifest.get("sha256"):
        print(f"[{name}] WARNING: checksum mismatch against manifest (backup is corrupted or was overwritten "
              f"since being verified) -- not restoring")
        return False

    result = subprocess.run(["unzip", "-oq", str(zip_path), "-d", "."])
    if result.returncode != 0:
        print(f"[{name}] WARNING: unzip failed (exit {result.returncode}) despite matching checksum -- not restoring")
        shutil.rmtree(local_dir, ignore_errors=True)
        return False

    if name in EXPECTED_MIN_ITERS:
        min_iters = EXPECTED_MIN_ITERS[name]
        iter_num = manifest.get("iter_num")
        if iter_num is None or iter_num < min_iters:
            print(f"[{name}] WARNING: checkpoint only reached iter_num={iter_num} "
                  f"(expected >= {min_iters}) -- undertrained, not using it")
            shutil.rmtree(local_dir, ignore_errors=True)
            return False
        print(f"[{name}] restored and verified (checksum OK, iter_num={iter_num})")
    else:
        print(f"[{name}] restored and verified (checksum OK)")

    return True


def backup_to_drive(name: str, drive_backup: Path) -> None:
    zip_path = Path(f"/tmp/{name}.zip") if sys.platform != "win32" else Path(f"{name}.zip")
    run(["zip", "-rq", str(zip_path), name])

    manifest = {"name": name, "sha256": sha256_file(zip_path), "size": zip_path.stat().st_size}
    if name in EXPECTED_MIN_ITERS:
        manifest["iter_num"] = read_nanogpt_iter_num(Path(name))

    drive_backup.mkdir(parents=True, exist_ok=True)
    dest_zip = drive_backup / f"{name}.zip"
    dest_manifest = drive_backup / f"{name}.manifest.json"
    shutil.copy(zip_path, dest_zip)
    dest_manifest.write_text(json.dumps(manifest, indent=2))

    detail = f", iter_num={manifest['iter_num']}" if "iter_num" in manifest else ""
    print(f"[{name}] backed up to {dest_zip} ({manifest['size'] / 1e6:.0f}MB{detail})")


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

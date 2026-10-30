import hashlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


TARGETS_README = ("README.md",)


def checksum_targets():
    """Everything whose hash is recorded: scripts, docs and result tables."""
    return sorted(
        list(ROOT.glob("*.py")) +
        [ROOT / name for name in TARGETS_README if (ROOT / name).exists()] +
        list(RESULTS.glob("*.csv")) +
        list(RESULTS.glob("summary_exp*.json")) +
        [ROOT / "requirements.txt"]
    )


def write_checksums():
    lines = [f"{sha256(p)}  {p.relative_to(ROOT)}" for p in checksum_targets()]
    (RESULTS / "checksums.sha256").write_text("\n".join(lines) + "\n")
    return len(lines)


def verify_checksums():
    """Check every published file against results/checksums.sha256 (no re-running)."""
    lines = [l for l in (RESULTS / "checksums.sha256").read_text().splitlines() if l.strip()]
    bad = []
    for line in lines:
        h, rel = line.split("  ", 1)
        p = ROOT / rel
        if not p.exists():
            bad.append(f"{rel} (missing)")
        elif sha256(p) != h:
            bad.append(f"{rel} (hash mismatch)")
    print(f"{len(lines)} entries checked, {len(bad)} mismatches", flush=True)
    for item in bad:
        print(f"  FAIL {item}", flush=True)
    return 1 if bad else 0


def main():
    if "--verify" in sys.argv:
        sys.exit(verify_checksums())

    t0 = time.time()
    for script in ("exp1_crypto_benchmark.py", "exp2_migration_model.py"):
        print(f"running {script} ...", flush=True)
        r = subprocess.run([sys.executable, "-u", str(ROOT / script)], cwd=ROOT)
        if r.returncode != 0:
            raise SystemExit(f"{script} failed with code {r.returncode}")

    n = write_checksums()
    print(f"wrote {n} checksums -> results/checksums.sha256", flush=True)
    print(f"total runtime {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

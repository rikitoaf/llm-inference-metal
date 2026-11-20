#!/usr/bin/env python3
"""Cap the total size of Splash's persistent prefix cache (~/Library/Caches/Splash/prefix-cache).

Splash caps each cache namespace at --max-cache-disk (24G here), but every model, KV format and quantizer
revision gets its own namespace, and old ones are never removed: 101 GB had piled up by 2026-10-15. This keeps
the most recently used namespaces whose sizes fit the budget and deletes the rest. The cache is only an
accelerator: a deleted namespace is rebuilt as prompts are prefilled again.

    prune_splash_cache.py [--budget 48G] [--dry-run]

scripts/serve-splash.sh runs it before every start, with SPLASH_CACHE_TOTAL_DISK as the budget. It never runs
while a Splash engine is running, and never touches a namespace whose lock another process holds.
"""
import argparse, fcntl, os, pathlib, shutil, subprocess, sys, time

ROOT = pathlib.Path.home() / "Library" / "Caches" / "Splash" / "prefix-cache"


def parse_size(text):
    units = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
    text = text.strip().upper().removesuffix("B")
    if text and text[-1] in units:
        return int(float(text[:-1]) * units[text[-1]])
    return int(text)


def usage(path):
    """Bytes on disk (allocated blocks, so sparse slot files count as what they use)."""
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_blocks * 512
        except OSError:
            pass
    return total


def last_used(path):
    times = [path.stat().st_mtime]
    for p in path.rglob("*"):
        try:
            times.append(p.stat().st_mtime)
        except OSError:
            pass
    return max(times)


def locked(path):
    """Whether another process holds the namespace's lock file (Splash holds it while serving from it)."""
    lock = path / "lock"
    if not lock.exists():
        return False
    try:
        with open(lock, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f, fcntl.LOCK_UN)
        return False
    except OSError:
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", default=os.environ.get("SPLASH_CACHE_TOTAL_DISK", "48G"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", type=pathlib.Path, default=ROOT, help=argparse.SUPPRESS)   # tests
    a = ap.parse_args()
    budget = parse_size(a.budget)
    root = a.root
    if not root.is_dir():
        return 0
    if subprocess.run(["pgrep", "-f", "serve-native"], capture_output=True).returncode == 0:
        print("prune_splash_cache: a Splash engine is running; nothing pruned", file=sys.stderr)
        return 0
    spaces = sorted((p for p in root.iterdir() if p.is_dir()), key=last_used, reverse=True)
    kept = 0
    for space in spaces:
        size = usage(space)
        if kept + size <= budget or locked(space):
            kept += size
            continue
        print(f"prune_splash_cache: {'would delete' if a.dry_run else 'deleting'} {space.name} "
              f"({size / (1 << 30):.1f} GiB, last used {time.strftime('%Y-%m-%d', time.localtime(last_used(space)))})",
              file=sys.stderr)
        if not a.dry_run:
            shutil.rmtree(space, ignore_errors=True)
    print(f"prune_splash_cache: {kept / (1 << 30):.1f} GiB kept of a {budget / (1 << 30):.0f} GiB budget", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

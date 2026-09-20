"""Validate/copy the cleaned submission agent.

Usage:
    python make_patched.py
    python make_patched.py my_agent.py my_agent_submission.py
"""
from pathlib import Path
import py_compile
import shutil
import sys

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("my_agent.py")
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("my_agent_submission.py")

if not src.exists():
    raise SystemExit(f"source not found: {src}")

py_compile.compile(str(src), doraise=True)
if src.resolve() != dst.resolve():
    shutil.copyfile(src, dst)

print(f"validated: {src}")
print(f"output:    {dst}")

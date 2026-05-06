#!/usr/bin/env python
from __future__ import annotations

import subprocess


if __name__ == "__main__":
    subprocess.run(["python", "scripts/run_batch.py", "encoder"], check=True)
    subprocess.run(["python", "scripts/run_batch.py", "robustness"], check=True)


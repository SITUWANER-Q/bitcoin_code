#!/usr/bin/env python
from __future__ import annotations

import subprocess


if __name__ == "__main__":
    subprocess.run(["python", "scripts/run_batch.py", "appendix"], check=True)


#!/usr/bin/env python3
"""Full-1024 completion-aligned MUSE depth/capacity sweep.

This is a thin, provenance-aware specialization of the frozen completion-
aligned implementation. It preserves six training exposures from the 128-row
pilot by scaling A1/A2 steps fourfold for 512 rows per corpus.
"""
import authorize_muse_recap_full1024 as authorization
import sweep_muse_recap_completion_aligned as aligned


aligned.VERSION = "muse-completion-aligned-full1024-v1"
aligned.VARIANTS = {
    "d4": {"depth": 4, "rank": 32, "a1_steps": 1536, "a2_steps": 1152},
    "d6": {"depth": 6, "rank": 32, "a1_steps": 1536, "a2_steps": 1152},
}
aligned.POINTS = {
    "light": (0.50, 0.50, 0.001),
    "a1light": (0.75, 0.50, 0.001),
    "balanced": (0.75, 0.75, 0.001),
    "strong": (1.00, 0.75, 0.001),
}
# Both validate_source() and train() call through this module reference.
aligned.offline.validate_authorized_rows = authorization.validate_rows


if __name__ == "__main__":
    aligned.main()

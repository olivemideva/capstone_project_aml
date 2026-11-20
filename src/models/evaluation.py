"""Shared validation mask provenance for model comparisons."""

import hashlib

import pandas as pd


def mask_fingerprint(counties: pd.Series, mask: pd.DataFrame) -> str:
    if not counties.index.equals(mask.index):
        raise ValueError("County identifiers and mask must have aligned indices")
    digest = hashlib.sha256()
    digest.update("\0".join(mask.columns).encode("utf-8"))
    for code, row in zip(counties, mask.to_numpy(dtype=bool)):
        digest.update(b"\0")
        digest.update(str(code).encode("utf-8"))
        digest.update(row.tobytes())
    return digest.hexdigest()

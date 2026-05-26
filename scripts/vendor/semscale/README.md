# SemScale Wordfish (vendored core)

## Provenance

- **Source:** [umanlp/SemScale](https://github.com/umanlp/SemScale) — `wfcode/scaler.py` (WordfishScaler)
- **Reference:** Slapin & Proksch (2008), Wordfish Poisson scaling
- **Adaptation:** Logic reimplemented in [`src/wordfish.py`](../../../src/wordfish.py); we do **not** import SemScale's `corpus.py` (NLTK/scipy dependencies).

## Licence

Check the upstream SemScale repository for licence terms before redistribution. This project uses an adapted numerical core only.

## Validation

The vendored numerical routine uses **NumPy only** (no `scipy` import). Run:

```bash
rg 'scipy' scripts/vendor/semscale/ src/wordfish.py
```

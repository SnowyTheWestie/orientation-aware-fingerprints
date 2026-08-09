# Orientation-Aware Molecular Fingerprints

This repository contains the reference implementation, datasets, and analysis notebooks accompanying the manuscript *Orientation-Aware Molecular Fingerprints*.

The project introduces two molecular fingerprints that encode molecular structure relative to a defined attachment position:

- **COAF**: Circular Orientation-Aware Fingerprint, based on rooted Weisfeiler–Lehman refinement.
- **POAF**: Path-Based Orientation-Aware Fingerprint, based on directed molecular paths.

## Repository contents

```text
.
├── coaf.py
├── data/
│   ├── Combinatorial_FS_DEL.csv
│   ├── heterocycles.csv
│   ├── LibraryComparison.csv
│   └── SMILES_1aa.csv
├── notebooks/
│   ├── 01_similarity_retention.ipynb
│   ├── 02_two_component_similarity.ipynb
│   ├── 03b_matched_attachment_positions.ipynb
│   └── 04_matched_topology_benchmark.ipynb
└── validation/
```

## Installation

Python 3.11 is recommended. Install the required packages with:

```bash
python -m pip install -r requirements.txt
```

The manuscript calculations used Python 3.11.14 and RDKit 2025.09.4.

## Input convention

Orientation-aware fingerprints require a SMILES string containing exactly one explicit attachment marker. By default, the marker is mercury, `[Hg]`.

```text
[Hg]CCO
```

The molecule must contain one connected component and exactly one marker atom. The marker defines the root node, and bonds are directed away from it according to shortest-path distance.

## Basic usage

```python
import coaf

smiles = "[Hg]CCO"

coaf_fp = coaf.coaf_fingerprint_from_smiles(
    smiles,
    radius=3,
    n_bits=1024,
)

poaf_fp = coaf.directed_linear_fingerprint_from_smiles(
    smiles,
    max_path_length=6,
    n_bits=1024,
)
```

Calculate Tanimoto similarity:

```python
first = coaf.coaf_fingerprint_from_smiles("[Hg]CCO")
second = coaf.coaf_fingerprint_from_smiles("[Hg]CCN")
similarity = coaf.tanimoto_similarity(first, second)
```

Feature folding uses SHA-256, making fingerprint generation deterministic for the same input, parameters, and software environment.

## Datasets

- `SMILES_1aa.csv`: carboxylic acid building blocks and assembled products used for the conserved-scaffold similarity-retention analysis.
- `Combinatorial_FS_DEL.csv`: building blocks and enumerated products used for the two-component combinatorial-library analysis.
- `heterocycles.csv`: fused heteroaromatic structures and pair classifications used to compare attachment-position and conventional chemical changes.
- `LibraryComparison.csv`: matched L1 and L2 library representations used for the attachment-topology analyses.

The filenames are referenced directly by the notebooks and should not be changed.

## Reproducing the analyses

Start Jupyter from the repository root so that `coaf.py`, `data/`, and `validation/` resolve correctly:

```bash
jupyter lab
```

Run the notebooks in numerical order. Each notebook creates its own output directory under `results/`.

The principal manuscript settings were:

| Fingerprint | Parameters |
|---|---|
| COAF6 | `radius=3`, `n_bits=1024` |
| POAF6 | `max_path_length=6`, `n_bits=1024` |
| ECFP6 and ECFP6-RN | `radius=3`, `n_bits=1024` |

Randomized sampling, shuffling, and partitioning in the analysis workflows use a fixed seed of 123 unless otherwise stated.

## Citation

If you use this code or these datasets, please cite:

> Franzini, R. M. *Orientation-Aware Molecular Fingerprints*. Manuscript submitted for publication.

The citation will be updated when the article is published.


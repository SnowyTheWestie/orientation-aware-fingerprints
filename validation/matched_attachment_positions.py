"""Matched attachment-position benchmark for rooted heterocycles.

The benchmark defines two mutually exclusive pair classes:

``same_scaffold_different_position``
    The canonical scaffold is identical but the labeled attachment position
    differs.  This is the primary attachment-position test.
``same_position_different_scaffold``
    The attachment-position label is identical but the canonical scaffold
    differs.  This is the chemical-change control.

The module contains data validation, fingerprinting, pair construction,
summary statistics, clustered bootstrap intervals, and result serialization.
Plotting deliberately remains in the accompanying notebook.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase

from .similarity_retention import (
    BenchmarkConfig,
    _suppress_rdkit_messages,
    _validate_marker_smiles,
    build_fingerprint_functions,
    compute_similarity_matrix,
)


PAIR_LABELS = {
    "same_scaffold_different_position": "Same scaffold, different position",
    "same_position_different_scaffold": "Same position, different scaffold",
}


@dataclass(frozen=True)
class MatchedAttachmentConfig:
    n_bits: int = 1024
    radius: int = 3
    poaf_max_path_length: int = 6
    marker_symbol: str = "Hg"
    bootstrap_replicates: int = 2000
    random_seed: int = 123
    ranking_size: int = 20

    def validate(self) -> None:
        if self.n_bits <= 0:
            raise ValueError("n_bits must be greater than zero")
        if self.radius < 0:
            raise ValueError("radius must be non-negative")
        if self.poaf_max_path_length < 1:
            raise ValueError("poaf_max_path_length must be at least 1 edge")
        if self.bootstrap_replicates <= 0:
            raise ValueError("bootstrap_replicates must be greater than zero")
        if self.ranking_size <= 0:
            raise ValueError("ranking_size must be greater than zero")


@dataclass(frozen=True)
class MatchedAttachmentDataset:
    frame: pd.DataFrame
    source_path: Path
    source_sha256: str
    n_blank_rows_removed: int


@dataclass(frozen=True)
class MatchedAttachmentResults:
    molecule_manifest: pd.DataFrame
    pairwise_comparison: pd.DataFrame
    group_summary: pd.DataFrame
    bootstrap_summary: pd.DataFrame
    fingerprint_summary: pd.DataFrame
    ranked_coaf_gain: pd.DataFrame
    ranked_poaf_gain: pd.DataFrame
    ranked_ecfp_hg_gain: pd.DataFrame
    ranked_ecfp_gain: pd.DataFrame
    run_metadata: Mapping[str, object]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_smiles(smiles: str, *, row_number: int, column: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(
            f"RDKit could not parse CSV row {row_number}, column {column}: {smiles!r}"
        )
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def load_matched_attachment_dataset(
    csv_path: str | Path,
    *,
    marker_symbol: str = "Hg",
) -> MatchedAttachmentDataset:
    """Load, clean, validate, and canonicalize the three-column dataset."""
    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")

    source = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = ["SMILES_Structure", "SMILES_Scaffold", "Position"]
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = source[required].copy()
    for column in required:
        frame[column] = frame[column].str.strip()
    all_blank = frame.eq("").all(axis=1)
    n_blank = int(all_blank.sum())
    frame = frame.loc[~all_blank].copy()
    if frame.empty:
        raise ValueError("Input CSV contains no nonblank records")
    partial_blank = frame.eq("").any(axis=1)
    if partial_blank.any():
        rows = (frame.index[partial_blank] + 2).tolist()
        raise ValueError(f"Partially blank records at CSV rows: {rows[:10]}")

    frame.insert(0, "source_row", frame.index.to_numpy(dtype=int) + 2)
    frame.reset_index(drop=True, inplace=True)
    frame.insert(0, "molecule_id", [f"HET_{i:04d}" for i in range(len(frame))])

    canonical_structures = []
    canonical_scaffolds = []
    with _suppress_rdkit_messages():
        for row in frame.itertuples(index=False):
            _validate_marker_smiles(
                row.SMILES_Structure,
                row_number=int(row.source_row),
                column="SMILES_Structure",
                marker_symbol=marker_symbol,
            )
            canonical_structures.append(
                _canonical_smiles(
                    row.SMILES_Structure,
                    row_number=int(row.source_row),
                    column="SMILES_Structure",
                )
            )
            canonical_scaffolds.append(
                _canonical_smiles(
                    row.SMILES_Scaffold,
                    row_number=int(row.source_row),
                    column="SMILES_Scaffold",
                )
            )

    frame["canonical_structure"] = canonical_structures
    frame["canonical_scaffold"] = canonical_scaffolds
    return MatchedAttachmentDataset(
        frame=frame,
        source_path=path,
        source_sha256=_sha256_file(path),
        n_blank_rows_removed=n_blank,
    )


def _fingerprints(dataset: MatchedAttachmentDataset, config: MatchedAttachmentConfig):
    fp_config = BenchmarkConfig(
        n_bits=config.n_bits,
        ecfp_radius=config.radius,
        coaf_radius=config.radius,
        linear_max_path_length=config.poaf_max_path_length,
        marker_symbol=config.marker_symbol,
    )
    functions = build_fingerprint_functions(fp_config)
    fingerprints = {}
    rows = []
    for descriptor, function_name in (
        ("ECFP", "ECFP_O"),
        ("ECFP_Hg", "ECFP_HG"),
        ("COAF", "COAF"),
        ("POAF", "DIRECTED_LINEAR"),
    ):
        function = functions[function_name]
        with _suppress_rdkit_messages():
            values = [function(s) for s in dataset.frame["SMILES_Structure"]]
        fingerprints[descriptor] = values
        for fingerprint in values:
            on_bits = int(fingerprint.GetNumOnBits())
            rows.append(
                {
                    "descriptor": descriptor,
                    "n_bits": fingerprint.GetNumBits(),
                    "n_bits_set": on_bits,
                    "bit_density": on_bits / fingerprint.GetNumBits(),
                }
            )
    density = pd.DataFrame(rows)
    summary = (
        density.groupby("descriptor", sort=False)
        .agg(
            n_molecules=("n_bits_set", "size"),
            n_bits=("n_bits", "first"),
            mean_bits_set=("n_bits_set", "mean"),
            std_bits_set=("n_bits_set", "std"),
            mean_bit_density=("bit_density", "mean"),
            std_bit_density=("bit_density", "std"),
        )
        .reset_index()
    )
    return fingerprints, summary


def _pairwise_table(dataset, similarity_matrices) -> pd.DataFrame:
    frame = dataset.frame
    first, second = np.triu_indices(len(frame), k=1)
    scaffold = frame["canonical_scaffold"].to_numpy()
    structure = frame["canonical_structure"].to_numpy()
    position = frame["Position"].to_numpy()
    same_scaffold = scaffold[first] == scaffold[second]
    same_position = position[first] == position[second]
    group_1 = same_scaffold & ~same_position
    group_2 = same_position & ~same_scaffold
    # Symmetry can make two nominal positions chemically identical after
    # canonicalization. Keep those records in the audit, but do not count an
    # identical molecular structure as an attachment-discrimination pair.
    different_structure = structure[first] != structure[second]
    selected = (group_1 | group_2) & different_structure
    first = first[selected]
    second = second[selected]
    pair_group = np.where(
        group_1[selected],
        "same_scaffold_different_position",
        "same_position_different_scaffold",
    )

    distances = {
        descriptor: 1.0 - matrix[first, second]
        for descriptor, matrix in similarity_matrices.items()
    }
    ecfp_distance = distances["ECFP"]
    identifiers = frame["molecule_id"].to_numpy()
    structures = frame["SMILES_Structure"].to_numpy()

    scaffold_1 = scaffold[first]
    scaffold_2 = scaffold[second]
    scaffold_pair = np.array(
        [" || ".join(sorted((a, b))) for a, b in zip(scaffold_1, scaffold_2)],
        dtype=object,
    )
    cluster_id = np.where(group_1[selected], scaffold_1, scaffold_pair)

    output = pd.DataFrame(
        {
            "pair_id": np.arange(len(first), dtype=int),
            "pair_group": pair_group,
            "pair_group_label": [PAIR_LABELS[value] for value in pair_group],
            "cluster_id": cluster_id,
            "index_1": first,
            "index_2": second,
            "molecule_id_1": identifiers[first],
            "molecule_id_2": identifiers[second],
            "position_1": position[first],
            "position_2": position[second],
            "canonical_scaffold_1": scaffold_1,
            "canonical_scaffold_2": scaffold_2,
            "SMILES_1": structures[first],
            "SMILES_2": structures[second],
        }
    )
    for descriptor, values in distances.items():
        output[f"{descriptor}_distance"] = values
    for descriptor in ("COAF", "POAF", "ECFP_Hg"):
        delta = distances[descriptor] - ecfp_distance
        output[f"{descriptor}_minus_ECFP_distance"] = delta
        output[f"{descriptor}_more_dissimilar_than_ECFP"] = delta > 1e-12
    return output


def _group_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, values in pairwise.groupby("pair_group", sort=False):
        for descriptor in ("ECFP", "ECFP_Hg", "COAF", "POAF"):
            distance = values[f"{descriptor}_distance"]
            if descriptor == "ECFP":
                delta = pd.Series(np.zeros(len(values)), index=values.index)
                fraction_greater = np.nan
            else:
                delta = values[f"{descriptor}_minus_ECFP_distance"]
                fraction_greater = float((delta > 1e-12).mean())
            rows.append(
                {
                    "pair_group": group,
                    "pair_group_label": PAIR_LABELS[group],
                    "descriptor": descriptor,
                    "n_pairs": len(values),
                    "n_clusters": values["cluster_id"].nunique(),
                    "mean_distance": distance.mean(),
                    "median_distance": distance.median(),
                    "std_distance": distance.std(),
                    "mean_difference_from_ECFP": delta.mean(),
                    "median_difference_from_ECFP": delta.median(),
                    "std_difference_from_ECFP": delta.std(),
                    "fraction_more_dissimilar_than_ECFP": fraction_greater,
                }
            )
    return pd.DataFrame(rows)


def _cluster_bootstrap(pairwise: pd.DataFrame, config: MatchedAttachmentConfig):
    rng = np.random.default_rng(config.random_seed)
    output = []
    for group, values in pairwise.groupby("pair_group", sort=False):
        for descriptor in ("COAF", "POAF", "ECFP_Hg"):
            difference_column = f"{descriptor}_minus_ECFP_distance"
            clusters = {
                name: block[difference_column].to_numpy()
                for name, block in values.groupby("cluster_id", sort=False)
            }
            names = np.array(list(clusters), dtype=object)
            means = np.empty(config.bootstrap_replicates)
            medians = np.empty(config.bootstrap_replicates)
            for replicate in range(config.bootstrap_replicates):
                sampled = rng.choice(names, size=len(names), replace=True)
                delta = np.concatenate([clusters[name] for name in sampled])
                means[replicate] = delta.mean()
                medians[replicate] = np.median(delta)
            for statistic, distribution in (("mean", means), ("median", medians)):
                output.append(
                    {
                        "pair_group": group,
                        "pair_group_label": PAIR_LABELS[group],
                        "descriptor": descriptor,
                        "reference_descriptor": "ECFP",
                        "statistic": statistic,
                        "estimate": (
                            values[difference_column].mean()
                            if statistic == "mean"
                            else values[difference_column].median()
                        ),
                        "bootstrap_standard_error": distribution.std(ddof=1),
                        "ci_95_lower": np.quantile(distribution, 0.025),
                        "ci_95_upper": np.quantile(distribution, 0.975),
                        "n_bootstrap_replicates": config.bootstrap_replicates,
                        "resampling_unit": (
                            "scaffold" if group == "same_scaffold_different_position"
                            else "unordered scaffold pair"
                        ),
                    }
                )
    return pd.DataFrame(output)


def _ensure_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}. Choose a new run directory "
            "or pass overwrite=True explicitly."
        )
    path.mkdir(parents=True, exist_ok=True)


def run_matched_attachment_benchmark(
    dataset: MatchedAttachmentDataset | str | Path,
    *,
    config: MatchedAttachmentConfig | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> MatchedAttachmentResults:
    active = config or MatchedAttachmentConfig()
    active.validate()
    if isinstance(dataset, (str, Path)):
        dataset = load_matched_attachment_dataset(
            dataset, marker_symbol=active.marker_symbol
        )

    fingerprints, fp_summary = _fingerprints(dataset, active)
    similarity_matrices = {
        descriptor: compute_similarity_matrix(values)
        for descriptor, values in fingerprints.items()
    }
    pairwise = _pairwise_table(dataset, similarity_matrices)
    ranking_size = min(active.ranking_size, len(pairwise))
    metadata = {
        "analysis": "matched heterocycle attachment-position discrimination",
        "input_file": str(dataset.source_path),
        "input_sha256": dataset.source_sha256,
        "n_blank_rows_removed": dataset.n_blank_rows_removed,
        "n_molecules": len(dataset.frame),
        "n_unique_scaffolds": int(dataset.frame["canonical_scaffold"].nunique()),
        "n_unique_canonical_structures": int(
            dataset.frame["canonical_structure"].nunique()
        ),
        "parameters": asdict(active),
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    results = MatchedAttachmentResults(
        molecule_manifest=dataset.frame.copy(),
        pairwise_comparison=pairwise,
        group_summary=_group_summary(pairwise),
        bootstrap_summary=_cluster_bootstrap(pairwise, active),
        fingerprint_summary=fp_summary,
        ranked_coaf_gain=pairwise.nlargest(
            ranking_size, "COAF_minus_ECFP_distance"
        ).reset_index(drop=True),
        ranked_poaf_gain=pairwise.nlargest(
            ranking_size, "POAF_minus_ECFP_distance"
        ).reset_index(drop=True),
        ranked_ecfp_hg_gain=pairwise.nlargest(
            ranking_size, "ECFP_Hg_minus_ECFP_distance"
        ).reset_index(drop=True),
        ranked_ecfp_gain=pairwise.nsmallest(
            ranking_size, "COAF_minus_ECFP_distance"
        ).reset_index(drop=True),
        run_metadata=metadata,
    )

    if output_dir is not None:
        path = Path(output_dir).expanduser().resolve()
        _ensure_output_directory(path, overwrite)
        for filename, table in {
            "molecule_manifest.csv": results.molecule_manifest,
            "pairwise_comparison.csv": results.pairwise_comparison,
            "group_summary.csv": results.group_summary,
            "bootstrap_summary.csv": results.bootstrap_summary,
            "fingerprint_summary.csv": results.fingerprint_summary,
            "ranked_coaf_gain.csv": results.ranked_coaf_gain,
            "ranked_poaf_gain.csv": results.ranked_poaf_gain,
            "ranked_ecfp_hg_gain.csv": results.ranked_ecfp_hg_gain,
            "ranked_ecfp_gain.csv": results.ranked_ecfp_gain,
        }.items():
            table.to_csv(path / filename, index=False)
        (path / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
    return results

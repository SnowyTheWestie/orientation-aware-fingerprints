"""Two-variable-building-block similarity benchmark for a combinatorial DEL.

The benchmark asks how well similarities calculated independently for BB1 and
BB2 predict the similarity of their assembled product. All identifier-level
library members are retained, including stereoisomers and structures that a
descriptor may encode identically.

The complete product library is deterministically shuffled and split into
balanced batches. Every product occurs in exactly one batch, every descriptor
uses the same batches, and all unique pairs within each batch are evaluated.
This uses the full library without allocating an infeasible full product
similarity matrix.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
from rdkit import DataStructs, rdBase

from .similarity_retention import (
    BenchmarkConfig,
    DESCRIPTOR_NAMES,
    _suppress_rdkit_messages,
    _validate_marker_smiles,
    build_fingerprint_functions,
    compare_similarity_matrices,
    compute_similarity_matrix,
)


REQUIRED_COLUMNS: Tuple[str, ...] = (
    "BB1",
    "BB2",
    "SMILES_Product",
    "SMILES_BB1",
    "SMILES_BB2",
)
AGGREGATION_METHODS: Tuple[str, ...] = (
    "arithmetic_mean",
    "geometric_mean",
)


@dataclass(frozen=True)
class TwoComponentConfig:
    """Fingerprint, batching, aggregation, and evaluation parameters."""

    n_bits: int = 1024
    ecfp_radius: int = 3
    coaf_radius: int = 3
    linear_max_path_length: int = 6
    marker_symbol: str = "Hg"
    neighbor_k: Tuple[int, ...] = (5, 10)
    batch_size: int = 1000
    random_seed: int = 123
    aggregation_methods: Tuple[str, ...] = AGGREGATION_METHODS
    representative_pair_sample_size: int = 10_000

    def fingerprint_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            n_bits=self.n_bits,
            ecfp_radius=self.ecfp_radius,
            coaf_radius=self.coaf_radius,
            linear_max_path_length=self.linear_max_path_length,
            marker_symbol=self.marker_symbol,
            neighbor_k=self.neighbor_k,
        )

    def validate(self, n_products: int) -> None:
        self.fingerprint_config().validate(min(n_products, self.batch_size))
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2")
        if self.representative_pair_sample_size <= 0:
            raise ValueError("representative_pair_sample_size must be positive")
        unknown = [
            method
            for method in self.aggregation_methods
            if method not in AGGREGATION_METHODS
        ]
        if unknown:
            raise ValueError(f"Unknown aggregation methods: {unknown}")
        if not self.aggregation_methods:
            raise ValueError("At least one aggregation method is required")


@dataclass(frozen=True)
class TwoComponentDataset:
    """Validated full-library records and component lookup tables."""

    frame: pd.DataFrame
    bb1_table: pd.DataFrame
    bb2_table: pd.DataFrame
    source_path: Path
    source_sha256: str

    @property
    def n_products(self) -> int:
        return len(self.frame)


@dataclass(frozen=True)
class TwoComponentResults:
    """Tidy outputs from the full-library batched benchmark."""

    per_batch_metrics: pd.DataFrame
    descriptor_summary: pd.DataFrame
    paired_descriptor_differences: pd.DataFrame
    batch_manifest: pd.DataFrame
    representative_pair_samples: pd.DataFrame
    dataset_summary: pd.DataFrame
    run_metadata: Mapping[str, object]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _component_table(
    frame: pd.DataFrame,
    *,
    id_column: str,
    smiles_column: str,
) -> pd.DataFrame:
    counts = frame.groupby(id_column, sort=False)[smiles_column].nunique(dropna=False)
    inconsistent = counts[counts != 1]
    if not inconsistent.empty:
        raise ValueError(
            f"Each {id_column} identifier must map to exactly one {smiles_column}; "
            f"inconsistent identifiers: {inconsistent.index.tolist()[:10]}"
        )
    return (
        frame[[id_column, smiles_column]]
        .drop_duplicates(subset=[id_column], keep="first")
        .reset_index(drop=True)
    )


def load_two_component_dataset(
    csv_path: str | Path,
    *,
    marker_symbol: str = "Hg",
) -> TwoComponentDataset:
    """Load and validate the five-column combinatorial-library CSV.

    No rows are deduplicated. Identifier-level stereoisomers and other isomers
    remain independent library members even if a descriptor cannot distinguish
    them.
    """
    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(frame.columns) != REQUIRED_COLUMNS:
        raise ValueError(
            f"Expected columns in order {list(REQUIRED_COLUMNS)}; "
            f"found {frame.columns.tolist()}"
        )
    if frame.empty:
        raise ValueError("Input CSV contains no product rows")
    missing = {
        column: int(frame[column].str.strip().eq("").sum())
        for column in REQUIRED_COLUMNS
    }
    missing = {column: count for column, count in missing.items() if count}
    if missing:
        raise ValueError(f"Missing values found: {missing}")
    if frame[["BB1", "BB2"]].duplicated().any():
        duplicate_rows = (frame.index[frame[["BB1", "BB2"]].duplicated()] + 2).tolist()
        raise ValueError(
            "Duplicate BB1/BB2 identifier combinations at CSV rows "
            f"{duplicate_rows[:10]}"
        )

    bb1_table = _component_table(
        frame, id_column="BB1", smiles_column="SMILES_BB1"
    )
    bb2_table = _component_table(
        frame, id_column="BB2", smiles_column="SMILES_BB2"
    )
    expected_size = len(bb1_table) * len(bb2_table)
    if len(frame) != expected_size:
        raise ValueError(
            "Library is not a complete BB1 x BB2 Cartesian product: "
            f"found {len(frame)} rows, expected {expected_size}"
        )

    unique_structures = {
        "SMILES_BB1": bb1_table["SMILES_BB1"].drop_duplicates(),
        "SMILES_BB2": bb2_table["SMILES_BB2"].drop_duplicates(),
        "SMILES_Product": frame["SMILES_Product"].drop_duplicates(),
    }
    with _suppress_rdkit_messages():
        for column, structures in unique_structures.items():
            for position, smiles in enumerate(structures, start=1):
                _validate_marker_smiles(
                    smiles,
                    row_number=position,
                    column=column,
                    marker_symbol=marker_symbol,
                )

    output = frame.copy()
    output.insert(0, "source_row", np.arange(2, len(output) + 2, dtype=int))
    output.insert(
        0,
        "product_id",
        [f"P_{index:06d}" for index in range(1, len(output) + 1)],
    )
    return TwoComponentDataset(
        frame=output,
        bb1_table=bb1_table,
        bb2_table=bb2_table,
        source_path=path,
        source_sha256=_sha256_file(path),
    )


def make_balanced_batches(
    n_products: int,
    *,
    batch_size: int,
    random_seed: int,
) -> Tuple[np.ndarray, ...]:
    """Shuffle once and partition all product indices into balanced batches."""
    if n_products < 2:
        raise ValueError("At least two products are required")
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2")
    n_batches = math.ceil(n_products / batch_size)
    permutation = np.random.default_rng(random_seed).permutation(n_products)
    return tuple(np.asarray(batch, dtype=int) for batch in np.array_split(permutation, n_batches))


def combine_component_similarities(
    first: np.ndarray,
    second: np.ndarray,
    *,
    method: str,
) -> np.ndarray:
    """Combine aligned BB1 and BB2 similarity matrices."""
    if first.shape != second.shape:
        raise ValueError("Component similarity matrices must have identical shapes")
    if method == "arithmetic_mean":
        return (first + second) / 2.0
    if method == "geometric_mean":
        return np.sqrt(np.clip(first, 0.0, None) * np.clip(second, 0.0, None))
    raise ValueError(f"Unknown aggregation method: {method}")


def _upper_triangle(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices_from(matrix, k=1)]


def _component_similarity_matrices(
    dataset: TwoComponentDataset,
    fingerprint_function,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[str, int]]:
    with _suppress_rdkit_messages():
        bb1_fingerprints = [
            fingerprint_function(smiles)
            for smiles in dataset.bb1_table["SMILES_BB1"]
        ]
        bb2_fingerprints = [
            fingerprint_function(smiles)
            for smiles in dataset.bb2_table["SMILES_BB2"]
        ]
    bb1_lookup = {
        identifier: index
        for index, identifier in enumerate(dataset.bb1_table["BB1"])
    }
    bb2_lookup = {
        identifier: index
        for index, identifier in enumerate(dataset.bb2_table["BB2"])
    }
    return (
        compute_similarity_matrix(bb1_fingerprints),
        compute_similarity_matrix(bb2_fingerprints),
        bb1_lookup,
        bb2_lookup,
    )


def _summary_table(per_batch: pd.DataFrame) -> pd.DataFrame:
    dimensions = {"descriptor", "aggregation", "batch", "n_products", "n_pairs"}
    metrics = [column for column in per_batch.columns if column not in dimensions]
    rows = []
    for (descriptor, aggregation), group in per_batch.groupby(
        ["descriptor", "aggregation"], sort=False
    ):
        row: Dict[str, object] = {
            "descriptor": descriptor,
            "aggregation": aggregation,
            "n_batches": len(group),
            "n_products_total": int(group["n_products"].sum()),
            "n_pairs_total": int(group["n_pairs"].sum()),
        }
        for metric in metrics:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_differences(
    per_batch: pd.DataFrame,
    *,
    reference: str = "ECFP_O",
) -> pd.DataFrame:
    dimensions = {"descriptor", "aggregation", "batch", "n_products", "n_pairs"}
    metrics = [column for column in per_batch.columns if column not in dimensions]
    reference_rows = per_batch[per_batch["descriptor"] == reference].set_index(
        ["aggregation", "batch"]
    )
    rows = []
    for descriptor in DESCRIPTOR_NAMES:
        if descriptor == reference:
            continue
        comparison = per_batch[per_batch["descriptor"] == descriptor].set_index(
            ["aggregation", "batch"]
        )
        for index in reference_rows.index:
            aggregation, batch = index
            row: Dict[str, object] = {
                "descriptor": descriptor,
                "reference_descriptor": reference,
                "aggregation": aggregation,
                "batch": int(batch),
            }
            for metric in metrics:
                row[f"delta_{metric}"] = float(
                    comparison.loc[index, metric] - reference_rows.loc[index, metric]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _dataset_summary(dataset: TwoComponentDataset) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "n_products": dataset.n_products,
                "n_bb1_ids": len(dataset.bb1_table),
                "n_bb2_ids": len(dataset.bb2_table),
                "n_unique_bb1_smiles": dataset.bb1_table["SMILES_BB1"].nunique(),
                "n_unique_bb2_smiles": dataset.bb2_table["SMILES_BB2"].nunique(),
                "n_unique_product_smiles": dataset.frame["SMILES_Product"].nunique(),
                "all_identifier_combinations_present": True,
                "rows_removed": 0,
            }
        ]
    )


def _metadata(
    dataset: TwoComponentDataset,
    config: TwoComponentConfig,
    batches: Sequence[np.ndarray],
) -> Dict[str, object]:
    return {
        "analysis": "two-variable-building-block product similarity",
        "input_file": str(dataset.source_path),
        "input_sha256": dataset.source_sha256,
        "n_products": dataset.n_products,
        "n_bb1_ids": len(dataset.bb1_table),
        "n_bb2_ids": len(dataset.bb2_table),
        "rows_removed": 0,
        "descriptors": list(DESCRIPTOR_NAMES),
        "parameters": asdict(config),
        "batching": {
            "n_batches": len(batches),
            "minimum_batch_size": min(len(batch) for batch in batches),
            "maximum_batch_size": max(len(batch) for batch in batches),
            "all_products_used_once": True,
        },
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }


def _save_results(results: TwoComponentResults, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "dataset_summary.csv": results.dataset_summary,
        "batch_manifest.csv": results.batch_manifest,
        "per_batch_metrics.csv": results.per_batch_metrics,
        "descriptor_summary.csv": results.descriptor_summary,
        "paired_descriptor_differences.csv": results.paired_descriptor_differences,
        "representative_pair_samples.csv": results.representative_pair_samples,
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False)
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(results.run_metadata, handle, indent=2, sort_keys=True)


def run_two_component_benchmark(
    dataset: TwoComponentDataset,
    *,
    config: TwoComponentConfig | None = None,
    output_dir: str | Path | None = None,
) -> TwoComponentResults:
    """Run all descriptors and aggregations over the complete DEL.

    Product fingerprints are calculated for every row. The library is divided
    into balanced random batches solely to bound pairwise matrix memory. All
    unique pairs within each batch are included in the reported metrics.
    """
    active = config or TwoComponentConfig()
    active.validate(dataset.n_products)
    batches = make_balanced_batches(
        dataset.n_products,
        batch_size=active.batch_size,
        random_seed=active.random_seed,
    )
    smallest_batch = min(len(batch) for batch in batches)
    invalid_k = [k for k in active.neighbor_k if k >= smallest_batch]
    if invalid_k:
        raise ValueError(
            "Neighbor k must be smaller than every balanced batch; "
            f"smallest batch={smallest_batch}, invalid values={invalid_k}"
        )
    manifest_rows = []
    for batch_number, indices in enumerate(batches):
        for within_batch_order, row_index in enumerate(indices):
            row = dataset.frame.iloc[int(row_index)]
            manifest_rows.append(
                {
                    "batch": batch_number,
                    "within_batch_order": within_batch_order,
                    "product_id": row["product_id"],
                    "source_row": int(row["source_row"]),
                    "BB1": row["BB1"],
                    "BB2": row["BB2"],
                }
            )
    batch_manifest = pd.DataFrame(manifest_rows)

    fingerprint_functions = build_fingerprint_functions(active.fingerprint_config())
    metric_rows = []
    representative_rows = []

    for descriptor, fingerprint_function in fingerprint_functions.items():
        bb1_sim, bb2_sim, bb1_lookup, bb2_lookup = _component_similarity_matrices(
            dataset, fingerprint_function
        )
        with _suppress_rdkit_messages():
            product_fingerprints = [
                fingerprint_function(smiles)
                for smiles in dataset.frame["SMILES_Product"]
            ]

        for batch_number, indices in enumerate(batches):
            batch = dataset.frame.iloc[indices]
            product_sim = compute_similarity_matrix(
                [product_fingerprints[int(index)] for index in indices]
            )
            bb1_positions = np.asarray(
                [bb1_lookup[value] for value in batch["BB1"]], dtype=int
            )
            bb2_positions = np.asarray(
                [bb2_lookup[value] for value in batch["BB2"]], dtype=int
            )
            batch_bb1_sim = bb1_sim[np.ix_(bb1_positions, bb1_positions)]
            batch_bb2_sim = bb2_sim[np.ix_(bb2_positions, bb2_positions)]
            product_values = _upper_triangle(product_sim)

            for aggregation in active.aggregation_methods:
                predicted = combine_component_similarities(
                    batch_bb1_sim, batch_bb2_sim, method=aggregation
                )
                metrics = compare_similarity_matrices(
                    predicted,
                    product_sim,
                    neighbor_k=active.neighbor_k,
                )
                predicted_values = _upper_triangle(predicted)
                residual = predicted_values - product_values
                metrics["mae"] = float(np.mean(np.abs(residual)))
                metrics["rmse"] = float(np.sqrt(np.mean(np.square(residual))))
                metric_rows.append(
                    {
                        "descriptor": descriptor,
                        "aggregation": aggregation,
                        "batch": batch_number,
                        "n_products": len(indices),
                        "n_pairs": len(product_values),
                        **metrics,
                    }
                )

                if batch_number == 0:
                    sample_size = min(
                        active.representative_pair_sample_size,
                        len(product_values),
                    )
                    sample_rng = np.random.default_rng(
                        np.random.SeedSequence([active.random_seed, 0, 991])
                    )
                    sample_positions = sample_rng.choice(
                        len(product_values), size=sample_size, replace=False
                    )
                    representative_rows.extend(
                        {
                            "descriptor": descriptor,
                            "aggregation": aggregation,
                            "batch": 0,
                            "predicted_component_similarity": float(predicted_values[pos]),
                            "product_similarity": float(product_values[pos]),
                        }
                        for pos in sample_positions
                    )

    per_batch = pd.DataFrame(metric_rows)
    results = TwoComponentResults(
        per_batch_metrics=per_batch,
        descriptor_summary=_summary_table(per_batch),
        paired_descriptor_differences=_paired_differences(per_batch),
        batch_manifest=batch_manifest,
        representative_pair_samples=pd.DataFrame(representative_rows),
        dataset_summary=_dataset_summary(dataset),
        run_metadata=_metadata(dataset, active, batches),
    )
    if output_dir is not None:
        _save_results(results, Path(output_dir).expanduser().resolve())
    return results

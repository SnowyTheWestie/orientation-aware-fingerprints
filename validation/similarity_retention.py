"""Building-block-to-product similarity-retention benchmark.

The workflow compares four binary fingerprints on the same molecules and
ordered records:

``ECFP_O``
    Morgan/ECFP after replacing the explicit ``[Hg]`` attachment marker by O.
``ECFP_HG``
    Morgan/ECFP with the Hg marker retained as an attachment-site control.
``COAF``
    The rooted Weisfeiler--Lehman COAF descriptor.
``DIRECTED_LINEAR``
    The complementary root-directed linear/path descriptor.

For each product-partner column, the building-block similarity matrix is
compared with the product similarity matrix using rank correlation and
nearest-neighbor preservation.  The module contains no plotting or notebook
state; it returns tidy pandas tables and can optionally save an auditable run.
"""

from __future__ import annotations

import hashlib
import json
import platform
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Callable, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
from rdkit import Chem, DataStructs, rdBase
from scipy.stats import pearsonr, spearmanr

from coaf import (
    coaf_fingerprint_from_smiles,
    directed_linear_fingerprint_from_smiles,
    ecfp_fingerprint,
    fingerprint_to_bitvect,
)


DESCRIPTOR_NAMES: Tuple[str, ...] = (
    "ECFP_O",
    "ECFP_HG",
    "COAF",
    "DIRECTED_LINEAR",
)


@contextmanager
def _suppress_rdkit_messages():
    """Temporarily suppress RDKit parser diagnostics handled by validation.

    Some source SMILES contain conflicting directional-bond annotations. RDKit
    resolves these by setting the affected bond stereo to ``STEREONONE`` and
    writes a warning to the notebook. The benchmark does not use chirality,
    and parse failures are still caught explicitly below, so these diagnostics
    can be suppressed without hiding invalid records from the validation
    report.
    """
    blocker = rdBase.BlockLogs()
    try:
        yield
    finally:
        del blocker


@dataclass(frozen=True)
class BenchmarkConfig:
    """Primary fingerprint and evaluation parameters."""

    n_bits: int = 1024
    ecfp_radius: int = 3
    coaf_radius: int = 3
    linear_max_path_length: int = 6
    marker_symbol: str = "Hg"
    neighbor_k: Tuple[int, ...] = (5, 10)

    def validate(self, n_molecules: int) -> None:
        if self.n_bits <= 0:
            raise ValueError("n_bits must be greater than zero")
        if self.ecfp_radius < 0 or self.coaf_radius < 0:
            raise ValueError("Fingerprint radii must be non-negative")
        if self.linear_max_path_length < 1:
            raise ValueError("linear_max_path_length must be at least 1 edge")
        if not self.marker_symbol:
            raise ValueError("marker_symbol must not be empty")
        if not self.neighbor_k:
            raise ValueError("neighbor_k must contain at least one value")
        invalid = [k for k in self.neighbor_k if k <= 0 or k >= n_molecules]
        if invalid:
            raise ValueError(
                "Each neighbor k must satisfy 0 < k < number of molecules; "
                f"invalid values: {invalid}"
            )


@dataclass(frozen=True)
class SimilarityRetentionDataset:
    """Validated input records and their provenance."""

    frame: pd.DataFrame
    product_columns: Tuple[str, ...]
    source_path: Path
    source_sha256: str

    @property
    def n_molecules(self) -> int:
        return len(self.frame)


@dataclass(frozen=True)
class BenchmarkResults:
    """Tidy result tables returned by the benchmark runner."""

    per_partner_metrics: pd.DataFrame
    descriptor_summary: pd.DataFrame
    paired_descriptor_differences: pd.DataFrame
    molecule_manifest: pd.DataFrame
    run_metadata: Mapping[str, object]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _infer_product_columns(columns: Iterable[str]) -> Tuple[str, ...]:
    matches = []
    for column in columns:
        match = re.fullmatch(r"SMILES_(\d+)", str(column))
        if match:
            matches.append((int(match.group(1)), str(column)))
    matches.sort()
    if not matches:
        raise ValueError("No product columns named SMILES_<number> were found")
    numbers = [number for number, _ in matches]
    expected = list(range(len(numbers)))
    if numbers != expected:
        raise ValueError(
            "Product columns must be consecutively numbered from SMILES_0; "
            f"found suffixes {numbers}"
        )
    return tuple(column for _, column in matches)


def _validate_marker_smiles(
    smiles: str,
    *,
    row_number: int,
    column: str,
    marker_symbol: str,
) -> None:
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError(f"Missing SMILES at CSV row {row_number}, column {column}")
    marker_token = f"[{marker_symbol}]"
    markers = smiles.count(marker_token)
    if markers != 1:
        raise ValueError(
            f"Expected exactly one {marker_symbol} marker at CSV row {row_number}, "
            f"column {column}; found {markers}"
        )
    if "." in smiles:
        raise ValueError(
            f"Disconnected structure at CSV row {row_number}, column {column}"
        )
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(
            f"RDKit could not parse CSV row {row_number}, column {column}: {smiles!r}"
        )


def load_similarity_retention_dataset(
    csv_path: str | Path,
    *,
    product_columns: Sequence[str] | None = None,
    marker_symbol: str = "Hg",
) -> SimilarityRetentionDataset:
    """Load and validate a wide similarity-retention dataset.

    The raw file is never modified.  The returned dataframe adds ``bb_id`` and
    ``source_row`` columns.  ``bb_id`` is a permanent analysis label for the
    frozen input manifest; ``source_row`` is the one-based CSV line number,
    including the header as line 1.
    """
    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "SMILES_BB" not in frame.columns:
        raise ValueError("Missing required column 'SMILES_BB'")

    selected_products = (
        tuple(product_columns)
        if product_columns is not None
        else _infer_product_columns(frame.columns)
    )
    if not selected_products:
        raise ValueError("At least one product column is required")
    missing_columns = [c for c in selected_products if c not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing product columns: {missing_columns}")

    expected_columns = {"SMILES_BB", *selected_products}
    unexpected = [c for c in frame.columns if c not in expected_columns]
    if unexpected:
        raise ValueError(f"Unexpected columns in input CSV: {unexpected}")
    if frame.empty:
        raise ValueError("Input CSV contains no molecule rows")

    structure_columns = ("SMILES_BB", *selected_products)
    with _suppress_rdkit_messages():
        for row_index, row in frame.iterrows():
            csv_row = int(row_index) + 2
            for column in structure_columns:
                _validate_marker_smiles(
                    row[column],
                    row_number=csv_row,
                    column=column,
                    marker_symbol=marker_symbol,
                )

    duplicated = frame["SMILES_BB"].duplicated(keep=False)
    if duplicated.any():
        rows = (frame.index[duplicated] + 2).tolist()
        raise ValueError(f"Duplicate building-block SMILES at CSV rows {rows}")

    output = frame.copy()
    width = max(4, len(str(len(output))))
    output.insert(0, "source_row", np.arange(2, len(output) + 2, dtype=int))
    output.insert(
        0,
        "bb_id",
        [f"BB_{index:0{width}d}" for index in range(1, len(output) + 1)],
    )
    return SimilarityRetentionDataset(
        frame=output,
        product_columns=selected_products,
        source_path=path,
        source_sha256=_sha256_file(path),
    )


def build_fingerprint_functions(
    config: BenchmarkConfig,
) -> Dict[str, Callable[[str], DataStructs.ExplicitBitVect]]:
    """Build the four agreed descriptor callables for a benchmark config."""
    def ecfp_o(smiles: str) -> DataStructs.ExplicitBitVect:
        return ecfp_fingerprint(
            smiles,
            radius=config.ecfp_radius,
            n_bits=config.n_bits,
            marker_replacement="O",
            marker_symbol=config.marker_symbol,
        )

    def ecfp_hg(smiles: str) -> DataStructs.ExplicitBitVect:
        return ecfp_fingerprint(
            smiles,
            radius=config.ecfp_radius,
            n_bits=config.n_bits,
            marker_replacement=None,
            marker_symbol=config.marker_symbol,
        )

    def coaf(smiles: str) -> DataStructs.ExplicitBitVect:
        return fingerprint_to_bitvect(
            coaf_fingerprint_from_smiles(
                smiles,
                radius=config.coaf_radius,
                n_bits=config.n_bits,
                marker_symbol=config.marker_symbol,
            )
        )

    def directed_linear(smiles: str) -> DataStructs.ExplicitBitVect:
        return fingerprint_to_bitvect(
            directed_linear_fingerprint_from_smiles(
                smiles,
                max_path_length=config.linear_max_path_length,
                n_bits=config.n_bits,
                marker_symbol=config.marker_symbol,
                binary=True,
            )
        )

    return {
        "ECFP_O": ecfp_o,
        "ECFP_HG": ecfp_hg,
        "COAF": coaf,
        "DIRECTED_LINEAR": directed_linear,
    }


def compute_similarity_matrix(
    fingerprints: Sequence[DataStructs.ExplicitBitVect],
) -> np.ndarray:
    """Compute a symmetric binary-Tanimoto similarity matrix."""
    n_fingerprints = len(fingerprints)
    if n_fingerprints == 0:
        raise ValueError("At least one fingerprint is required")
    n_bits = fingerprints[0].GetNumBits()
    if any(fp.GetNumBits() != n_bits for fp in fingerprints):
        raise ValueError("All fingerprints must have the same length")

    matrix = np.eye(n_fingerprints, dtype=np.float64)
    for index in range(1, n_fingerprints):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[index], fingerprints[:index]
        )
        matrix[index, :index] = similarities
        matrix[:index, index] = similarities
    return matrix


def _upper_triangle(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Similarity matrix must be square")
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def _top_k_indices(matrix: np.ndarray, k: int) -> Sequence[set[int]]:
    """Return deterministic top-k sets, breaking similarity ties by row index."""
    indices = np.arange(matrix.shape[0])
    neighborhoods = []
    for row_index in range(matrix.shape[0]):
        values = matrix[row_index].copy()
        values[row_index] = -np.inf
        order = np.lexsort((indices, -values))
        neighborhoods.append(set(int(value) for value in order[:k]))
    return neighborhoods


def compare_similarity_matrices(
    building_block_matrix: np.ndarray,
    product_matrix: np.ndarray,
    *,
    neighbor_k: Sequence[int] = (5, 10),
) -> Dict[str, float]:
    """Compare aligned building-block and product similarity matrices."""
    if building_block_matrix.shape != product_matrix.shape:
        raise ValueError("Similarity matrices must have identical shapes")
    bb_values = _upper_triangle(building_block_matrix)
    product_values = _upper_triangle(product_matrix)
    metrics: Dict[str, float] = {
        "spearman": float(spearmanr(bb_values, product_values).statistic),
        "pearson": float(pearsonr(bb_values, product_values).statistic),
    }
    for k in neighbor_k:
        bb_neighbors = _top_k_indices(building_block_matrix, int(k))
        product_neighbors = _top_k_indices(product_matrix, int(k))
        metrics[f"top_{k}_neighbor_preservation"] = float(
            np.mean(
                [
                    len(bb_set & product_set) / k
                    for bb_set, product_set in zip(bb_neighbors, product_neighbors)
                ]
            )
        )
    return metrics


def _descriptor_summary(per_partner: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in per_partner.columns
        if column not in {"descriptor", "product_column", "n_molecules", "n_pairs"}
    ]
    rows = []
    for descriptor, group in per_partner.groupby("descriptor", sort=False):
        row: Dict[str, object] = {
            "descriptor": descriptor,
            "n_partners": len(group),
            "n_molecules": int(group["n_molecules"].iloc[0]),
        }
        for metric in metric_columns:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_differences(
    per_partner: pd.DataFrame,
    *,
    reference: str = "ECFP_O",
) -> pd.DataFrame:
    metric_columns = [
        column
        for column in per_partner.columns
        if column not in {"descriptor", "product_column", "n_molecules", "n_pairs"}
    ]
    rows = []
    reference_rows = per_partner[per_partner["descriptor"] == reference].set_index(
        "product_column"
    )
    for descriptor in DESCRIPTOR_NAMES:
        if descriptor == reference:
            continue
        comparison_rows = per_partner[
            per_partner["descriptor"] == descriptor
        ].set_index("product_column")
        for product_column in reference_rows.index:
            row: Dict[str, object] = {
                "descriptor": descriptor,
                "reference_descriptor": reference,
                "product_column": product_column,
            }
            for metric in metric_columns:
                row[f"delta_{metric}"] = float(
                    comparison_rows.loc[product_column, metric]
                    - reference_rows.loc[product_column, metric]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _run_metadata(
    dataset: SimilarityRetentionDataset,
    config: BenchmarkConfig,
) -> Dict[str, object]:
    return {
        "analysis": "building-block-to-product similarity retention",
        "input_file": str(dataset.source_path),
        "input_sha256": dataset.source_sha256,
        "n_molecules": dataset.n_molecules,
        "product_columns": list(dataset.product_columns),
        "descriptors": list(DESCRIPTOR_NAMES),
        "parameters": asdict(config),
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }


def _save_results(
    results: BenchmarkResults,
    output_dir: Path,
    matrices: Mapping[Tuple[str, str], np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results.molecule_manifest.to_csv(
        output_dir / "molecule_manifest.csv", index=False
    )
    results.per_partner_metrics.to_csv(
        output_dir / "per_partner_metrics.csv", index=False
    )
    results.descriptor_summary.to_csv(
        output_dir / "descriptor_summary.csv", index=False
    )
    results.paired_descriptor_differences.to_csv(
        output_dir / "paired_descriptor_differences.csv", index=False
    )
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(results.run_metadata, handle, indent=2, sort_keys=True)

    matrix_dir = output_dir / "similarity_matrices"
    matrix_dir.mkdir(exist_ok=True)
    for (descriptor, structure_set), matrix in matrices.items():
        np.save(matrix_dir / f"{descriptor}__{structure_set}.npy", matrix)


def run_similarity_retention_benchmark(
    dataset: SimilarityRetentionDataset,
    *,
    config: BenchmarkConfig | None = None,
    output_dir: str | Path | None = None,
) -> BenchmarkResults:
    """Run all four descriptors on every partner column.

    When ``output_dir`` is supplied, tidy CSV results, metadata, the molecule
    manifest, and the similarity matrices are saved.  The returned tables are
    identical whether or not outputs are written.
    """
    active_config = config or BenchmarkConfig()
    active_config.validate(dataset.n_molecules)
    fingerprint_functions = build_fingerprint_functions(active_config)
    structure_columns = ("SMILES_BB", *dataset.product_columns)

    matrices: Dict[Tuple[str, str], np.ndarray] = {}
    for descriptor, fingerprint_function in fingerprint_functions.items():
        for column in structure_columns:
            with _suppress_rdkit_messages():
                fingerprints = [
                    fingerprint_function(smiles)
                    for smiles in dataset.frame[column].tolist()
                ]
            matrices[(descriptor, column)] = compute_similarity_matrix(fingerprints)

    rows = []
    n_pairs = dataset.n_molecules * (dataset.n_molecules - 1) // 2
    for descriptor in DESCRIPTOR_NAMES:
        bb_matrix = matrices[(descriptor, "SMILES_BB")]
        for product_column in dataset.product_columns:
            metrics = compare_similarity_matrices(
                bb_matrix,
                matrices[(descriptor, product_column)],
                neighbor_k=active_config.neighbor_k,
            )
            rows.append(
                {
                    "descriptor": descriptor,
                    "product_column": product_column,
                    "n_molecules": dataset.n_molecules,
                    "n_pairs": n_pairs,
                    **metrics,
                }
            )

    per_partner = pd.DataFrame(rows)
    manifest_columns = ["bb_id", "source_row", "SMILES_BB"]
    results = BenchmarkResults(
        per_partner_metrics=per_partner,
        descriptor_summary=_descriptor_summary(per_partner),
        paired_descriptor_differences=_paired_differences(per_partner),
        molecule_manifest=dataset.frame[manifest_columns].copy(),
        run_metadata=_run_metadata(dataset, active_config),
    )
    if output_dir is not None:
        _save_results(results, Path(output_dir).expanduser().resolve(), matrices)
    return results

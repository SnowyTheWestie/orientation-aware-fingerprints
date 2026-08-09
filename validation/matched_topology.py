"""Matched combinatorial-library topology benchmark for four descriptors.

The input contains the same building-block combinations assembled in two
topologies (L1 and L2).  ECFP is calculated from ordinary SMILES, whereas
COAF, POAF, and ECFP-Hg are calculated from the corresponding [Hg]-rooted
SMILES. Standard ECFP is calculated from the ordinary structures.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, DataStructs, RDLogger
from sklearn.mixture import GaussianMixture

from coaf import (
    coaf_fingerprint_from_smiles,
    directed_linear_fingerprint_from_smiles,
    ecfp_fingerprint,
    fingerprint_to_bitvect,
)


REQUIRED_COLUMNS = ("ID", "ECFP_L1", "ECFP_L2", "COAF_L1", "COAF_L2")
DESCRIPTORS = ("ECFP", "ECFP_Hg", "COAF", "POAF")


@dataclass(frozen=True)
class MatchedTopologyConfig:
    n_bits: int = 1024
    ecfp_radius: int = 3
    coaf_radius: int = 3
    poaf_max_path_length: int = 6
    n_pair_samples: int = 100_000
    neighbor_subset_ids: int = 2_000
    neighbor_k: tuple[int, ...] = (1, 5, 10, 20)
    ranking_size: int = 20
    random_seed: int = 123
    population_component_counts: tuple[int, ...] = (1, 2, 3, 4)
    population_primary_components: int = 2


@dataclass
class MatchedTopologyResults:
    dataset_summary: pd.DataFrame
    fingerprint_summary: pd.DataFrame
    matched_pair_distances: pd.DataFrame
    matched_summary: pd.DataFrame
    controlled_pair_distances: pd.DataFrame
    topology_separation_summary: pd.DataFrame
    descriptor_comparison: pd.DataFrame
    neighbor_query_results: pd.DataFrame
    neighbor_summary: pd.DataFrame
    ranked_coaf_gain: pd.DataFrame
    ranked_poaf_gain: pd.DataFrame
    ranked_ecfp_hg_gain: pd.DataFrame
    ranked_ecfp_favored: pd.DataFrame
    population_model_comparison: pd.DataFrame
    population_component_summary: pd.DataFrame
    population_assignments: pd.DataFrame
    output_dir: Path | None = None


@contextmanager
def _suppress_rdkit_messages():
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    try:
        yield
    finally:
        RDLogger.EnableLog("rdApp.warning")
        RDLogger.EnableLog("rdApp.error")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_matched_topology_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the matched-library input table."""
    path = Path(path)
    data = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    data = data.loc[:, REQUIRED_COLUMNS].copy()
    if data.empty:
        raise ValueError("The matched-library dataset is empty.")
    if data.isna().any().any():
        columns = data.columns[data.isna().any()].tolist()
        raise ValueError(f"Missing values found in columns: {columns}")
    data["ID"] = data["ID"].astype(str)
    if data["ID"].duplicated().any():
        examples = data.loc[data["ID"].duplicated(keep=False), "ID"].head().tolist()
        raise ValueError(f"ID values must be unique; duplicate examples: {examples}")

    failures: list[str] = []
    with _suppress_rdkit_messages():
        for column in REQUIRED_COLUMNS[1:]:
            rooted = column.startswith("COAF_")
            for row_index, smiles in data[column].items():
                smiles = str(smiles)
                if rooted and smiles.count("[Hg]") != 1:
                    failures.append(f"row {row_index}, {column}: expected exactly one [Hg]")
                    continue
                if Chem.MolFromSmiles(smiles) is None:
                    failures.append(f"row {row_index}, {column}: invalid SMILES")
    if failures:
        preview = "; ".join(failures[:10])
        suffix = " ..." if len(failures) > 10 else ""
        raise ValueError(f"SMILES validation failed: {preview}{suffix}")
    return data


def _fingerprint_table(data: pd.DataFrame, config: MatchedTopologyConfig):
    fingerprints: dict[tuple[str, str], list] = {}
    density_rows = []
    with _suppress_rdkit_messages():
        for descriptor, topology, column in (
            ("ECFP", "L1", "ECFP_L1"),
            ("ECFP", "L2", "ECFP_L2"),
            ("ECFP_Hg", "L1", "COAF_L1"),
            ("ECFP_Hg", "L2", "COAF_L2"),
            ("COAF", "L1", "COAF_L1"),
            ("COAF", "L2", "COAF_L2"),
            ("POAF", "L1", "COAF_L1"),
            ("POAF", "L2", "COAF_L2"),
        ):
            fps = []
            for smiles in data[column]:
                if descriptor in ("ECFP", "ECFP_Hg"):
                    fp = ecfp_fingerprint(smiles, radius=config.ecfp_radius, n_bits=config.n_bits)
                elif descriptor == "COAF":
                    fp = fingerprint_to_bitvect(
                        coaf_fingerprint_from_smiles(
                            smiles, radius=config.coaf_radius, n_bits=config.n_bits
                        )
                    )
                else:
                    fp = fingerprint_to_bitvect(
                        directed_linear_fingerprint_from_smiles(
                            smiles,
                            max_path_length=config.poaf_max_path_length,
                            n_bits=config.n_bits,
                        )
                    )
                fps.append(fp)
            fingerprints[(descriptor, topology)] = fps
            on_bits = np.asarray([fp.GetNumOnBits() for fp in fps], dtype=float)
            density_rows.append(
                {
                    "descriptor": descriptor,
                    "topology": topology,
                    "n_structures": len(fps),
                    "mean_on_bits": on_bits.mean(),
                    "median_on_bits": np.median(on_bits),
                    "mean_bit_density": on_bits.mean() / config.n_bits,
                }
            )
    return fingerprints, pd.DataFrame(density_rows)


def _distance(fp_a, fp_b) -> float:
    return 1.0 - float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def _effect_size(values: pd.Series) -> float:
    std = float(values.std(ddof=1))
    return float(values.mean() / std) if std > 0 else np.nan


def _sample_unique_pairs(n_items: int, n_samples: int, seed: int) -> list[tuple[int, int]]:
    total = n_items * (n_items - 1) // 2
    target = min(n_samples, total)
    if target == total:
        return [(i, j) for i in range(n_items) for j in range(i + 1, n_items)]
    rng = random.Random(seed)
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < target:
        i, j = rng.sample(range(n_items), 2)
        pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def _matched_analysis(data, fingerprints):
    rows = []
    for index, identifier in enumerate(data["ID"]):
        row = {"ID": identifier}
        for descriptor in DESCRIPTORS:
            row[f"{descriptor}_distance_L1_L2"] = _distance(
                fingerprints[(descriptor, "L1")][index],
                fingerprints[(descriptor, "L2")][index],
            )
        for descriptor in DESCRIPTORS[1:]:
            row[f"{descriptor}_minus_ECFP_distance"] = (
                row[f"{descriptor}_distance_L1_L2"] - row["ECFP_distance_L1_L2"]
            )
        rows.append(row)
    distances = pd.DataFrame(rows)
    summary_rows = []
    for descriptor in DESCRIPTORS:
        values = distances[f"{descriptor}_distance_L1_L2"]
        summary_rows.append(
            {
                "descriptor": descriptor,
                "n_pairs": len(values),
                "mean_distance": values.mean(),
                "median_distance": values.median(),
                "std_distance": values.std(ddof=1),
                "q25_distance": values.quantile(0.25),
                "q75_distance": values.quantile(0.75),
            }
        )
    for descriptor in DESCRIPTORS[1:]:
        delta = distances[f"{descriptor}_minus_ECFP_distance"]
        summary_rows.append(
            {
                "descriptor": f"{descriptor}_minus_ECFP",
                "n_pairs": len(delta),
                "mean_distance": delta.mean(),
                "median_distance": delta.median(),
                "std_distance": delta.std(ddof=1),
                "q25_distance": delta.quantile(0.25),
                "q75_distance": delta.quantile(0.75),
            }
        )
    return distances, pd.DataFrame(summary_rows)


def analyze_descriptor_difference_populations(
    matched_pair_distances: pd.DataFrame,
    component_counts: Sequence[int] = (1, 2, 3, 4),
    primary_components: int = 2,
    random_seed: int = 123,
):
    """Fit Gaussian mixtures to the matched COAF-minus-ECFP distances.

    Returns model-selection statistics, a summary of the selected descriptive
    mixture, and the most likely component assignment for every matched ID.
    The components are ordered by increasing fitted mean difference.
    """
    required = {"ID", "COAF_minus_ECFP_distance"}
    if not required.issubset(matched_pair_distances.columns):
        raise ValueError(f"matched_pair_distances must contain {sorted(required)}")
    counts = tuple(sorted(set(int(value) for value in component_counts)))
    if not counts or min(counts) < 1:
        raise ValueError("component_counts must contain positive integers")
    if primary_components not in counts:
        raise ValueError("primary_components must be included in component_counts")

    values = matched_pair_distances["COAF_minus_ECFP_distance"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Descriptor differences must all be finite")
    x = values.reshape(-1, 1)
    models = {}
    comparison_rows = []
    for n_components in counts:
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            n_init=20,
            init_params="random_from_data",
            random_state=random_seed,
        ).fit(x)
        models[n_components] = model
        comparison_rows.append(
            {
                "n_components": n_components,
                "aic": model.aic(x),
                "bic": model.bic(x),
            }
        )
    model_comparison = pd.DataFrame(comparison_rows)
    model_comparison["delta_aic"] = model_comparison["aic"] - model_comparison["aic"].min()
    model_comparison["delta_bic"] = model_comparison["bic"] - model_comparison["bic"].min()

    model = models[primary_components]
    order = np.argsort(model.means_.ravel())
    original_to_ordered = {original: ordered for ordered, original in enumerate(order)}
    raw_labels = model.predict(x)
    labels = np.asarray([original_to_ordered[label] for label in raw_labels], dtype=int)
    ordered_probabilities = model.predict_proba(x)[:, order]
    assigned_probability = ordered_probabilities[np.arange(len(x)), labels]
    assignments = pd.DataFrame(
        {
            "ID": matched_pair_distances["ID"].astype(str).to_numpy(),
            "COAF_minus_ECFP_distance": values,
            "population_component": labels + 1,
            "assignment_probability": assigned_probability,
        }
    )

    summary_rows = []
    for ordered_label, original_label in enumerate(order):
        assigned = values[labels == ordered_label]
        summary_rows.append(
            {
                "population_component": ordered_label + 1,
                "model_weight": model.weights_[original_label],
                "fitted_mean": model.means_[original_label, 0],
                "fitted_std": np.sqrt(model.covariances_[original_label, 0, 0]),
                "n_assigned": len(assigned),
                "fraction_assigned": len(assigned) / len(values),
                "assigned_mean": assigned.mean(),
                "assigned_median": np.median(assigned),
                "assigned_std": assigned.std(ddof=1) if len(assigned) > 1 else np.nan,
                "assigned_min": assigned.min(),
                "assigned_max": assigned.max(),
            }
        )
    return model_comparison, pd.DataFrame(summary_rows), assignments


def _controlled_analysis(data, fingerprints, pairs):
    rows = []
    for pair_number, (i, j) in enumerate(pairs):
        base = {"pair_number": pair_number, "ID_i": data.iloc[i]["ID"], "ID_j": data.iloc[j]["ID"]}
        for descriptor in DESCRIPTORS:
            l1, l2 = fingerprints[(descriptor, "L1")], fingerprints[(descriptor, "L2")]
            within_l1 = _distance(l1[i], l1[j])
            within_l2 = _distance(l2[i], l2[j])
            cross_l1_l2 = _distance(l1[i], l2[j])
            cross_l2_l1 = _distance(l2[i], l1[j])
            within_mean = (within_l1 + within_l2) / 2.0
            cross_mean = (cross_l1_l2 + cross_l2_l1) / 2.0
            rows.append(
                {
                    **base,
                    "descriptor": descriptor,
                    "within_L1_distance": within_l1,
                    "within_L2_distance": within_l2,
                    "cross_L1_L2_distance": cross_l1_l2,
                    "cross_L2_L1_distance": cross_l2_l1,
                    "within_mean_distance": within_mean,
                    "cross_mean_distance": cross_mean,
                    "topology_separation": cross_mean - within_mean,
                }
            )
    distances = pd.DataFrame(rows)
    summaries = []
    for descriptor, group in distances.groupby("descriptor", sort=False):
        values = group["topology_separation"]
        summaries.append(
            {
                "descriptor": descriptor,
                "n_id_pairs": len(group),
                "mean_within_distance": group["within_mean_distance"].mean(),
                "mean_cross_distance": group["cross_mean_distance"].mean(),
                "mean_topology_separation": values.mean(),
                "median_topology_separation": values.median(),
                "std_topology_separation": values.std(ddof=1),
                "fraction_positive": (values > 0).mean(),
                "paired_effect_size": _effect_size(values),
            }
        )
    wide = distances.pivot(index="pair_number", columns="descriptor", values="topology_separation")
    comparison_rows = []
    for descriptor in DESCRIPTORS[1:]:
        difference = wide[descriptor] - wide["ECFP"]
        comparison_rows.append(
            {
                "descriptor": descriptor,
                "reference_descriptor": "ECFP",
                "mean_separation_difference": difference.mean(),
                "median_separation_difference": difference.median(),
                "fraction_greater_separation": (difference > 0).mean(),
                "paired_effect_size_difference": _effect_size(difference),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    return distances, pd.DataFrame(summaries), comparison


def _neighbor_analysis(data, fingerprints, config):
    n_ids = min(config.neighbor_subset_ids, len(data))
    rng = np.random.default_rng(config.random_seed)
    indices = np.sort(rng.choice(len(data), size=n_ids, replace=False))
    query_rows = []
    for descriptor in DESCRIPTORS:
        fps = [fingerprints[(descriptor, topology)][i] for topology in ("L1", "L2") for i in indices]
        topologies = np.asarray([topology for topology in ("L1", "L2") for _ in indices])
        identifiers = np.asarray([data.iloc[i]["ID"] for _ in ("L1", "L2") for i in indices])
        for query_index, query_fp in enumerate(fps):
            similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(query_fp, fps))
            similarities[query_index] = -1.0
            order = np.lexsort((np.arange(len(fps)), -similarities))
            for k in config.neighbor_k:
                effective_k = min(k, len(fps) - 1)
                neighbors = order[:effective_k]
                query_rows.append(
                    {
                        "descriptor": descriptor,
                        "query_ID": identifiers[query_index],
                        "query_topology": topologies[query_index],
                        "k": k,
                        "effective_k": effective_k,
                        "same_topology_fraction": float(np.mean(topologies[neighbors] == topologies[query_index])),
                    }
                )
    results = pd.DataFrame(query_rows)
    summary = (
        results.groupby(["descriptor", "k"], as_index=False)
        .agg(
            n_queries=("query_ID", "size"),
            mean_same_topology_fraction=("same_topology_fraction", "mean"),
            median_same_topology_fraction=("same_topology_fraction", "median"),
            std_same_topology_fraction=("same_topology_fraction", "std"),
        )
    )
    return results, summary


def _dataset_summary(data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": ["n_matched_ids", "n_topologies", "n_descriptor_structure_columns"],
            "value": [len(data), 2, 4],
        }
    )


def _write_results(results: MatchedTopologyResults, output_dir: Path, metadata: dict, overwrite: bool):
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already contains files: {output_dir}. "
            "Choose a new directory or set overwrite=True explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "dataset_summary.csv": results.dataset_summary,
        "fingerprint_summary.csv": results.fingerprint_summary,
        "matched_pair_distances.csv": results.matched_pair_distances,
        "matched_summary.csv": results.matched_summary,
        "controlled_pair_distances.csv": results.controlled_pair_distances,
        "topology_separation_summary.csv": results.topology_separation_summary,
        "descriptor_comparison.csv": results.descriptor_comparison,
        "neighbor_query_results.csv": results.neighbor_query_results,
        "neighbor_summary.csv": results.neighbor_summary,
        "ranked_coaf_gain.csv": results.ranked_coaf_gain,
        "ranked_poaf_gain.csv": results.ranked_poaf_gain,
        "ranked_ecfp_hg_gain.csv": results.ranked_ecfp_hg_gain,
        "ranked_ecfp_favored.csv": results.ranked_ecfp_favored,
        "population_model_comparison.csv": results.population_model_comparison,
        "population_component_summary.csv": results.population_component_summary,
        "population_assignments.csv": results.population_assignments,
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False)
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_matched_topology_benchmark(
    dataset: pd.DataFrame | str | Path,
    config: MatchedTopologyConfig | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> MatchedTopologyResults:
    """Run matched, controlled-pair, and neighborhood topology analyses."""
    config = config or MatchedTopologyConfig()
    input_path = Path(dataset) if isinstance(dataset, (str, Path)) else None
    data = load_matched_topology_dataset(input_path) if input_path is not None else dataset.copy()
    if not set(REQUIRED_COLUMNS).issubset(data.columns):
        raise ValueError(f"Dataset must contain columns: {list(REQUIRED_COLUMNS)}")

    fingerprints, fingerprint_summary = _fingerprint_table(data, config)
    matched, matched_summary = _matched_analysis(data, fingerprints)
    sampled_pairs = _sample_unique_pairs(len(data), config.n_pair_samples, config.random_seed)
    controlled, separation_summary, comparison = _controlled_analysis(data, fingerprints, sampled_pairs)
    neighbor_results, neighbor_summary = _neighbor_analysis(data, fingerprints, config)
    ranked_gain = matched.nlargest(config.ranking_size, "COAF_minus_ECFP_distance")
    ranked_poaf = matched.nlargest(config.ranking_size, "POAF_minus_ECFP_distance")
    ranked_ecfp_hg = matched.nlargest(
        config.ranking_size, "ECFP_Hg_minus_ECFP_distance"
    )
    ranked_ecfp = matched.nsmallest(config.ranking_size, "COAF_minus_ECFP_distance")
    population_models, population_summary, population_assignments = (
        analyze_descriptor_difference_populations(
            matched,
            component_counts=config.population_component_counts,
            primary_components=config.population_primary_components,
            random_seed=config.random_seed,
        )
    )

    results = MatchedTopologyResults(
        dataset_summary=_dataset_summary(data),
        fingerprint_summary=fingerprint_summary,
        matched_pair_distances=matched,
        matched_summary=matched_summary,
        controlled_pair_distances=controlled,
        topology_separation_summary=separation_summary,
        descriptor_comparison=comparison,
        neighbor_query_results=neighbor_results,
        neighbor_summary=neighbor_summary,
        ranked_coaf_gain=ranked_gain,
        ranked_poaf_gain=ranked_poaf,
        ranked_ecfp_hg_gain=ranked_ecfp_hg,
        ranked_ecfp_favored=ranked_ecfp,
        population_model_comparison=population_models,
        population_component_summary=population_summary,
        population_assignments=population_assignments,
        output_dir=Path(output_dir) if output_dir is not None else None,
    )
    if output_dir is not None:
        metadata = {
            "config": asdict(config),
            "python_version": platform.python_version(),
            "rdkit_version": rdkit.__version__,
            "input_path": str(input_path.resolve()) if input_path else None,
            "input_sha256": _file_sha256(input_path) if input_path else None,
            "descriptor_definition": {
                "ECFP": "Morgan bit vector from unrooted ECFP_L1/ECFP_L2 SMILES",
                "ECFP_Hg": "Morgan bit vector from [Hg]-rooted COAF_L1/COAF_L2 SMILES",
                "COAF": "coaf.py coaf_fingerprint_from_smiles from [Hg]-rooted COAF_L1/COAF_L2 SMILES",
                "POAF": "coaf.py directed_linear_fingerprint_from_smiles from [Hg]-rooted COAF_L1/COAF_L2 SMILES",
            },
            "sampled_unique_id_pairs": len(sampled_pairs),
        }
        _write_results(results, Path(output_dir), metadata, overwrite)
    return results

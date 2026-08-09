"""Reproducible validation workflows for the COAF project."""

from .similarity_retention import (
    BenchmarkConfig,
    BenchmarkResults,
    SimilarityRetentionDataset,
    build_fingerprint_functions,
    load_similarity_retention_dataset,
    run_similarity_retention_benchmark,
)
from .two_component_similarity import (
    TwoComponentConfig,
    TwoComponentDataset,
    TwoComponentResults,
    load_two_component_dataset,
    run_two_component_benchmark,
)
from .attachment_site_discrimination import (
    AttachmentSiteConfig,
    AttachmentSiteDataset,
    AttachmentSiteResults,
    load_attachment_site_dataset,
    run_attachment_site_analysis,
)
from .matched_topology import (
    MatchedTopologyConfig,
    MatchedTopologyResults,
    analyze_descriptor_difference_populations,
    load_matched_topology_dataset,
    run_matched_topology_benchmark,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResults",
    "SimilarityRetentionDataset",
    "build_fingerprint_functions",
    "load_similarity_retention_dataset",
    "run_similarity_retention_benchmark",
    "TwoComponentConfig",
    "TwoComponentDataset",
    "TwoComponentResults",
    "load_two_component_dataset",
    "run_two_component_benchmark",
    "AttachmentSiteConfig",
    "AttachmentSiteDataset",
    "AttachmentSiteResults",
    "load_attachment_site_dataset",
    "run_attachment_site_analysis",
]

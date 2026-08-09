"""Orientation-aware molecular fingerprints for attachment-point chemistry.

This module implements the Circular Orientation-Aware Fingerprint (COAF), a
rooted Weisfeiler-Lehman fingerprint, and a complementary directed linear-path
fingerprint.  Molecules are supplied as SMILES containing exactly one explicit
attachment marker (``[Hg]`` by default).  The marker is retained as an
artificial root, and bonds are directed according to shortest-path distance
from that root.

The implementation is deterministic across Python processes: explicit feature
representations are folded with SHA-256 rather than Python's randomized
``hash`` function.
"""

from __future__ import annotations

import hashlib
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Counter as CounterType, Dict, Hashable, Iterable, List, Mapping, Optional, Tuple

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.rdchem import Bond, Mol

__all__ = [
    "COAFError",
    "NodeData",
    "EdgeData",
    "RootedGraph",
    "build_rooted_graph",
    "validate_rooted_graph",
    "initial_atom_labels",
    "coaf_features",
    "coaf_fingerprint",
    "coaf_fingerprint_from_smiles",
    "directed_linear_features",
    "directed_linear_fingerprint",
    "directed_linear_fingerprint_from_smiles",
    "ecfp_fingerprint",
    "tanimoto_similarity",
    "fingerprint_to_bitvect",
    "fingerprint_to_bitstring",
]


class COAFError(ValueError):
    """Raised when an input cannot be converted into a valid rooted graph."""


@dataclass(frozen=True)
class NodeData:
    """Attributes retained for one molecular-graph node."""

    atom_index: int
    element: str
    is_aromatic: bool
    formal_charge: int
    hybridization: Optional[str]
    morgan_invariant: int


@dataclass(frozen=True)
class EdgeData:
    """A directed molecular-graph edge and its bond attributes."""

    source: int
    target: int
    bond_type: str
    is_aromatic: bool
    is_in_ring: bool


@dataclass(frozen=True)
class RootedGraph:
    """A molecular graph directed by shortest-path depth from a root node."""

    root_id: int
    nodes: Dict[int, NodeData]
    edges: List[EdgeData]
    depths: Dict[int, int]


def _parse_smiles(smiles: str) -> Mol:
    if not isinstance(smiles, str) or not smiles.strip():
        raise COAFError("smiles must be a non-empty string")
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        raise COAFError(f"Failed to parse SMILES: {smiles!r}")
    return mol


def _attachment_marker_index(mol: Mol, marker_symbol: str) -> int:
    indices = [
        atom.GetIdx() for atom in mol.GetAtoms()
        if atom.GetSymbol() == marker_symbol
    ]
    if not indices:
        raise COAFError(f"No attachment marker atom {marker_symbol!r} found")
    if len(indices) != 1:
        raise COAFError(
            f"Expected exactly one attachment marker atom {marker_symbol!r}; "
            f"found {len(indices)}"
        )
    return indices[0]


def _depths_from_root(mol: Mol, root_id: int) -> Dict[int, int]:
    depths = {root_id: 0}
    queue = deque([root_id])
    while queue:
        current = queue.popleft()
        neighbors = sorted(
            atom.GetIdx()
            for atom in mol.GetAtomWithIdx(current).GetNeighbors()
        )
        for neighbor in neighbors:
            if neighbor not in depths:
                depths[neighbor] = depths[current] + 1
                queue.append(neighbor)
    if len(depths) != mol.GetNumAtoms():
        raise COAFError("The molecule contains atoms unreachable from the root")
    return depths


def _orient_bond(
    first: int,
    second: int,
    depths: Mapping[int, int],
) -> List[Tuple[int, int]]:
    """Orient a bond outward; represent equal-depth bonds in both directions."""
    first_depth = depths[first]
    second_depth = depths[second]
    if first_depth < second_depth:
        return [(first, second)]
    if second_depth < first_depth:
        return [(second, first)]
    return [(first, second), (second, first)]


def _node_data(
    atom: Chem.Atom,
    root_id: int,
    morgan_invariant: int,
) -> NodeData:
    atom_index = atom.GetIdx()
    if atom_index == root_id:
        return NodeData(atom_index, "ROOT", False, 0, None, morgan_invariant)
    return NodeData(
        atom_index=atom_index,
        element=atom.GetSymbol(),
        is_aromatic=atom.GetIsAromatic(),
        formal_charge=atom.GetFormalCharge(),
        hybridization=str(atom.GetHybridization()),
        morgan_invariant=morgan_invariant,
    )


def _bond_attributes(bond: Bond) -> Tuple[str, bool, bool]:
    return str(bond.GetBondType()), bond.GetIsAromatic(), bond.IsInRing()


def build_rooted_graph(
    smiles: str,
    marker_symbol: str = "Hg",
) -> RootedGraph:
    """Build a deterministic rooted, directed molecular graph.

    Exactly one marker atom is required.  Direction follows increasing
    shortest-path distance from the marker.  Bonds joining nodes at equal
    depth (which can occur in rings) are stored in both directions.
    """
    mol = _parse_smiles(smiles)
    if len(Chem.GetMolFrags(mol)) != 1:
        raise COAFError("The molecule must contain exactly one connected component")
    root_id = _attachment_marker_index(mol, marker_symbol)
    depths = _depths_from_root(mol, root_id)
    morgan_invariants = rdMolDescriptors.GetConnectivityInvariants(
        mol,
        includeRingMembership=True,
    )
    nodes = {
        atom.GetIdx(): _node_data(
            atom,
            root_id,
            int(morgan_invariants[atom.GetIdx()]),
        )
        for atom in mol.GetAtoms()
    }

    edges: List[EdgeData] = []
    bonds = sorted(
        mol.GetBonds(),
        key=lambda bond: (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
    )
    for bond in bonds:
        first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type, is_aromatic, is_in_ring = _bond_attributes(bond)
        for source, target in _orient_bond(first, second, depths):
            edges.append(
                EdgeData(
                    source, target, bond_type, is_aromatic, is_in_ring
                )
            )

    graph = RootedGraph(root_id, nodes, edges, depths)
    validate_rooted_graph(graph)
    return graph


def validate_rooted_graph(graph: RootedGraph) -> None:
    """Raise ``COAFError`` if a rooted graph violates its invariants."""
    if graph.root_id not in graph.nodes or graph.root_id not in graph.depths:
        raise COAFError("The root must be present in both nodes and depths")
    if graph.depths[graph.root_id] != 0:
        raise COAFError("The root must have depth zero")
    if graph.nodes[graph.root_id].element != "ROOT":
        raise COAFError("The root node must have the element label 'ROOT'")
    if set(graph.nodes) != set(graph.depths):
        raise COAFError("Node and depth identifiers do not match")
    for node_id, node in graph.nodes.items():
        if node.atom_index != node_id:
            raise COAFError("A node key does not match its stored atom index")

    edge_signatures = {
        (e.source, e.target, e.bond_type, e.is_aromatic, e.is_in_ring)
        for e in graph.edges
    }
    if len(edge_signatures) != len(graph.edges):
        raise COAFError("The graph contains duplicate directed edges")
    for edge in graph.edges:
        if edge.source not in graph.nodes or edge.target not in graph.nodes:
            raise COAFError("An edge refers to an unknown node")
        source_depth = graph.depths[edge.source]
        target_depth = graph.depths[edge.target]
        if source_depth > target_depth:
            raise COAFError("An edge points toward a smaller rooted depth")
        if source_depth == target_depth:
            reverse = (
                edge.target,
                edge.source,
                edge.bond_type,
                edge.is_aromatic,
                edge.is_in_ring,
            )
            if reverse not in edge_signatures:
                raise COAFError("An equal-depth edge is missing its reverse edge")


# COAF (rooted Weisfeiler-Lehman) -------------------------------------------

NeighborContribution = Tuple[str, str, bool, Hashable]
GraphFeature = Tuple[int, Hashable]


def initial_atom_labels(graph: RootedGraph) -> Dict[int, Hashable]:
    """Return the shared Morgan-compatible labels used by COAF and POAF.

    Ordinary atoms use RDKit's standard Morgan connectivity invariants,
    including ring membership.  The attachment marker is replaced by the
    distinct abstract label ``"ROOT"``.  Both orientation-aware descriptors
    consume this same label mapping; only their feature aggregation differs.
    """
    validate_rooted_graph(graph)
    return {
        node_id: "ROOT" if node_id == graph.root_id else node.morgan_invariant
        for node_id, node in graph.nodes.items()
    }


def _neighbor_contributions(
    graph: RootedGraph,
    labels: Mapping[int, Hashable],
    node_id: int,
) -> Tuple[List[NeighborContribution], List[NeighborContribution], List[NeighborContribution]]:
    incoming: List[NeighborContribution] = []
    outgoing: List[NeighborContribution] = []
    lateral: List[NeighborContribution] = []
    for edge in graph.edges:
        if edge.source == node_id:
            direction = (
                "outgoing"
                if graph.depths[edge.source] < graph.depths[edge.target]
                else "lateral"
            )
            contribution = (
                direction, edge.bond_type, edge.is_in_ring, labels[edge.target]
            )
            (outgoing if direction == "outgoing" else lateral).append(contribution)
        elif edge.target == node_id:
            direction = (
                "incoming"
                if graph.depths[edge.source] < graph.depths[edge.target]
                else "lateral"
            )
            contribution = (
                direction, edge.bond_type, edge.is_in_ring, labels[edge.source]
            )
            (incoming if direction == "incoming" else lateral).append(contribution)
    return incoming, outgoing, lateral


def coaf_features(
    graph: RootedGraph,
    radius: int = 3,
) -> CounterType[GraphFeature]:
    """Return the explicit COAF feature multiset through ``radius`` rounds."""
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    validate_rooted_graph(graph)
    labels = initial_atom_labels(graph)
    features: CounterType[GraphFeature] = Counter(
        (0, label) for label in labels.values()
    )
    for iteration in range(1, radius + 1):
        refined: Dict[int, Hashable] = {}
        for node_id in sorted(graph.nodes):
            incoming, outgoing, lateral = _neighbor_contributions(
                graph, labels, node_id
            )
            refined[node_id] = (
                labels[node_id],
                tuple(sorted(incoming)),
                tuple(sorted(outgoing)),
                tuple(sorted(lateral)),
            )
        labels = refined
        features.update((iteration, label) for label in labels.values())
    return features


def _sha256_integer(value: Any) -> int:
    digest = hashlib.sha256(repr(value).encode("utf-8")).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _fold_binary_features(features: Iterable[Any], n_bits: int) -> List[int]:
    if n_bits <= 0:
        raise ValueError(f"n_bits must be > 0, got {n_bits}")
    fingerprint = [0] * n_bits
    for feature in features:
        fingerprint[_sha256_integer(feature) % n_bits] = 1
    return fingerprint


def coaf_fingerprint(
    graph: RootedGraph,
    radius: int = 3,
    n_bits: int = 1024,
) -> List[int]:
    """Generate a fixed-length binary COAF fingerprint."""
    return _fold_binary_features(coaf_features(graph, radius), n_bits)


def coaf_fingerprint_from_smiles(
    smiles: str,
    radius: int = 3,
    n_bits: int = 1024,
    marker_symbol: str = "Hg",
) -> List[int]:
    """Generate COAF directly from a marker-containing SMILES string."""
    return coaf_fingerprint(
        build_rooted_graph(smiles, marker_symbol), radius, n_bits
    )


# Directed linear/path fingerprint -----------------------------------------

DirectedPath = Tuple[int, ...]


def _graph_access(
    graph: RootedGraph,
) -> Tuple[Dict[int, List[EdgeData]], Dict[int, List[EdgeData]], Dict[Tuple[int, int], EdgeData]]:
    outgoing = {node_id: [] for node_id in graph.nodes}
    incoming = {node_id: [] for node_id in graph.nodes}
    lookup: Dict[Tuple[int, int], EdgeData] = {}
    for edge in graph.edges:
        outgoing[edge.source].append(edge)
        incoming[edge.target].append(edge)
        lookup[(edge.source, edge.target)] = edge
    for node_id in graph.nodes:
        outgoing[node_id].sort(
            key=lambda e: (e.target, e.bond_type, e.is_aromatic, e.is_in_ring)
        )
        incoming[node_id].sort(
            key=lambda e: (e.source, e.bond_type, e.is_aromatic, e.is_in_ring)
        )
    return outgoing, incoming, lookup


def _stable_text_integer(fields: Iterable[Any]) -> int:
    payload = "|".join(str(field) for field in fields).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _enumerate_directed_paths(
    graph: RootedGraph,
    outgoing: Mapping[int, List[EdgeData]],
    max_path_length: int,
) -> List[DirectedPath]:
    if max_path_length < 1:
        raise ValueError(
            f"max_path_length must be >= 1 edge, got {max_path_length}"
        )
    paths: List[DirectedPath] = []

    def extend(current: List[int]) -> None:
        if len(current) >= 2:
            paths.append(tuple(current))
        # A path containing n nodes contains n - 1 edges.  Public POAF path
        # length and the POAFX suffix are both defined in terms of edges.
        if len(current) - 1 == max_path_length:
            return
        for edge in outgoing[current[-1]]:
            if edge.target not in current:
                current.append(edge.target)
                extend(current)
                current.pop()

    for start in sorted(graph.nodes):
        extend([start])
    return paths


def _bond_class(edge: EdgeData) -> int:
    classes = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}
    try:
        return classes[edge.bond_type.strip().upper()]
    except KeyError as error:
        raise ValueError(f"Unsupported bond type {edge.bond_type!r}") from error


def directed_linear_features(
    graph: RootedGraph,
    max_path_length: int = 4,
) -> CounterType[int]:
    """Return directed-path feature counts.

    Paths may start at any node, must follow the outward edge orientation, and
    may not repeat a node.  ``max_path_length`` is measured in edges, so a
    maximum length of ``X`` permits paths containing up to ``X + 1`` nodes and
    corresponds to the POAFX naming convention.
    """
    validate_rooted_graph(graph)
    outgoing, incoming, lookup = _graph_access(graph)
    labels = initial_atom_labels(graph)
    seeds = {
        node_id: _sha256_integer(label)
        for node_id, label in labels.items()
    }
    paths = _enumerate_directed_paths(graph, outgoing, max_path_length)
    features: CounterType[int] = Counter()
    for path in paths:
        reversed_nodes = list(reversed(path))
        current = seeds[reversed_nodes[0]]
        for downstream, upstream in zip(reversed_nodes[:-1], reversed_nodes[1:]):
            edge = lookup[(upstream, downstream)]
            current = _stable_text_integer(
                (current, _bond_class(edge), seeds[upstream])
            )
            features[current] += 1
    return features


def directed_linear_fingerprint(
    graph: RootedGraph,
    max_path_length: int = 4,
    n_bits: int = 1024,
    binary: bool = True,
) -> List[int]:
    """Generate a POAF with maximum path length measured in edges."""
    if n_bits <= 0:
        raise ValueError(f"n_bits must be > 0, got {n_bits}")
    features = directed_linear_features(graph, max_path_length)
    fingerprint = [0] * n_bits
    for feature, count in features.items():
        index = feature % n_bits
        fingerprint[index] = 1 if binary else fingerprint[index] + count
    return fingerprint


def directed_linear_fingerprint_from_smiles(
    smiles: str,
    max_path_length: int = 4,
    n_bits: int = 1024,
    marker_symbol: str = "Hg",
    binary: bool = True,
) -> List[int]:
    """Generate a POAF from rooted SMILES.

    ``max_path_length=X`` enumerates directed simple paths containing at most
    ``X`` edges (and therefore at most ``X + 1`` nodes), corresponding to POAFX.
    """
    return directed_linear_fingerprint(
        build_rooted_graph(smiles, marker_symbol),
        max_path_length,
        n_bits,
        binary,
    )


# ECFP benchmark helpers ----------------------------------------------------

def _replace_marker_atom(mol: Mol, marker_symbol: str, replacement_symbol: str) -> Mol:
    replacement_atomic_number = Chem.GetPeriodicTable().GetAtomicNumber(
        replacement_symbol
    )
    if replacement_atomic_number <= 0:
        raise ValueError(f"Unknown replacement element {replacement_symbol!r}")
    marker_indices = [
        atom.GetIdx() for atom in mol.GetAtoms()
        if atom.GetSymbol() == marker_symbol
    ]
    if len(marker_indices) != 1:
        raise COAFError(
            f"Expected exactly one {marker_symbol!r} marker for replacement; "
            f"found {len(marker_indices)}"
        )
    editable = Chem.RWMol(mol)
    atom = editable.GetAtomWithIdx(marker_indices[0])
    atom.SetAtomicNum(replacement_atomic_number)
    atom.SetFormalCharge(0)
    result = editable.GetMol()
    Chem.SanitizeMol(result)
    return result


def ecfp_fingerprint(
    smiles: str,
    radius: int = 3,
    n_bits: int = 1024,
    *,
    marker_replacement: Optional[str] = None,
    marker_symbol: str = "Hg",
) -> DataStructs.ExplicitBitVect:
    """Generate a standard binary Morgan/ECFP fingerprint for benchmarking.

    Set ``marker_replacement`` explicitly (for example ``"O"`` or ``"N"``)
    to reproduce a benchmark control in which the artificial attachment marker
    is replaced before ECFP generation.  With ``None``, the input is used as
    supplied; this avoids silently imposing a chemistry-specific replacement.
    """
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    if n_bits <= 0:
        raise ValueError(f"n_bits must be > 0, got {n_bits}")
    mol = _parse_smiles(smiles)
    if marker_replacement is not None:
        mol = _replace_marker_atom(mol, marker_symbol, marker_replacement)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits
    )
    return generator.GetFingerprint(mol)


def fingerprint_to_bitvect(
    fingerprint: Iterable[int] | DataStructs.ExplicitBitVect,
) -> DataStructs.ExplicitBitVect:
    """Convert a binary sequence to an RDKit bit vector."""
    if isinstance(fingerprint, DataStructs.ExplicitBitVect):
        return fingerprint
    values = list(fingerprint)
    if any(value not in (0, 1, False, True) for value in values):
        raise ValueError("A binary fingerprint may contain only zero and one")
    bitvect = DataStructs.ExplicitBitVect(len(values))
    for index, value in enumerate(values):
        if value:
            bitvect.SetBit(index)
    return bitvect


def fingerprint_to_bitstring(
    fingerprint: Iterable[int] | DataStructs.ExplicitBitVect,
) -> str:
    """Return a portable ``0``/``1`` representation of a binary fingerprint."""
    return fingerprint_to_bitvect(fingerprint).ToBitString()


def tanimoto_similarity(
    first: Iterable[int] | DataStructs.ExplicitBitVect,
    second: Iterable[int] | DataStructs.ExplicitBitVect,
) -> float:
    """Calculate Tanimoto similarity between two binary fingerprints."""
    first_bv = fingerprint_to_bitvect(first)
    second_bv = fingerprint_to_bitvect(second)
    if first_bv.GetNumBits() != second_bv.GetNumBits():
        raise ValueError("Fingerprints must have the same length")
    return float(DataStructs.TanimotoSimilarity(first_bv, second_bv))

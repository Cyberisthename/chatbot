"""Toy molecular graphs used by the synthetic docking experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

DATA_ROOT = Path(__file__).resolve().parents[4] / "examples" / "med"


@dataclass
class Atom:
    identifier: str
    charge: float
    hydrophobicity: float
    hbond_donor: bool
    hbond_acceptor: bool


@dataclass
class Bond:
    a: str
    b: str
    order: int


@dataclass
class Molecule:
    name: str
    atoms: List[Atom]
    bonds: List[Bond]

    def atom_ids(self) -> List[str]:
        return [atom.identifier for atom in self.atoms]

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "atoms": [atom.__dict__ for atom in self.atoms],
            "bonds": [bond.__dict__ for bond in self.bonds],
        }


def _load_json(path: Path) -> Dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def load_catalog(root: Path | None = None) -> Dict[str, Molecule]:
    base = DATA_ROOT if root is None else Path(root)
    molecules: Dict[str, Molecule] = {}
    for path in sorted(base.glob("*.json")):
        data = _load_json(path)
        atoms = [
            Atom(
                identifier=entry["id"],
                charge=float(entry.get("charge", 0.0)),
                hydrophobicity=float(entry.get("hydrophobicity", 0.0)),
                hbond_donor=bool(entry.get("hbond_donor", False)),
                hbond_acceptor=bool(entry.get("hbond_acceptor", False)),
            )
            for entry in data.get("atoms", [])
        ]
        bonds = [
            Bond(
                a=item["a"],
                b=item["b"],
                order=int(item.get("order", 1)),
            )
            for item in data.get("bonds", [])
        ]
        molecule = Molecule(name=data.get("name", path.stem), atoms=atoms, bonds=bonds)
        molecules[molecule.name] = molecule
    if not molecules:
        raise FileNotFoundError(f"No molecules found under {base}")
    return molecules


def load_molecule(name: str, root: Path | None = None) -> Molecule:
    catalog = load_catalog(root=root)
    try:
        return catalog[name]
    except KeyError:  # pragma: no cover - defensive
        available = ", ".join(sorted(catalog))
        raise KeyError(f"Unknown molecule '{name}'. Available: {available}") from None


def molecule_features(molecule: Molecule) -> Dict[str, float]:
    charges = [atom.charge for atom in molecule.atoms]
    hydroph = [atom.hydrophobicity for atom in molecule.atoms]
    donor_count = sum(atom.hbond_donor for atom in molecule.atoms)
    accept_count = sum(atom.hbond_acceptor for atom in molecule.atoms)
    bond_orders = [bond.order for bond in molecule.bonds]
    return {
        "charge_mean": sum(charges) / max(len(charges), 1),
        "charge_abs": sum(abs(c) for c in charges),
        "hydrophobicity_mean": sum(hydroph) / max(len(hydroph), 1),
        "donors": float(donor_count),
        "acceptors": float(accept_count),
        "bond_weight": float(sum(bond_orders)),
    }


def iter_molecules(names: Sequence[str] | None = None) -> Iterable[Molecule]:
    catalog = load_catalog()
    if names:
        for name in names:
            yield catalog[name]
    else:
        yield from catalog.values()

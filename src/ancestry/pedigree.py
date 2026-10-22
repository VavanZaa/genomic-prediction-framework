"""Pedigree representation and exact rational breed fraction arithmetic.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd


class BreedFraction:
    """Represents exact breed composition fractions using rational numbers (fractions.Fraction).

    Eliminates floating-point drift and guarantees exact pedigree algebra:
    q_child = (q_sire + q_dam) / 2
    """
    def __init__(self, fractions: Optional[Dict[str, Fraction | int | float | str]] = None):
        self._comp: Dict[str, Fraction] = {}
        if fractions:
            for breed, val in fractions.items():
                frac = Fraction(val) if not isinstance(val, Fraction) else val
                if frac > 0:
                    self._comp[breed] = frac
            self._normalize()

    def _normalize(self) -> None:
        total = sum(self._comp.values())
        if total == 0:
            return
        if total != 1:
            self._comp = {b: val / total for b, val in self._comp.items()}

    def get(self, breed: str) -> Fraction:
        return self._comp.get(breed, Fraction(0, 1))

    @property
    def composition(self) -> Dict[str, Fraction]:
        return dict(self._comp)

    @classmethod
    def pure(cls, breed: str) -> BreedFraction:
        return cls({breed: Fraction(1, 1)})

    @classmethod
    def cross(cls, sire_bf: BreedFraction, dam_bf: BreedFraction) -> BreedFraction:
        all_breeds = set(sire_bf._comp.keys()) | set(dam_bf._comp.keys())
        new_comp = {}
        half = Fraction(1, 2)
        for b in all_breeds:
            f = half * sire_bf.get(b) + half * dam_bf.get(b)
            if f > 0:
                new_comp[b] = f
        return cls(new_comp)

    def matches(self, target: Dict[str, Fraction | str | float], tolerance: float = 0.0) -> bool:
        """Verify if current composition matches target fractions exactly (or within tolerance)."""
        for breed, val in target.items():
            t_frac = Fraction(val) if not isinstance(val, Fraction) else val
            actual = self.get(breed)
            if tolerance == 0.0:
                if actual != t_frac:
                    return False
            else:
                if abs(float(actual - t_frac)) > tolerance:
                    return False
        # Also ensure no unexpected unaccounted breeds
        for breed in self._comp:
            if breed not in target and self._comp[breed] > tolerance:
                return False
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BreedFraction):
            return False
        all_breeds = set(self._comp.keys()) | set(other._comp.keys())
        return all(self.get(b) == other.get(b) for b in all_breeds)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self._comp.items())))

    def __repr__(self) -> str:
        items = [f"{b}: {float(val):.4f} ({val})" for b, val in sorted(self._comp.items())]
        return f"BreedFraction({', '.join(items)})"

    def to_dict(self) -> Dict[str, float]:
        return {b: float(v) for b, v in self._comp.items()}


@dataclass
class Individual:
    id: str
    sex: str  # 'M' or 'F'
    sire_id: Optional[str] = None
    dam_id: Optional[str] = None
    breed_fraction: BreedFraction = field(default_factory=BreedFraction)
    generation: int = 0
    meta: Dict = field(default_factory=dict)

    @property
    def is_founder(self) -> bool:
        return self.sire_id is None and self.dam_id is None


class Pedigree:
    """Directed Acyclic Graph (DAG) for animal pedigree tracking and kinship calculation."""

    def __init__(self):
        self.individuals: Dict[str, Individual] = {}
        self._kinship_cache: Dict[Tuple[str, str], float] = {}

    def add_founder(
        self,
        id: str,
        sex: str,
        breed: str,
        meta: Optional[Dict] = None
    ) -> Individual:
        if id in self.individuals:
            raise ValueError(f"Individual {id} already exists in pedigree.")
        sex = sex.upper()
        if sex not in ('M', 'F'):
            raise ValueError("Sex must be 'M' or 'F'")
        ind = Individual(
            id=id,
            sex=sex,
            sire_id=None,
            dam_id=None,
            breed_fraction=BreedFraction.pure(breed),
            generation=0,
            meta=meta or {}
        )
        # Adding an animal never changes relationships among existing animals, so the cache stays valid.
        self.individuals[id] = ind
        return ind

    def add_cross(
        self,
        id: str,
        sire_id: str,
        dam_id: str,
        sex: str = 'F',
        meta: Optional[Dict] = None
    ) -> Individual:
        if id in self.individuals:
            raise ValueError(f"Individual {id} already exists in pedigree.")
        if sire_id not in self.individuals:
            raise ValueError(f"Sire {sire_id} not found in pedigree.")
        if dam_id not in self.individuals:
            raise ValueError(f"Dam {dam_id} not found in pedigree.")

        sire = self.individuals[sire_id]
        dam = self.individuals[dam_id]

        if sire.sex != 'M':
            raise ValueError(f"Specified sire {sire_id} has sex '{sire.sex}', expected 'M'.")
        if dam.sex != 'F':
            raise ValueError(f"Specified dam {dam_id} has sex '{dam.sex}', expected 'F'.")

        sex = sex.upper()
        if sex not in ('M', 'F'):
            raise ValueError("Sex must be 'M' or 'F'")

        child_bf = BreedFraction.cross(sire.breed_fraction, dam.breed_fraction)
        generation = max(sire.generation, dam.generation) + 1

        ind = Individual(
            id=id,
            sex=sex,
            sire_id=sire_id,
            dam_id=dam_id,
            breed_fraction=child_bf,
            generation=generation,
            meta=meta or {}
        )
        self.individuals[id] = ind
        return ind

    def get_ancestors(self, ind_id: str) -> Set[str]:
        ancestors: Set[str] = set()
        stack = [ind_id]
        while stack:
            curr = stack.pop()
            ind = self.individuals.get(curr)
            if ind:
                if ind.sire_id and ind.sire_id not in ancestors:
                    ancestors.add(ind.sire_id)
                    stack.append(ind.sire_id)
                if ind.dam_id and ind.dam_id not in ancestors:
                    ancestors.add(ind.dam_id)
                    stack.append(ind.dam_id)
        return ancestors

    def get_founders(self) -> List[Individual]:
        return [ind for ind in self.individuals.values() if ind.is_founder]

    def kinship(self, id1: str, id2: str) -> float:
        """Additive genetic relationship a_ij (recursive tabular method).

        Despite the method name this is the relationship-matrix entry, i.e. twice the coancestry:
        a_ii = 1 + F_i, and a_ij = 0.5 for parent-offspring between non-inbred animals. Use
        `coancestry` for the probability-of-IBD scale.
        """
        if (id1, id2) in self._kinship_cache:
            return self._kinship_cache[(id1, id2)]
        if (id2, id1) in self._kinship_cache:
            return self._kinship_cache[(id2, id1)]

        ind1 = self.individuals.get(id1)
        ind2 = self.individuals.get(id2)
        if not ind1 or not ind2:
            return 0.0

        if id1 == id2:
            # Self kinship: A_ii = 1 + F_i = 1 + 0.5 * A(sire, dam)
            if ind1.is_founder:
                val = 1.0
            else:
                sire = ind1.sire_id
                dam = ind1.dam_id
                f_i = 0.5 * self.kinship(sire, dam) if (sire and dam) else 0.0
                val = 1.0 + f_i
        else:
            # Sort by generation to expand older individual
            if ind1.generation > ind2.generation:
                older, younger = ind2, ind1
            else:
                older, younger = ind1, ind2

            if younger.is_founder:
                val = 0.0
            else:
                s_val = self.kinship(older.id, younger.sire_id) if younger.sire_id else 0.0
                d_val = self.kinship(older.id, younger.dam_id) if younger.dam_id else 0.0
                val = 0.5 * (s_val + d_val)

        self._kinship_cache[(id1, id2)] = val
        self._kinship_cache[(id2, id1)] = val
        return val

    def additive_relationship(self, id1: str, id2: str) -> float:
        """Additive relationship a_ij (alias of `kinship`)."""
        return self.kinship(id1, id2)

    def coancestry(self, id1: str, id2: str) -> float:
        """Coancestry (kinship coefficient) f_ij = a_ij / 2; equals the inbreeding of their offspring."""
        return 0.5 * self.kinship(id1, id2)

    def inbreeding(self, id: str) -> float:
        """Compute Wright's inbreeding coefficient F = a(sire, dam) / 2."""
        ind = self.individuals.get(id)
        if not ind or ind.is_founder:
            return 0.0
        if not ind.sire_id or not ind.dam_id:
            return 0.0
        return 0.5 * self.kinship(ind.sire_id, ind.dam_id)

    def to_dataframe(self) -> pd.DataFrame:
        records = []
        for ind in self.individuals.values():
            rec = {
                "id": ind.id,
                "sex": ind.sex,
                "sire_id": ind.sire_id or "",
                "dam_id": ind.dam_id or "",
                "generation": ind.generation,
                "inbreeding_F": self.inbreeding(ind.id)
            }
            for breed, frac in ind.breed_fraction.composition.items():
                rec[f"breed_{breed}"] = float(frac)
                rec[f"frac_{breed}"] = str(frac)
            records.append(rec)
        return pd.DataFrame(records)

"""Animal QTLdb and Bovine Genome Database (BGD) annotation and prior mapping engine.

The built-in catalogue is a small set of candidate-gene loci. Each record was checked on 11 September
2026 against PubMed (E-utilities: title, authors, journal, abstract) and NCBI Gene (Bos taurus
ARS-UCD2.0, GCF_002263795.3, annotation RS_2024_12). Traits and populations are those the cited study
reports, not the target use. Intervals are the gene span +/- 1 Mb (an assumed window; SNP maps on
ARS-UCD1.2 differ only slightly from ARS-UCD2.0).

The previous catalogue carried PubMed IDs that all resolved to unrelated papers, and effect sizes,
p-values, sample sizes, cM positions and QTL IDs with no traceable source; those values were removed.
Records loaded from GFF3/TSV exports keep whatever the export provides.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd

VERIFIED_ASSEMBLY = "ARS-UCD2.0 (GCF_002263795.3), NCBI annotation RS_2024_12"
WINDOW_BP = 1_000_000


@dataclass(frozen=True)
class AnimalQTLdbRecord:
    """Represents an association record from Animal QTLdb or Bovine Genome Database."""
    qtl_id: str
    trait_name: str
    trait_class: str          # e.g., "Meat and Carcass", "Growth", "Physiology"
    chrom: str
    start_bp: int
    end_bp: int
    peak_bp: Optional[int] = None
    peak_cM: Optional[float] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    breed: Optional[str] = None
    sample_size: Optional[int] = None
    pubmed_id: Optional[str] = None
    study_reference: Optional[str] = None
    candidate_gene: Optional[str] = None
    assembly: Optional[str] = None
    evidence_note: Optional[str] = None


# Backwards compatibility alias
QtlInterval = AnimalQTLdbRecord


def _gene_locus(gene: str, chrom: str, gene_start: int, gene_end: int, **kw) -> AnimalQTLdbRecord:
    return AnimalQTLdbRecord(
        qtl_id=f"CAND_{gene.split('/')[0]}", chrom=chrom, candidate_gene=gene,
        start_bp=max(1, gene_start - WINDOW_BP), end_bp=gene_end + WINDOW_BP, peak_bp=(gene_start + gene_end) // 2,
        assembly=VERIFIED_ASSEMBLY, **kw,
    )


class BGDQtlLoader:
    """Parses Bovine Genome Database / AnimalQTLdb QTL records and maps them to SNP arrays."""

    CURATED_CATALOGUE: List[AnimalQTLdbRecord] = [
        _gene_locus("LCORL/NCAPG", "6", 37_377_805, 37_556_969,
                    trait_name="Body weight and carcass traits (large-effect pleiotropic QTL)", trait_class="Growth",
                    breed="Ten US beef breeds", pubmed_id="24906442", study_reference="Saatchi et al. 2014, BMC Genomics"),
        _gene_locus("PLAG1", "14", 23_323_708, 23_375_678,
                    trait_name="Stature", trait_class="Body conformation",
                    breed="Dairy cattle (population not stated in abstract)", pubmed_id="21516082",
                    study_reference="Karim et al. 2011, Nat Genet"),
        _gene_locus("MSTN", "2", 6_278_863, 6_285_490,
                    trait_name="Double muscling (muscle mass)", trait_class="Meat and Carcass",
                    breed="Belgian Blue", pubmed_id="9288100", study_reference="Grobet et al. 1997, Nat Genet"),
        _gene_locus("IGF1", "5", 66_186_010, 66_264_047,
                    trait_name="Serum IGF-I concentration and growth", trait_class="Growth",
                    breed="Angus", pubmed_id="11465363", study_reference="Ge et al. 2001, J Anim Sci"),
        _gene_locus("DGAT1", "14", 604_178, 614_154,
                    trait_name="Milk fat content and composition", trait_class="Milk",
                    breed="Dairy cattle (population not stated in abstract)", pubmed_id="11827942",
                    study_reference="Grisart et al. 2002, Genome Res",
                    evidence_note="Pannier et al. 2010 (PMID 20416823) found no association of DGAT1 SNPs with "
                                  "intramuscular fat in crossbred Bos taurus."),
        _gene_locus("SCD", "26", 21_263_975, 21_279_184,
                    trait_name="Fatty acid composition of intramuscular fat", trait_class="Meat and Carcass",
                    breed="Japanese Black (Wagyu)", pubmed_id="15058385", study_reference="Taniguchi et al. 2004, Mamm Genome"),
        _gene_locus("TG", "14", 8_217_496, 8_453_434,
                    trait_name="Marbling score", trait_class="Meat and Carcass",
                    breed="Beef cattle (breed not stated in abstract)", sample_size=271, pubmed_id="18670061",
                    study_reference="Gan et al. 2008, J Appl Genet",
                    evidence_note="Not replicated for intramuscular fat by Pannier et al. 2010 (PMID 20416823); "
                                  "Bennett et al. 2013 (PMID 23148258) report tenderness and fat-thickness effects."),
        _gene_locus("FASN", "19", 50_776_230, 50_794_938,
                    trait_name="Fatty acid composition of adipose tissue and milk fat", trait_class="Meat and Carcass",
                    breed="Beef and dairy cattle (pasture-fed)", pubmed_id="17242864", study_reference="Morris et al. 2007, Mamm Genome"),
        _gene_locus("CAPN1", "29", 43_399_901, 43_427_418,
                    trait_name="Meat tenderness (shear force)", trait_class="Meat and Carcass",
                    breed="Piedmontese x Angus; Jersey x Limousin", pubmed_id="12542147", study_reference="Page et al. 2002, J Anim Sci"),
        _gene_locus("CAST", "7", 96_034_549, 96_167_157,
                    trait_name="Meat tenderness", trait_class="Meat and Carcass",
                    breed="Crossbred beef cattle (Hereford, Angus, Brahman and others)", pubmed_id="16478943",
                    study_reference="Casas et al. 2006, J Anim Sci"),
        _gene_locus("PRLR", "20", 38_915_986, 39_108_970,
                    trait_name="Slick hair coat / thermoregulation", trait_class="Physiology",
                    breed="Not stated in abstract", pubmed_id="25519203", study_reference="Littlejohn et al. 2014, Nat Commun"),
    ]

    # Backward compatibility alias
    CURATED_MAJOR_CATTLE_QTLS = CURATED_CATALOGUE

    def __init__(self, records: Optional[List[AnimalQTLdbRecord]] = None):
        self.records = records or list(self.CURATED_CATALOGUE)

    @property
    def intervals(self) -> List[AnimalQTLdbRecord]:
        """Backward-compatible property."""
        return self.records

    def filter(
        self,
        trait_filter: Optional[str] = None,
        trait_class: Optional[str] = None,
        breed_filter: Optional[str] = None,
        max_p_value: Optional[float] = None
    ) -> BGDQtlLoader:
        """Returns a new BGDQtlLoader filtered by trait, class, breed, or significance threshold."""
        filtered = self.records
        if trait_filter:
            tf = trait_filter.lower()
            filtered = [r for r in filtered if tf in r.trait_name.lower() or (r.candidate_gene and tf in r.candidate_gene.lower())]
        if trait_class:
            tc = trait_class.lower()
            filtered = [r for r in filtered if tc in r.trait_class.lower()]
        if breed_filter:
            bf = breed_filter.lower()
            filtered = [r for r in filtered if r.breed and bf in r.breed.lower()]
        if max_p_value is not None:
            filtered = [r for r in filtered if r.p_value is not None and r.p_value <= max_p_value]

        return BGDQtlLoader(filtered)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts QTL records into a pandas DataFrame."""
        return pd.DataFrame([
            {
                "qtl_id": r.qtl_id,
                "trait_name": r.trait_name,
                "trait_class": r.trait_class,
                "chrom": r.chrom,
                "start_bp": r.start_bp,
                "end_bp": r.end_bp,
                "peak_bp": r.peak_bp,
                "peak_cM": r.peak_cM,
                "p_value": r.p_value,
                "effect_size": r.effect_size,
                "breed": r.breed,
                "sample_size": r.sample_size,
                "pubmed_id": r.pubmed_id,
                "candidate_gene": r.candidate_gene,
                "study_reference": r.study_reference,
                "assembly": r.assembly,
                "evidence_note": r.evidence_note,
            }
            for r in self.records
        ])

    @classmethod
    def load_from_gff3(cls, gff3_path: Union[str, Path]) -> BGDQtlLoader:
        """Parses an official Animal QTLdb or BGD GFF3 annotation file."""
        records = []
        p = Path(gff3_path)
        if not p.exists():
            raise FileNotFoundError(f"GFF3 file not found: {p}")

        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 9:
                    continue
                chrom = parts[0].replace("chr", "").replace("BTA", "")
                feature_type = parts[2]
                if "QTL" in feature_type or "QTL" in parts[8] or "GWAS" in parts[8]:
                    start_bp = int(parts[3])
                    end_bp = int(parts[4])

                    # Parse attributes (semicolon-separated key=value pairs)
                    attr_dict = {}
                    for item in parts[8].split(";"):
                        if "=" in item:
                            k, v = item.split("=", 1)
                            attr_dict[k.strip()] = v.strip().strip('"')

                    qtl_id = attr_dict.get("ID", attr_dict.get("QTL_ID", f"QTL_{chrom}_{start_bp}"))
                    trait = attr_dict.get("Name", attr_dict.get("trait", attr_dict.get("Trait", "Unknown")))
                    t_class = attr_dict.get("Trait_Class", attr_dict.get("Class", "Unclassified"))
                    breed = attr_dict.get("Breed", attr_dict.get("breed", None))
                    pubmed = attr_dict.get("PubMed", attr_dict.get("PMID", attr_dict.get("pubmed_id", None)))
                    gene = attr_dict.get("gene", attr_dict.get("Candidate_Gene", None))

                    pval_str = attr_dict.get("p_value", attr_dict.get("P_value", None))
                    p_val = float(pval_str) if pval_str and pval_str != "." else None

                    eff_str = attr_dict.get("effect", attr_dict.get("Effect", None))
                    effect = float(eff_str) if eff_str and eff_str != "." else None

                    records.append(AnimalQTLdbRecord(
                        qtl_id=qtl_id,
                        trait_name=trait,
                        trait_class=t_class,
                        chrom=chrom,
                        start_bp=start_bp,
                        end_bp=end_bp,
                        p_value=p_val,
                        effect_size=effect,
                        breed=breed,
                        pubmed_id=pubmed,
                        candidate_gene=gene
                    ))

        return cls(records)

    @classmethod
    def load_from_tsv(cls, tsv_path: Union[str, Path]) -> BGDQtlLoader:
        """Parses an Animal QTLdb tab-delimited export file."""
        df = pd.read_csv(tsv_path, sep="\t", comment="#", dtype=str)
        records = []
        col_map = {c.lower(): c for c in df.columns}

        def get_val(row, *aliases):
            for a in aliases:
                col = col_map.get(a.lower())
                if col and pd.notna(row[col]) and row[col] != "":
                    return row[col]
            return None

        for _, row in df.iterrows():
            chrom = str(get_val(row, "chr", "chromosome", "chrom") or "0").replace("chr", "")
            start = int(float(get_val(row, "start", "start_bp", "bp_start") or 0))
            end = int(float(get_val(row, "end", "end_bp", "bp_end") or start + 1000))
            qtl_id = str(get_val(row, "qtl_id", "id", "qtl") or f"QTL_{chrom}_{start}")
            trait = str(get_val(row, "trait", "trait_name", "qtl_name") or "Unknown")
            t_class = str(get_val(row, "trait_class", "class", "category") or "Unclassified")
            breed = get_val(row, "breed", "breed_name")
            pubmed = get_val(row, "pubmed_id", "pmid", "pubmed")
            gene = get_val(row, "gene", "candidate_gene", "symbol")

            p_str = get_val(row, "p_value", "pvalue", "p-value", "pval")
            p_val = float(p_str) if p_str else None

            eff_str = get_val(row, "effect", "effect_size", "beta")
            effect = float(eff_str) if eff_str else None

            records.append(AnimalQTLdbRecord(
                qtl_id=qtl_id,
                trait_name=trait,
                trait_class=t_class,
                chrom=chrom,
                start_bp=start,
                end_bp=end,
                p_value=p_val,
                effect_size=effect,
                breed=breed,
                pubmed_id=pubmed,
                candidate_gene=gene
            ))

        return cls(records)

    def map_snps_to_qtls(
        self,
        bim_df: pd.DataFrame,
        trait_filter: Optional[str] = None,
        breed_filter: Optional[str] = None,
        weighting_mode: str = "binary"
    ) -> np.ndarray:
        """Maps SNPs in a PLINK .bim table to QTL intervals with optional evidence weighting.

        Parameters:
            bim_df: PLINK .bim DataFrame containing 'chrom' and 'pos_bp'.
            trait_filter: Optional substring for trait or gene selection (e.g. 'marbling', 'carcass').
            breed_filter: Optional substring for breed-specific discovery filtering.
            weighting_mode: 'binary' (default) or 'evidence' (scaled by -log10(p-value)); records without
                a p-value (all built-in records) fall back to the binary weight.

        Returns:
            np.ndarray of shape (P,) containing SNP prior weights.
        """
        P = len(bim_df)
        weights = np.ones(P, dtype=np.float64)

        loader = self.filter(trait_filter=trait_filter, breed_filter=breed_filter)
        active_intervals = loader.records

        chrom_series = bim_df["chrom"].astype(str)
        pos_series = bim_df["pos_bp"].astype(np.int64)

        for qtl in active_intervals:
            in_qtl = (chrom_series == str(qtl.chrom)) & (pos_series >= qtl.start_bp) & (pos_series <= qtl.end_bp)
            if not np.any(in_qtl):
                continue

            if weighting_mode == "evidence" and qtl.p_value and qtl.p_value > 0:
                log_p = min(20.0, max(1.0, -np.log10(qtl.p_value)))
                weights[in_qtl] += float(log_p / 4.0)
            else:
                # Default 2.5x prior weight for SNPs within verified QTL intervals
                weights[in_qtl] += 1.5

        return weights

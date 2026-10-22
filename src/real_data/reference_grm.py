"""Validate provenance before using the full-density training-reference GRM."""
from pathlib import Path
import json
import numpy as np
from .genotype_preprocessing import source_hashes, validate_ids

def load_reference_grm(path):
    path=Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Build the training-reference cache first: scripts/build_full_density_122k_grm.py ({path})")
    with np.load(path,allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    if str(data.get("preprocessing_version","")) != "train_reference_v3":
        raise ValueError("Obsolete globally preprocessed GRM; rebuild the v3 reference cache")
    expected=source_hashes([path.parent/n for n in ("SNP.raw.gz","BOA.raw.gz","Thermo.Phenotypes.csv")])
    if json.loads(str(data["source_hashes"])) != expected:
        raise ValueError("GRM source inputs changed; rebuild the v3 cache")
    ids=validate_ids(data["animal_ids"])
    for name in ("G_122k","B_122k"):
        if data[name].shape != (len(ids),len(ids)) or not np.isfinite(data[name]).all():
            raise ValueError(f"Invalid cached {name}")
    return data

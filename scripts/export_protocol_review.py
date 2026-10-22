"""Build a self-contained review package for a pre-registration, before it is locked.

The review bundle (scripts/export_review_bundle.py) asks "what do we claim and on what evidence?".
This asks a different question: **"is this experiment worth running, before we run it?"**

It exists because the last two failures in this project were design errors, not coding errors: a
feasibility gate validated only on in-domain animals (6C2), and a measurement operator that counted
reconstruction gaps as recombinations (Layer 7 H7 v1). Both were pre-registered, and both were
caught only afterwards. Pre-registration fixes when a decision is made, not whether it is sound, so
the design itself is put up for review while changing it is still free.

    python scripts/export_protocol_review.py --protocol h7v2
    python scripts/export_protocol_review.py --protocol h7v2 --dest "C:/path/to/share"

Writes output/protocol_review/<protocol>/ with PROTOCOL_REVIEW.md and protocol_review.json.
Everything cited is embedded verbatim with its SHA-256; no repository access is needed.

Manual step. No experiment, test or pipeline script calls it (tests/test_pipeline_hygiene.py).
Embeds no genotype or phenotype data.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.risk_calibration import sha256_file  # noqa: E402

OUT = ROOT / "output" / "protocol_review"


def embed(paths):
    out = {}
    for rel in paths:
        p = ROOT / rel
        if not p.exists():
            out[rel] = {"present": False}
            continue
        text = p.read_text(encoding="utf-8")
        out[rel] = {"present": True, "sha256": sha256_file(p), "bytes": p.stat().st_size,
                    "content": json.loads(text) if p.suffix == ".json" else text}
    return out


H7V2 = {
    "protocol": "h7v2",
    "title": "Layer 7 / H7 v2 - cross-species inheritance with symmetric measurement",
    "status": ("Implemented through step 2. The feasibility gate returned INFEASIBLE, so the protocol "
               "was never locked and no endpoint was computed."),
    "gate_outcome": {
        "decision": "INFEASIBLE",
        "binding_criterion": ("gap-identical transition share: fails on BOTH cohorts at EVERY rung; at "
                              "minprob 0.50 it is 0.179 observed and 0.119 painted against a bound of 0.05"),
        "rungs": {
            "0.95": {"observed": [0.132, 0.446, 6.79], "painted": [0.069, 0.443, 6.00]},
            "0.90": {"observed": [0.100, 0.432, 5.36], "painted": [0.046, 0.425, 4.50]},
            "0.80": {"observed": [0.073, 0.420, 4.38], "painted": [0.030, 0.409, 3.64]},
            "0.70": {"observed": [0.056, 0.402, 3.49], "painted": [0.022, 0.386, 2.92]},
            "0.60": {"observed": [0.038, 0.354, 2.40], "painted": [0.014, 0.327, 2.05]},
            "0.50": {"observed": [0.020, 0.179, 1.32], "painted": [0.007, 0.119, 1.18]},
        },
        "rung_columns": ["uncalled share (<= 0.02)", "gap-identical share (<= 0.05)",
                         "bound ratio (<= 1.25)"],
        "finding_1_structural_limit": (
            "Uncalled cells concentrate in short stretches that resolve back to the founder they left: "
            "at minprob 0.50 the painted cohort has only 0.7% uncalled cells yet 11.9% of transitions "
            "are gaps flanked by the same founder. Hypothesis, untested: local haplotype sharing among "
            "DO founder strains makes diplotypes ambiguous regardless of threshold, so a hard-call "
            "transition statistic cannot reach the required precision on DO data at any minprob."),
        "finding_2_painting_asymmetry": (
            "The observed cohort leaves about twice the painted cohort's uncalled share at every rung "
            "(0.132 vs 0.069 at 0.95; 0.020 vs 0.007 at 0.50). Independent random missingness and error "
            "do not reproduce real MegaMUGA information loss - confirming review target 2 with data."),
        "what_it_does_not_license": (
            "Designing v3 immediately after an infeasible gate is how analyses get redesigned until "
            "they run. A further version needs a justification that stands without this result and "
            "should pass external design review before implementation."),
        "incidents_during_implementation": [
            ("Two whole-genome reconstruction attempts were stopped before writing output because "
             "parallel qtl2 workers drove Windows commit headroom from ~40 GB to 2.9 GB. Rebuilt to run "
             "one chromosome at a time (identical HMM results; observed called share at 0.95 reproduced "
             "v1's 0.8677 exactly), at ~1.5 GB instead of 17+ GB."),
            ("Blinding slip: raw pre-filter run counts for the painted cohort were displayed from the "
             "reconstruction log. Recorded in the protocol before the gate ran; the observed cohort's "
             "counts were not displayed; no design element changed afterwards."),
        ],
    },
    "one_line": ("Does the framework's meiosis simulation reproduce a real Diversity Outbred "
                 "population's founder-mosaic structure, with no parameter fitted to mice?"),
    "why_this_version_exists": (
        "H7 v1 was pre-registered, locked across 67 files, unblinded once, and is VOID. Its observed "
        "statistic counted switches as (segments per haplotype) - (chromosomes); at minprob 0.95, "
        "13.2% of markers were uncalled and every gap split a segment that was then counted as a "
        "switch. Of 164,914 transitions only 8.2% were clean state changes, 44.6% were one segment "
        "split by a gap with an IDENTICAL founder state on both sides, and 47.2% were unknowable. "
        "The deeper fault was asymmetry: the observed side passed through a lossy reconstruction "
        "while the simulated side was exact, so even with the counting fixed the two sides would "
        "not have measured the same quantity."),
    "the_central_change": (
        "Simulated genomes are painted into MegaMUGA genotypes from the real founder strain files "
        "and passed through the IDENTICAL qtl2 reconstruction, so both sides suffer the same "
        "information loss and any residual difference is about recombination rather than about "
        "reconstruction."),
    "what_is_being_asked_of_the_reviewer": (
        "The gate has already returned INFEASIBLE, so the question is no longer whether to run this "
        "design. It is: (1) was INFEASIBLE reached correctly - is the gate implemented as the protocol "
        "states, and is the transition classification sound; (2) is the structural-limit hypothesis "
        "(founder haplotype sharing makes hard-call transition statistics unworkable on DO data) "
        "plausible, and how would you test it; and (3) should there be a v3 at all, or is "
        "'infeasible on this dataset' the right place for H7 to stop. The original design review "
        "targets below are kept for the record; some are now moot because no endpoint was reached."),
    "design_summary": {
        "simulated_side": ("Frozen v1 engine and parameters (8 founder strains both sexes, 100 "
                           "breeding pairs, random outcrossing, seed 20260912), producing 99 animals "
                           "at G=8 and 192 at G=11 to match the observed cohort exactly."),
        "painting": ("Each simulated haplotype takes the covering founder strain's allele at every "
                     "MegaMUGA marker (founders are inbred, one allele each); haplotypes combine to "
                     "A/B/H; then missing calls at the measured 2.17%, genotyping error at 0.2%, and "
                     "markers with an uninformative founder genotype set missing."),
        "reconstruction": ("calc_genoprob (error_prob 0.002, Haldane) -> maxmarg at the gate-selected "
                           "minprob -> guess_phase -> vectorised run-length extraction, applied "
                           "identically to both sides."),
        "short_run_filter": ("founder runs shorter than 2 cM are absorbed into the longer neighbour on "
                             "both sides, to remove single-marker noise flips introduced by forcing "
                             "calls at lower minprob."),
        "feasibility_gate": {
            "purpose": "no endpoint is computed until this passes; uses only properties of the "
                       "reconstructions, never a comparison between them",
            "ladder": [0.95, 0.90, 0.80, 0.70, 0.60, 0.50],
            "rule": "take the LARGEST minprob meeting all three criteria on BOTH sides; if none "
                    "does, H7 v2 is INFEASIBLE and that is the reported result",
            "criteria": {"uncalled_marker_share": "<= 0.02",
                         "gap_with_identical_flanking_state_share": "<= 0.05",
                         "upper_over_lower_switch_bound_ratio": "<= 1.25"},
            "v1_would_have_failed_all_three": {"uncalled": 0.132, "gap_identical": 0.446,
                                               "bound_ratio": 6.2},
        },
        "endpoints": {
            "E1_prime_primary": "mean switches per haplotype, per generation; observed inside the "
                                "simulated 95% interval AND relative difference <= 0.20",
            "E2_prime": "median and p90 segment length within +/-0.20 relatively",
            "E3_prime": "ratio of mean switches G11/G8 within +/-0.15",
            "E4_descriptive": "founder contribution balance; never an endpoint",
            "why_primary_changed": ("v1 made the ratio primary to cancel the observed/simulated "
                                    "asymmetry; v2 removes that asymmetry, so the absolute "
                                    "comparison is meaningful and is the quantity H7 is about."),
        },
    },
    "review_agenda": [
        {"target": "the 2 cM short-run filter",
         "ask": ("This is the step most likely to bias the result. Short segments are exactly what "
                 "accumulates with generation, so a filter that removes them could suppress the "
                 "generation-scaling signal E3' tests and flatten the E1' difference between G=8 and "
                 "G=11. It is applied symmetrically, which helps - but symmetric removal of a "
                 "generation-dependent quantity is not obviously neutral. Is 2 cM defensible? Should "
                 "the threshold instead be derived from the marker spacing, or the filter replaced by "
                 "reporting with and without?"),
         "our_view": "the weakest part of the design; we would rather be talked out of it"},
        {"target": "whether painting is genuinely symmetric",
         "ask": ("Missing calls and genotyping errors are applied at random, but real missingness is "
                 "correlated across markers and samples (plate effects, poorly performing probes). If "
                 "real missingness is clustered and simulated missingness is not, the observed side "
                 "still loses more information than the simulated side, and the asymmetry v2 exists "
                 "to remove partially survives. Does this matter at 2.17%? Should missingness be "
                 "resampled from the real per-marker pattern instead?")},
        {"target": "the feasibility gate's independence",
         "ask": ("The gate selects minprob using criteria that involve no endpoint, then the same "
                 "reconstruction is used to compute endpoints. Is that leakage? Our argument is that "
                 "the criteria are properties of call quality, not of the observed-vs-simulated "
                 "comparison, and the ladder and rule are fixed in advance. Is that sufficient, or "
                 "should minprob be fixed outright at one pre-declared value and the test simply "
                 "declared infeasible if it fails?")},
        {"target": "random mating versus the real DO funnel",
         "ask": ("The simulation random-mates from generation 1; the real DO descends from partially "
                 "inbred pre-CC funnel lines. Symmetric measurement does not fix a structural "
                 "difference in the breeding design itself. Does that invalidate the absolute "
                 "comparison E1' even in principle, leaving only the ratio E3' interpretable - i.e. "
                 "was v1 right to make the ratio primary, for a different reason than we thought?"),
         "our_view": "a genuine threat to E1' as primary; we would like this attacked hard"},
        {"target": "the simulated interval",
         "ask": ("Only one painted cohort passes through the pipeline, because each reconstruction "
                 "costs about 45 minutes of CPU. The 95% interval therefore combines an animal-level "
                 "bootstrap within that cohort with the replicate-to-replicate drift spread taken "
                 "from the frozen v1 EXACT simulation. Is combining a bootstrap from one pipeline "
                 "with a drift component from a different pipeline valid, or does E1' need several "
                 "painted cohorts to have an honest interval?")},
        {"target": "the tolerances",
         "ask": ("+/-0.20 for E1'/E2' and +/-0.15 for E3' are asserted as 'the magnitudes that would "
                 "distinguish a species-general inheritance model from a cattle-tuned one'. That is a "
                 "judgement, not a calculation. Is there a principled basis - for example the "
                 "simulated replicate spread, or the reconstruction's own measurement error - that "
                 "should set them instead?")},
        {"target": "what a pass would actually license",
         "ask": ("If every endpoint passes, the claim is 'the inheritance layer is species-general'. "
                 "Is that supportable from one population of 291 mice with two generation levels, or "
                 "should the claim be narrower - e.g. 'consistent with species-general recombination "
                 "at this resolution'? We would rather fix the wording before the result exists.")},
    ],
    "known_weaknesses_we_already_accept": [
        "Two generation levels only (G=8 with 99 mice, G=11 with 192), so E3' rests on two points.",
        "Autosomes only; the X chromosome is excluded because the simulator does not model "
        "sex-specific inheritance.",
        "No sex-specific recombination map, though real mouse maps differ by sex.",
        "Founder contribution balance cannot be tested (E4 descriptive) because real DO contributions "
        "are unequal for design reasons the simulator does not model.",
        "The gate may fail on every rung of the ladder, in which case the honest result is "
        "'infeasible on this dataset' rather than another void run.",
    ],
    "what_this_cannot_claim_under_any_outcome": [
        "Nothing about G_genomic: no mouse phenotype exists in the archive.",
        "Nothing about cross-species absolute phenotype prediction, which is excluded as meaningless.",
        "Nothing about the cattle results; no cattle threshold, panel or model is touched.",
    ],
    "implementation_status": {
        "exists": ["scripts/paint_simulated_do_genotypes.py (tested: tests/test_painting.py)",
                   "scripts/reconstruct_do_mosaics_v2.R (per-chromosome, ladder in one pass)",
                   "scripts/run_layer7_h7v2.py (gate run; unblind stage written, never run)",
                   "scripts/run_layer7_h7.py (frozen v1 simulation engine, reused unchanged)",
                   "src/inheritance/meiosis.py and chromosome.py (run an 8-founder design unmodified)"],
        "to_be_written": ["nothing for v2: the gate returned INFEASIBLE, so no lock script is needed"],
        "note": ("Steps 1-2 preceded the lock by design so the gate could not see an endpoint. The gate "
                 "failed, so the protocol ends here without a lock."),
        "estimated_cost": "actual: ~18 minutes per per-chromosome reconstruction, both cohorts in parallel",
    },
    "documents": [
        "docs/LAYER7_H7v2_PROTOCOL.md",
        "docs/LAYER7_H7_PROTOCOL.md",
        "docs/LAYER7_HYPOTHESIS.md",
        "docs/EVIDENCE_POLICY.md",
        "reports/2026-09-12_layer7_h7_results.md",
    ],
    "results_for_context": [
        "output/layer7v2/gate_result.json",
        "output/layer7v2/painting_summary.json",
        "output/layer7/results_h7.json",
        "output/layer7/results_h7_diagnosis.json",
        "output/layer7/simulated_mosaics.json",
        "output/layer7/observed_summary.json",
    ],
    "code": [
        "scripts/paint_simulated_do_genotypes.py",
        "scripts/reconstruct_do_mosaics_v2.R",
        "scripts/run_layer7_h7v2.py",
        "tests/test_painting.py",
        "scripts/run_layer7_h7.py",
        "scripts/run_layer7_h7_diagnosis.py",
        "scripts/reconstruct_do_mosaics.R",
        "src/inheritance/meiosis.py",
        "src/inheritance/chromosome.py",
    ],
}

PROTOCOLS = {"h7v2": H7V2}


def to_markdown(b):
    d = b["design_summary"]
    L = [f"# {b['title']}", "",
         f"**{b['status']}**", "",
         f"Generated {b['generated_utc']} by `{b['generated_by']}`.", "",
         "## The question", "", b["one_line"], "",
         "## Why this version exists", "", b["why_this_version_exists"], "",
         "**The central change.** " + b["the_central_change"], "",
         "## Gate outcome: " + b["gate_outcome"]["decision"], "",
         f"**Binding criterion.** {b['gate_outcome']['binding_criterion']}", "",
         "| minprob | Observed: " + " / ".join(b["gate_outcome"]["rung_columns"]) +
         " | Painted: same |", "|---|---|---|"]
    for mp, r in b["gate_outcome"]["rungs"].items():
        L.append(f"| {mp} | {' / '.join(str(x) for x in r['observed'])} | "
                 f"{' / '.join(str(x) for x in r['painted'])} |")
    L += ["", f"**Finding 1 — a structural limit.** {b['gate_outcome']['finding_1_structural_limit']}", "",
          f"**Finding 2 — painting is not symmetric.** {b['gate_outcome']['finding_2_painting_asymmetry']}", "",
          f"**What it does not license.** {b['gate_outcome']['what_it_does_not_license']}", "",
          "**Incidents during implementation:**", ""]
    L += [f"- {x}" for x in b["gate_outcome"]["incidents_during_implementation"]]
    L += ["", "## What is being asked of you", "", b["what_is_being_asked_of_the_reviewer"], "",
         "## Design", "",
         f"- **Simulated side.** {d['simulated_side']}",
         f"- **Painting.** {d['painting']}",
         f"- **Reconstruction.** {d['reconstruction']}",
         f"- **Short-run filter.** {d['short_run_filter']}", "",
         "### Feasibility gate", "",
         f"{d['feasibility_gate']['purpose']}.", "",
         f"- Ladder: {d['feasibility_gate']['ladder']}",
         f"- Rule: {d['feasibility_gate']['rule']}", "",
         "| Criterion | Bound |", "|---|---|"]
    for k, v in d["feasibility_gate"]["criteria"].items():
        L.append(f"| {k.replace('_', ' ')} | {v} |")
    v1 = d["feasibility_gate"]["v1_would_have_failed_all_three"]
    L += ["", f"The voided v1 configuration would have failed all three "
              f"(uncalled {v1['uncalled']}, gap-identical {v1['gap_identical']}, "
              f"bound ratio {v1['bound_ratio']}).", "",
          "### Endpoints", ""]
    for k, v in d["endpoints"].items():
        L.append(f"- **{k}** — {v}")
    L += ["", "## Review agenda", "",
          "Ordered by how much we want them attacked.", ""]
    for t in b["review_agenda"]:
        L.append(f"### {t['target']}")
        L.append("")
        L.append(t["ask"])
        if t.get("our_view"):
            L.append("")
            L.append(f"*Our view: {t['our_view']}.*")
        L.append("")
    L += ["## Weaknesses we already accept", ""] + [f"- {x}" for x in b["known_weaknesses_we_already_accept"]]
    L += ["", "## What this cannot claim under any outcome", ""] + \
         [f"- {x}" for x in b["what_this_cannot_claim_under_any_outcome"]]
    im = b["implementation_status"]
    L += ["", "## Implementation status", "", "Already written:", ""] + [f"- `{x}`" for x in im["exists"]]
    L += ["", "Still to write:", ""] + [f"- `{x}`" for x in im["to_be_written"]]
    L += ["", im["note"], "", f"Estimated cost: {im['estimated_cost']}.", "",
          "## Embedded files", "",
          "Each is included in full, with its SHA-256, in the JSON alongside this document.", "",
          "| File | SHA-256 | Bytes |", "|---|---|---|"]
    for group in b["embedded_files"].values():
        for rel, meta in group.items():
            if meta.get("present"):
                L.append(f"| `{rel}` | `{meta['sha256'][:16]}…` | {meta['bytes']:,} |")
    return "\n".join(L) + "\n"


def build(spec):
    b = dict(spec)
    b["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    b["generated_by"] = "scripts/export_protocol_review.py"
    b["purpose"] = ("Self-contained review package for a pre-registration that has NOT been locked or "
                    "run. The design is open to change; findings are free now and expensive later.")
    b["embedded_files"] = {
        "documents": embed(spec["documents"]),
        "results_for_context": embed(spec["results_for_context"]),
        "code": embed(spec["code"]),
    }
    for key in ("documents", "results_for_context", "code"):
        b.pop(key)
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    ap.add_argument("--dest", type=Path, help="Also copy the package to this folder")
    args = ap.parse_args()

    b = build(PROTOCOLS[args.protocol])
    dest = OUT / args.protocol
    dest.mkdir(parents=True, exist_ok=True)
    js, md = dest / "protocol_review.json", dest / "PROTOCOL_REVIEW.md"
    js.write_text(json.dumps(b, indent=2), encoding="utf-8")
    md.write_text(to_markdown(b), encoding="utf-8")

    missing = [rel for group in b["embedded_files"].values()
               for rel, meta in group.items() if not meta.get("present")]
    if missing:
        raise SystemExit(f"cited files are not present, so the package is not self-contained: {missing}")
    n = sum(1 for group in b["embedded_files"].values() for m in group.values() if m.get("present"))
    print(f"protocol: {args.protocol}   review targets: {len(b['review_agenda'])}   embedded files: {n}")
    print(f"wrote {js.relative_to(ROOT)} ({js.stat().st_size:,} bytes)")
    print(f"wrote {md.relative_to(ROOT)} ({md.stat().st_size:,} bytes)")
    if args.dest:
        args.dest.mkdir(parents=True, exist_ok=True)
        for p in (js, md):
            shutil.copy2(p, args.dest / p.name)
            print(f"copied {p.name} -> {args.dest / p.name}")


if __name__ == "__main__":
    main()

# Validation of this source snapshot

Prepared on 2026-10-22. Checks ran on Windows using existing local toolchains and installed dependencies, with imports directed at this exported copy. This is not a clean-room dependency-install test.

| Check | Result |
| --- | --- |
| unit-tests | passed |

149 tests passed and 11 were skipped. Only explicitly synthetic PLINK fixtures are included. Empirical genotype/phenotype checks require separately obtained inputs; --run-research-assets opts into them.

The repository contains selected main source, required small configuration/reference inputs and documentation. Installed dependencies, model weights, caches, private records and bulk research outputs are excluded. Reproduce the commands in README.md after installing the declared dependencies.

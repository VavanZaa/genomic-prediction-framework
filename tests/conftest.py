

# Portfolio source-only asset profile
# Original assertions remain intact. Enable only after restoring the matching
# research artifacts with --run-research-assets.
import pytest as _portfolio_pytest

@_portfolio_pytest.hookimpl(specname="pytest_addoption")
def pytest_portfolio_asset_options(parser):
    parser.addoption("--run-research-assets", action="store_true", default=False,
                     help="Run checks requiring excluded local research archives/indexes")

@_portfolio_pytest.hookimpl(specname="pytest_collection_modifyitems")
def pytest_portfolio_asset_collection(config, items):
    if config.getoption("--run-research-assets"):
        return
    requires_assets = {'tests/test_provenance.py::test_real_cattle_folder_holds_only_empirical_data', 'tests/test_simulation_fidelity.py::test_real_haplotype_resampling_fidelity', 'tests/test_layer3_real_data.py::test_florida_out_of_sample_genomic_prediction', 'tests/test_simulation_fidelity.py::test_methods_prediction_convergence', 'tests/test_layer3_real_data.py::test_florida_environmental_deconfounding', 'tests/test_layer3b_boa_prediction.py::test_canonical_split_reproduces_identical_m0_baseline', 'tests/test_layer3_real_data.py::test_florida_real_dataset_ingestion', 'tests/test_layer3b_boa_prediction.py::test_boa_allele_coding_against_known_reference_animals', 'tests/test_provenance.py::test_florida_loader_reports_empirical'}
    for item in items:
        if item.nodeid in requires_assets:
            item.add_marker(_portfolio_pytest.mark.skip(reason='requires excluded empirical genotype/phenotype inputs; use --run-research-assets after supplying the documented data'))

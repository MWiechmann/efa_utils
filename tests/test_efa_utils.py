import pytest
import pandas as pd
import numpy as np
from efa_utils import (
    reduce_multicoll,
    kmo_check,
    parallel_analysis,
    iterative_efa,
    print_sorted_loadings,
    rev_items_and_return,
    factor_int_reliability
)

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_samples = 1000
    n_factors = 3
    n_variables = 15  # 5 variables per factor

    # Create factor loadings with very clear structure
    loadings = np.zeros((n_variables, n_factors))
    for i in range(n_factors):
        # Strong main loadings
        loadings[i*5:(i+1)*5, i] = np.random.uniform(0.8, 0.9, 5)
        
        # Add minimal cross-loadings
        for j in range(n_factors):
            if i != j:
                loadings[i*5:(i+1)*5, j] = np.random.uniform(0.0, 0.1, 5)

    # Generate uncorrelated factor scores
    factor_scores = np.random.normal(0, 1, (n_samples, n_factors))

    # Generate observed variables with minimal noise
    observed = np.dot(factor_scores, loadings.T) + np.random.normal(0, 0.1, (n_samples, n_variables))

    # Convert to DataFrame
    df = pd.DataFrame(observed, columns=[f'var_{i}' for i in range(n_variables)])
    
    return df

def test_reduce_multicoll(sample_data):
    vars_li = sample_data.columns.tolist()
    try:
        reduced_vars = reduce_multicoll(sample_data, vars_li, print_details=True)
        assert len(reduced_vars) <= len(vars_li)
        assert all(var in vars_li for var in reduced_vars)
        assert len(reduced_vars) >= 2  # We expect to retain at least 2 variables
        print(f"Reduced variables: {reduced_vars}")
    except Exception as e:
        pytest.fail(f"reduce_multicoll raised {type(e).__name__} unexpectedly: {e}")

def test_kmo_check(sample_data):
    vars_li = sample_data.columns.tolist()
    kmo = kmo_check(sample_data, vars_li, return_kmos=True)
    assert isinstance(kmo, tuple)
    assert len(kmo) == 2
    assert 0.7 <= kmo[1] <= 1  # Overall KMO should be good for this data

def test_parallel_analysis(sample_data):
    vars_li = sample_data.columns.tolist()
    n_factors = parallel_analysis(sample_data, vars_li, print_graph=False, print_table=False)
    assert 2 <= n_factors <= 4  # We expect to detect 3 factors, but allow some flexibility

def test_iterative_efa(sample_data):
    vars_li = sample_data.columns.tolist()
    efa, final_vars = iterative_efa(sample_data, vars_li, n_facs=3, comm_thresh=0.3, print_details=False, print_par_plot=False, print_par_table=False)
    assert efa is not None
    assert len(final_vars) >= 10  # We expect to retain most variables
    assert efa.n_factors == 3

def test_iterative_efa_pca(sample_data):
    try:
        from sklearn.decomposition import PCA
        from factor_analyzer.rotator import Rotator
        
        vars_li = sample_data.columns.tolist()
        pca, final_vars = iterative_efa(
            sample_data, vars_li, n_facs=3, 
            comm_thresh=0.3, cross_thres=0.5,  # Much more lenient cross-loading threshold
            print_details=True,  # Enable details for debugging
            print_par_plot=False, print_par_table=False,
            use_pca=True
        )
        assert pca is not None
        assert len(final_vars) >= 10  # We expect to retain most variables
        assert isinstance(pca, PCA)
        assert pca.n_components == 3
        
        # Check explained variance ratio is reasonable
        assert np.sum(pca.explained_variance_ratio_) > 0.5  # Should explain >50% of variance
        
        # Check loadings calculation
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        assert loadings.shape == (len(final_vars), 3)
        
        # Check communalities
        comms = np.sum(loadings**2, axis=1)
        assert all(comm >= 0.3 for comm in comms)  # All should be above comm_thresh
        
    except ImportError:
        pytest.skip("scikit-learn is not installed, skipping PCA test")

def test_iterative_efa_pca_no_sklearn(sample_data, monkeypatch):
    # Remove sklearn.decomposition from both sys.modules and globals
    import sys
    monkeypatch.delitem(sys.modules, 'sklearn.decomposition', raising=False)
    monkeypatch.delitem(globals(), 'PCA', raising=False)
    
    # Also remove it from efa_utils_functions module
    import efa_utils.efa_utils_functions as efa_utils_functions
    monkeypatch.delattr(efa_utils_functions, 'PCA', raising=False)
    
    vars_li = sample_data.columns.tolist()
    with pytest.raises(ImportError, match="PCA from sklearn.decomposition is required for PCA analysis"):
        iterative_efa(
            sample_data, vars_li, n_facs=3,
            print_details=False, print_par_plot=False, print_par_table=False,
            use_pca=True
        )

def test_parallel_analysis_pca(sample_data):
    vars_li = sample_data.columns.tolist()
    n_components = parallel_analysis(
        sample_data, vars_li,
        print_graph=False, print_table=False,
        extraction="components"
    )
    assert 2 <= n_components <= 4  # We expect to detect 3 components, but allow some flexibility

def test_print_sorted_loadings(sample_data):
    vars_li = sample_data.columns.tolist()
    efa, _ = iterative_efa(sample_data, vars_li, n_facs=3, comm_thresh=0.3, print_details=False, print_par_plot=False, print_par_table=False)
    # This function prints output, so we're just checking it runs without errors
    try:
        print_sorted_loadings(efa, vars_li)
    except Exception as e:
        pytest.fail(f"print_sorted_loadings raised {type(e).__name__} unexpectedly: {e}")

def test_print_sorted_loadings_pca(sample_data):
    try:
        from sklearn.decomposition import PCA
        vars_li = sample_data.columns.tolist()
        pca, final_vars = iterative_efa(
            sample_data, vars_li, n_facs=3,
            comm_thresh=0.3, print_details=False,
            print_par_plot=False, print_par_table=False,
            use_pca=True
        )
        # Test that print_sorted_loadings runs without errors for PCA objects
        try:
            print_sorted_loadings(pca, final_vars)
            # Verify output contains "component" instead of "factor"
            # This would require capturing stdout, but we're at least ensuring no errors
        except Exception as e:
            pytest.fail(f"print_sorted_loadings with PCA raised {type(e).__name__} unexpectedly: {e}")
    except ImportError:
        pytest.skip("scikit-learn is not installed, skipping PCA test")

def test_rev_items_and_return(sample_data):
    vars_li = sample_data.columns.tolist()
    efa, _ = iterative_efa(sample_data, vars_li, n_facs=3, comm_thresh=0.3, print_details=False, print_par_plot=False, print_par_table=False)
    new_df, items_per_fact_dict = rev_items_and_return(sample_data, efa, vars_li)
    assert isinstance(new_df, pd.DataFrame)
    assert isinstance(items_per_fact_dict, dict)
    assert len(new_df.columns) >= len(sample_data.columns)
    assert len(items_per_fact_dict) == 3  # We expect 3 factors

def test_factor_int_reliability(sample_data):
    vars_li = sample_data.columns.tolist()
    efa, _ = iterative_efa(sample_data, vars_li, n_facs=3, comm_thresh=0.3, print_details=False, print_par_plot=False, print_par_table=False)
    _, items_per_fact_dict = rev_items_and_return(sample_data, efa, vars_li)
    try:
        reliability = factor_int_reliability(sample_data, items_per_fact_dict, print_results=False)
        assert isinstance(reliability, tuple), f"Expected tuple, got {type(reliability)}"
        assert len(reliability) == 2, f"Expected tuple of length 2, got length {len(reliability)}"
        fac_reliab, fac_reliab_excl = reliability
        assert isinstance(fac_reliab, pd.DataFrame), f"Expected DataFrame, got {type(fac_reliab)}"
        assert isinstance(fac_reliab_excl, dict), f"Expected dict, got {type(fac_reliab_excl)}"
        assert not fac_reliab.empty, "DataFrame is empty"
        assert all(0.7 <= r <= 1.0 for r in fac_reliab['cronbach']), "Cronbach's alpha values out of expected range"
        print(f"fac_reliab:\n{fac_reliab}")
        print(f"fac_reliab_excl keys: {fac_reliab_excl.keys()}")
    except ImportError:
        pytest.skip("reliabilipy is not installed, skipping this test")

def test_kmo_check_warning_message(capsys):
    # Create a dataset with highly correlated variables to trigger the warning
    np.random.seed(0)
    n_samples = 100
    A = np.random.rand(n_samples)
    # Create variables highly correlated with A
    B = A * 0.99 + np.random.rand(n_samples) * 0.01
    C = A * 0.98 + np.random.rand(n_samples) * 0.02

    data = pd.DataFrame({'A': A, 'B': B, 'C': C})
    vars_li = ['A', 'B', 'C']

    # Run kmo_check to trigger the informative message
    kmo_check(data, vars_li)

    # Capture the output
    captured = capsys.readouterr()
    output = captured.out

    # Check if the informative message is in the output
    assert "The analysis detected high correlations between variables." in output, \
        "Informative message not found in output"

    # Optionally, check if the overall KMO is still printed
    assert "Overall KMO" in output, "Overall KMO not found in output"

    # Print output for debugging (optional)
    print(output)

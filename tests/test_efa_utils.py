import pytest
import pandas as pd
import numpy as np
from efa_utils import reduce_multicoll, kmo_check, parallel_analysis, iterative_efa, print_sorted_loadings, rev_items_and_return, factor_int_reliability

# Generate a sample dataset
@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    X = np.random.randn(n_samples, n_features)
    df = pd.DataFrame(X, columns=[f'var_{i}' for i in range(n_features)])
    return df

def test_reduce_multicoll(sample_data):
    vars_li = sample_data.columns.tolist()
    reduced_vars = reduce_multicoll(sample_data, vars_li)
    assert len(reduced_vars) <= len(vars_li)
    assert all(var in vars_li for var in reduced_vars)

def test_kmo_check(sample_data):
    vars_li = sample_data.columns.tolist()
    kmo = kmo_check(sample_data, vars_li, return_kmos=True)
    assert isinstance(kmo, tuple)
    assert len(kmo) == 2
    assert 0 <= kmo[1] <= 1  # Overall KMO should be between 0 and 1

def test_parallel_analysis(sample_data):
    vars_li = sample_data.columns.tolist()
    n_factors = parallel_analysis(sample_data, vars_li, print_graph=False, print_table=False)
    assert isinstance(n_factors, int)
    assert 1 <= n_factors <= len(vars_li)

def test_iterative_efa(sample_data):
    vars_li = sample_data.columns.tolist()
    efa, final_vars = iterative_efa(sample_data, vars_li, print_details=False, print_par_plot=False, print_par_table=False)
    assert len(final_vars) <= len(vars_li)
    assert all(var in vars_li for var in final_vars)

def test_print_sorted_loadings(sample_data):
    vars_li = sample_data.columns.tolist()
    efa, _ = iterative_efa(sample_data, vars_li, print_details=False, print_par_plot=False, print_par_table=False)
    # This function prints output, so we're just checking it runs without errors
    print_sorted_loadings(efa, vars_li)

def test_rev_items_and_return(sample_data):
    vars_li = sample_data.columns.tolist()
    efa, _ = iterative_efa(sample_data, vars_li, print_details=False, print_par_plot=False, print_par_table=False)
    new_df, items_per_fact_dict = rev_items_and_return(sample_data, efa, vars_li)
    assert isinstance(new_df, pd.DataFrame)
    assert isinstance(items_per_fact_dict, dict)
    assert len(new_df.columns) >= len(sample_data.columns)

def test_factor_int_reliability(sample_data):
    vars_li = sample_data.columns.tolist()
    efa, _ = iterative_efa(sample_data, vars_li, print_details=False, print_par_plot=False, print_par_table=False)
    _, items_per_fact_dict = rev_items_and_return(sample_data, efa, vars_li)
    reliability = factor_int_reliability(sample_data, items_per_fact_dict, print_results=False)
    assert isinstance(reliability, pd.DataFrame)
    assert not reliability.empty
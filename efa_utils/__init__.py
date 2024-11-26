"""EFA Utilities (efa_utils)
Custom utility functions for exploratory factor analysis with the factor_analyzer package.

Functions:
- reduce_multicoll: Reduce multicollinearity in a dataset (intended for EFA).
- kmo_check: Check the Kaiser-Meyer-Olkin measure of sampling adequacy (KMO) and Bartlett's test of sphericity.
- parallel_analysis: Perform parallel analysis to determine the number of factors to retain.
- iterative_efa: Perform iterative exploratory factor analysis.
- print_sorted_loadings: Print strongly loading variables for each factor.
- rev_items_and_return: Reverse-code items and return a new dataframe.
- factor_int_reliability: Calculate and print internal reliability for each factor.

Note: Some functions may raise warnings about matrix inversion when dealing with highly correlated data.

Requirements:
- factor_analyzer
- numpy
- pandas
- statsmodels (for reduce_multicoll and kmo_check)
- matplotlib (for parallel_analysis and iterative_efa with parallel analysis option)
- reliabilipy (for factor_int_reliability)

For detailed function descriptions, please refer to the individual function docstrings.
"""

from .efa_utils_functions import (
    reduce_multicoll,
    kmo_check,
    parallel_analysis,
    iterative_efa,
    print_sorted_loadings,
    rev_items_and_return,
    factor_int_reliability
)

try:
    from reliabilipy import reliability_analysis
except ImportError:
    reliability_analysis = None

__all__ = [
    "reduce_multicoll",
    "kmo_check",
    "parallel_analysis",
    "iterative_efa",
    "print_sorted_loadings",
    "rev_items_and_return",
    "factor_int_reliability",
    "reliability_analysis"
]

__version__ = "0.8.5"

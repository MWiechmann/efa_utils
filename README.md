# efa_utils
Custom utility functions for exploratory factor analysis

## Installation
Install with pip:
```bash
pip install efa_utils
```

## Functions
### efa_utils.reduce_multicoll
Function to reduce multicollinearity in a dataset (intended for EFA). Uses the determinant of the correlation matrix to determine if multicollinearity is present. If the determinant is below a threshold (0.00001 by default), the function will drop the variable with the highest VIF until the determinant is above the threshold. Requires statsmodels package.
### efa_utils.kmo_check
Function to check the Kaiser-Meyer-Olkin measure of sampling adequacy (KMO) and Bartlett's test of sphericity for a dataset. Requires statsmodels package. Main use is to print a report of total KMO and item KMOs, but can also return the KMO values.

### efa_utils.parallel_analysis
Function to perform parallel analysis to determine the number of factors to retain. Requires matplotlib.

### efa_utils.iterative_efa
Function to perform iterative exploratory factor analysis. Runs EFA with an iterative process, eliminating variables with low communality, low main loadings or high cross loadings in a stepwise process. If parallel analysis option is to be used, requires matplotlib.

### efa_utils.print_sorted_loadings
Print strongly loading variables for each factor. Will only print loadings above load_thresh for each factor.

### efa_utils.rev_items_and_return
Takes an EFA object and automatically reverse-codes (Likert-scale) items where necessary and adds the reverse-coded version to a new dataframe. Returns the new dataframe.

### efa_utils.factor_int_reliability
Takes a pandas dataframe and dictionary with name of factors as keys and list of variables as values. Prints results for the internal reliability for each factor. Requires reliability package.
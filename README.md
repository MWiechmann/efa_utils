# efa_utils

Custom utility functions for exploratory factor analysis with the factor_analyzer package.

## Installation

Install with pip:

```bash
pip install efa_utils
```

For optional dependencies:

```bash
pip install efa_utils[optional]
```

## Requirements

- Python 3.11+
- numpy
- pandas
- factor-analyzer
- statsmodels (for reduce_multicoll and kmo_check)
- matplotlib (optional, for parallel_analysis and iterative_efa with parallel analysis option)
- reliabilipy (optional, for factor_int_reliability)
- scikit-learn (optional, for PCA functionality in iterative_efa)

## Functions

### efa_utils.reduce_multicoll

Reduces multicollinearity in a dataset intended for EFA. Uses the determinant of the correlation matrix to determine if multicollinearity is present. If the determinant is below a threshold (0.00001 by default), the function will drop the variable with the highest VIF until the determinant is above the threshold.

### efa_utils.kmo_check

Checks the Kaiser-Meyer-Olkin measure of sampling adequacy (KMO) and Bartlett's test of sphericity for a dataset. Main use is to print a report of total KMO and item KMOs, but can also return the KMO values.

### efa_utils.parallel_analysis

Performs parallel analysis to determine the number of factors to retain. Requires matplotlib (optional dependency).

### efa_utils.iterative_efa

Performs iterative exploratory factor analysis or principal component analysis (PCA). Runs EFA/PCA with an iterative process, eliminating variables with low communality, low main loadings or high cross loadings in a stepwise process. 

For EFA (default), uses factor_analyzer package. For PCA (when use_pca=True), uses scikit-learn's PCA implementation. PCA functionality requires scikit-learn (optional dependency).

If parallel analysis option is used, it requires matplotlib (optional dependency).

### efa_utils.print_sorted_loadings

Prints strongly loading variables for each factor. Will only print loadings above a specified threshold for each factor.

### efa_utils.rev_items_and_return

Takes an EFA object and automatically reverse-codes (Likert-scale) items where necessary and adds the reverse-coded version to a new dataframe. Returns the new dataframe.

### efa_utils.factor_int_reliability

Calculates and prints the internal reliability for each factor. Takes a pandas dataframe and dictionary with name of factors as keys and list of variables as values. Requires reliabilipy (optional dependency).

## Usage

Here's a basic example of how to use efa_utils with both EFA and PCA:

```python
import pandas as pd
from efa_utils import reduce_multicoll, kmo_check, parallel_analysis, iterative_efa

# Load your data
data = pd.read_csv('your_data.csv')

# Reduce multicollinearity
reduced_vars = reduce_multicoll(data, data.columns)

# Check KMO
kmo_check(data, reduced_vars)

# For EFA:
# Perform parallel analysis
n_factors = parallel_analysis(data, reduced_vars)

# Perform iterative EFA
efa, final_vars = iterative_efa(data, reduced_vars, n_facs=n_factors)

# Print EFA results
print("EFA Results:")
print(f"Final variables: {final_vars}")
print(efa.loadings_)

# For PCA:
# Perform parallel analysis with components
n_components = parallel_analysis(data, reduced_vars, extraction="components")

# Perform iterative PCA
pca, final_vars = iterative_efa(
    data, reduced_vars, n_facs=n_components,
    use_pca=True  # This enables PCA instead of EFA
)

# Print PCA results
print("\nPCA Results:")
print(f"Final variables: {final_vars}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
# Calculate loadings (standardized components)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
print("Component loadings:")
print(loadings)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

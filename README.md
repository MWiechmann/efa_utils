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

## Functions

### efa_utils.reduce_multicoll

Reduces multicollinearity in a dataset intended for EFA. Uses the determinant of the correlation matrix to determine if multicollinearity is present. If the determinant is below a threshold (0.00001 by default), the function will drop the variable with the highest VIF until the determinant is above the threshold.

### efa_utils.kmo_check

Checks the Kaiser-Meyer-Olkin measure of sampling adequacy (KMO) and Bartlett's test of sphericity for a dataset. Main use is to print a report of total KMO and item KMOs, but can also return the KMO values.

### efa_utils.parallel_analysis

Performs parallel analysis to determine the number of factors to retain. Requires matplotlib (optional dependency).

### efa_utils.iterative_efa

Performs iterative exploratory factor analysis. Runs EFA with an iterative process, eliminating variables with low communality, low main loadings or high cross loadings in a stepwise process. If parallel analysis option is used, it requires matplotlib (optional dependency).

### efa_utils.print_sorted_loadings

Prints strongly loading variables for each factor. Will only print loadings above a specified threshold for each factor.

### efa_utils.rev_items_and_return

Takes an EFA object and automatically reverse-codes (Likert-scale) items where necessary and adds the reverse-coded version to a new dataframe. Returns the new dataframe.

### efa_utils.factor_int_reliability

Calculates and prints the internal reliability for each factor. Takes a pandas dataframe and dictionary with name of factors as keys and list of variables as values. Requires reliabilipy (optional dependency).

## Usage

Here's a basic example of how to use efa_utils:

```python
import pandas as pd
from efa_utils import reduce_multicoll, kmo_check, parallel_analysis, iterative_efa

# Load your data
data = pd.read_csv('your_data.csv')

# Reduce multicollinearity
reduced_vars = reduce_multicoll(data, data.columns)

# Check KMO
kmo_check(data, reduced_vars)

# Perform parallel analysis
n_factors = parallel_analysis(data, reduced_vars)

# Perform iterative EFA
efa, final_vars = iterative_efa(data, reduced_vars, n_facs=n_factors)

# Print results
print(f"Final variables: {final_vars}")
print(efa.loadings_)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
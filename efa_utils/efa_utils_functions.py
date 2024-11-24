import copy
import warnings
import numpy as np
import pandas as pd
import factor_analyzer as fa
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from factor_analyzer.rotator import Rotator

# optional imports
try:
    from reliabilipy import reliability_analysis
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

try:
    from sklearn.decomposition import PCA
except ImportError:
    pass

# Function to reduce multicollinearity
def reduce_multicoll(df, vars_li, det_thre=0.00001, vars_descr=None, print_details=True, deletion_method='pairwise', keep_vars=None):
    """
    Function to reduce multicollinearity in a dataset (intended for EFA).
    Uses the determinant of the correlation matrix to determine if multicollinearity is present.
    If the determinant is below a threshold (0.00001 by default),
    the function will drop the variable with the highest VIF until the determinant is above the threshold.

    In cases where multiple variables have the same highest VIF, the function uses the following tiebreakers:
    1. The variable with the highest sum of absolute correlations with other variables is chosen.
    2. If there's still a tie, the variable with the most missing data is chosen.

    Parameters:
    df (pandas dataframe): dataframe containing the variables to be checked for multicollinearity
    vars_li (list): list of variables to be checked for multicollinearity
    det_thre (float): Threshold for the determinant of the correlation matrix. Default is 0.00001.
                      If the determinant is below this threshold, the function will drop the variable
                      with the highest VIF until the determinant is above the threshold.
    vars_descr (list or dict): Dataframe or dictionary containing the variable descriptions (variable names as index/key).
                       If provided, the function will also print the variable descriptions additionally to the variable names.
    print_details (bool): If True, the function will print a detailed report of the process. Default is True.
    deletion_method (str): Method for handling missing data. Options are 'listwise' or 'pairwise' (default).
    keep_vars (list): List of variables that should not be removed during the multicollinearity reduction process.

    Returns:
    reduced_vars(list): List of variables after multicollinearity reduction.
    """
    if deletion_method not in ['listwise', 'pairwise']:
        raise ValueError("deletion_method must be either 'listwise' or 'pairwise'")

    reduced_vars = copy.deepcopy(vars_li)
    if keep_vars is None:
        keep_vars = []
    print("Beginning check for multicollinearity")

    curr_vars = reduced_vars.copy()

    if deletion_method == 'listwise':
        vars_corr = df[reduced_vars].corr()
        count_missing = df[vars_li].isna().any(axis=1).sum()
        if count_missing > 0:
            print(f"This requires dropping missing values. The procedure will ignore {count_missing} cases with missing values")
    else:  # pairwise
        vars_corr = df[reduced_vars].corr(method='pearson', min_periods=1)
        print("Using pairwise deletion for handling missing data")

    det = np.linalg.det(vars_corr)
    print(f"\nDeterminant of initial correlation matrix: {det}\n")

    if det > det_thre:
        print(f"Determinant is > {det_thre}. No issues with multicollinearity detected.")
        return reduced_vars

    print("Starting to remove redundant variables by assessing multicollinearity with VIF...\n")

    while det <= det_thre:
        if len(curr_vars) < 2:
            print(f"Not enough variables left (only {len(curr_vars)}). Stopping iteration.")
            return None, curr_vars

        # Calculate VIF for current variables
        vif_df = pd.DataFrame()
        vif_df["Variable"] = curr_vars
        vif_df["VIF"] = [vif(pd.DataFrame(df[curr_vars]).dropna().values, i) for i in range(len(curr_vars))]

        # Exclude variables in keep_vars from being dropped
        vif_df = vif_df[~vif_df["Variable"].isin(keep_vars)]

        # Find variable with highest VIF
        max_vif = vif_df["VIF"].max()
        max_vif_var = vif_df.loc[vif_df["VIF"] == max_vif, "Variable"].iloc[0]

        # Remove variable with highest VIF
        curr_vars.remove(max_vif_var)
        print(f"Removed variable '{max_vif_var}' with VIF={max_vif}")
        if vars_descr is not None and max_vif_var in vars_descr:
            print(f"Variable description: {vars_descr[max_vif_var]}\n")

        # Recalculate determinant
        if deletion_method == 'listwise':
            vars_corr = df[curr_vars].corr()
        else:  # pairwise
            vars_corr = df[curr_vars].corr(method='pearson', min_periods=1)

        det = np.linalg.det(vars_corr)
        print(f"Determinant after removing '{max_vif_var}': {det}\n")

    return curr_vars

# Function to check KMO
def kmo_check(df, vars_li, dropna_thre=0, check_item_kmos=True, return_kmos=False, vars_descr=None):
    """Function to check the Kaiser–Meyer–Olkin (KMO) measure of sampling adequacy of a dataset and print a report.
    Requires statsmodels package.
    The KMO value is a measure of the suitability of data for factor analysis.
    The KMO value ranges from 0 to 1, where 0 indicates that the correlations are too spread out to be useful for factor analysis,
    and values close to 1 indicate that correlation patterns are relatively compact and that the factors are well defined.

    Parameters:
    df (pandas dataframe): dataframe containing the variables to be checked for multicollinearity
    vars_li (list): list of variables to be checked for multicollinearity
    dropna_thre (int): Threshold for the number of missing values. Default is 0. If the number of missing values is above this threshold, the function will drop the variable. If the SVD does not converge, try increasing this threshold.
    check_item_kmos (bool): If True, the function will also check the KMO for each item. Default is True.
    return_kmos (bool): If True, the function will return the item KMOs value and the overall KMO. Default is False.
    vars_descr (pandas dataframe or dictionary): Dataframe or dictionary containing the variable descriptions (variable names as index/key). If provided, the function will also print the variable descriptions additionally to the variable names.

    Returns:
    kmo (numpy.ndarray): Array with the KMO score per item and the overall KMO score.
    """
    # drop missing values
    if dropna_thre > 0:
        df = df.dropna(subset=vars_li, thresh=dropna_thre)

    # calculate KMO
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        kmo = fa.factor_analyzer.calculate_kmo(df[vars_li])

        # Check if we got the specific warning about Moore-Penrose inverse
        for warning in w:
            if "Moore-Penrose" in str(warning.message):
                print("\nNote: The analysis detected high correlations between variables.")
                print("This is often normal in factor analysis and doesn't necessarily indicate a problem,")
                print("especially given the high KMO values below.")
                print("However, if you want to investigate further, you could:")
                print("1. Check for extremely high correlations between variables (e.g., r > 0.9)")
                print("2. Consider using the reduce_multicoll() function to address potential multicollinearity")
                print("3. Examine if any variables are linear combinations of others\n")

    print(f"Overall KMO: {kmo[1]}")

    if check_item_kmos:
        # Check KMO for each variable
        low_item_kmo = False
        for i, item_kmo in enumerate(kmo[0]):
            if item_kmo < .6:
                low_item_kmo = True
                print(f"Low KMO for {vars_li[i]} : {item_kmo}")
                if vars_descr is not None:
                    print(f"('{vars_descr[vars_li[i]]}')")

        if not low_item_kmo:
            print("All item KMOs are >.6")

    if return_kmos:
        return kmo

# Function to conduct parallel analysis
def parallel_analysis(
    df, vars_li, k=200, facs_to_display=15, print_graph=True,
    print_table=True, return_rec_n=True, extraction="minres",
    percentile=99, standard=1.1, missing='pairwise'):
    """Function to perform parallel analysis on a dataset.
    
    Parameters:
    df (pandas dataframe): Dataframe containing the data
    vars_li (list): List of variables to analyze
    k (int): Number of random datasets to generate. Default is 200.
    facs_to_display (int): Number of factors to display in output. Default is 15.
    print_graph (bool): Whether to print scree plot. Default is True.
    print_table (bool): Whether to print table of eigenvalues. Default is True.
    return_rec_n (bool): Whether to return recommended number of factors. Default is True.
    extraction (str): Method for factor extraction. Default is "minres".
    percentile (int): Percentile to use for random data. Default is 99.
    standard (float): Multiplier for random data threshold. Default is 1.1.
    missing (str): Method for handling missing data. Options are 'pairwise' or 'listwise'. Default is 'pairwise'.

    Returns:
    int: Suggested number of factors if return_rec_n is True
    """
    # Check for valid missing parameter
    if missing not in ['pairwise', 'listwise']:
        raise ValueError("missing must be either 'pairwise' or 'listwise'")

    if missing == 'listwise':
        # Remove rows with any NaN values
        df = df.dropna()
        corr_matrix = df[vars_li].corr()
        n = len(df)
    else:  # pairwise
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        # Calculate correlation matrix with pairwise deletion
        corr_matrix = df[vars_li].corr(method='pearson', min_periods=1)
        n = df[vars_li].notna().sum().min()  # Use the minimum number of non-missing values

    m = len(vars_li)

    # Check for singular matrix
    if np.linalg.matrix_rank(corr_matrix) < len(vars_li):
        raise ValueError("Singular matrix: Check for redundant or constant variables.")

    # Eigenvalues computation based on extraction method
    if extraction == "components":
        # Direct PCA eigenvalues
        evs = np.linalg.eigvalsh(corr_matrix)[::-1]
    else:
        efa = fa.FactorAnalyzer(rotation=None, method=extraction, n_factors=m)
        efa.fit(corr_matrix)
        evs = efa.get_eigenvalues()[0]

    # Prepare FactorAnalyzer object for random data
    if extraction == "components":
        par_efa = fa.FactorAnalyzer(rotation=None, n_factors=m)
    else:
        par_efa = fa.FactorAnalyzer(rotation=None, method=extraction, n_factors=m)

    # Create list to store the eigenvalues from random data
    ev_par_list = []

    # Run the fit 'k' times over a random matrix
    successful_runs = 0
    while successful_runs < k:
        random_data = np.random.normal(size=(n, m))
        while np.linalg.matrix_rank(random_data) < m:
            random_data = np.random.normal(size=(n, m))
        random_corr = np.corrcoef(random_data.T)
        try:
            par_efa.fit(random_corr)
            ev_par_list.append(pd.Series(par_efa.get_eigenvalues()[0], index=range(1, m+1)))
            successful_runs += 1
        except np.linalg.LinAlgError:
            # Skip iteration if the random_corr is singular
            continue

    ev_par_df = pd.DataFrame(ev_par_list)

    # Get percentile for the eigenvalues
    par_per = ev_par_df.quantile(percentile / 100)

    # Adjust facs_to_display to avoid index errors
    facs_to_display = min(facs_to_display, m)

    if print_graph and 'plt' in globals():
        # Draw graph
        plt.figure(figsize=(10, 6))

        # Line for eigenvalue 1
        plt.plot([1, facs_to_display + 1], [1, 1], 'k--', alpha=0.3)
        # For the random data (parallel analysis)
        plt.plot(range(1, len(par_per.iloc[:facs_to_display]) + 1),
                 par_per.iloc[:facs_to_display], 'b', label=f'EVs - random: {percentile}th percentile', alpha=0.4)
        # Markers and line for actual EFA eigenvalues
        plt.scatter(range(1, len(evs[:facs_to_display]) + 1), evs[:facs_to_display])
        plt.plot(range(1, len(evs[:facs_to_display]) + 1),
                 evs[:facs_to_display], label='EVs - survey data')

        plt.title('Parallel Analysis Scree Plots', {'fontsize': 20})
        if extraction == "components":
            plt.xlabel('Components', {'fontsize': 15})
        else:
            plt.xlabel('Factors', {'fontsize': 15})
        plt.xticks(range(1, facs_to_display + 1), range(1, facs_to_display + 1))
        plt.ylabel('Eigenvalue', {'fontsize': 15})
        plt.legend()
        plt.show()

    # Find the suggested number of factors
    suggested_factors = 0
    for factor_n in range(1, facs_to_display + 1):
        cur_ev_par = par_per.iloc[factor_n - 1]
        cur_ev_efa = evs[factor_n - 1]
        
        if cur_ev_par * standard >= cur_ev_efa:
            break
        suggested_factors = factor_n

    # Print table if requested
    if print_table:
        print(f"\n{'Factor':<10}{'EV - random data':<25}{'EV survey data':<15}")
        print("-" * 50)
        
        for factor_n in range(1, facs_to_display + 1):
            cur_ev_par = par_per.iloc[factor_n - 1]
            cur_ev_efa = evs[factor_n - 1]
            print(f"{factor_n:<10}{cur_ev_par:<25.2f}{cur_ev_efa:<15.2f}")
        
        print(f"\nSuggested number of factors: {suggested_factors}")

    if return_rec_n:
        return suggested_factors

# Function to run iterative EFA
def iterative_efa(data, vars_analsis, n_facs=4, rotation_method="Oblimin",
                  comm_thresh=0.2, main_thresh=0.4, cross_thres=0.3, load_diff_thresh=0.2,
                  print_details=False, print_par_plot=False, print_par_table=False,
                  par_k=100, par_n_facs=15, iterative=True, auto_stop_par=False,
                  items_descr=None, do_det_check=True,
                  do_kmo_check=True, kmo_dropna_thre=0, use_pca=False,
                  never_exclude=None, par_percentile=99, par_standard=1.1):
    """Run EFA or PCA with iterative process, eliminating variables with low communality, low main loadings or high cross loadings in a stepwise process.

    Parameters:
    data (pandas dataframe): Dataframe with data to be analyzed
    vars_analsis (list): List of variables to be analyzed
    n_facs (int): Number of factors/components to extract
    rotation_method (str): Rotation method to be used. Default is "Oblimin". Has to be one of the methods supported by the factor_analyzer package.
    comm_thresh (float): Threshold for communalities. Variables with communality below this threshold will be dropped from analysis.
    main_thresh (float): Threshold for main loadings. Variables with main loadings below this threshold will be dropped from analysis.
    cross_thres (float): Threshold for cross loadings. Variables with cross loadings above this threshold will be dropped from analysis.
    load_diff_thresh (float): Threshold for difference between main and cross loadings. Variables with difference between main and cross loadings below this threshold will be dropped from analysis.
    print_details (bool): If True, print details for each step of the iterative process.
    print_par_plot (bool): If True, print parallel analysis scree plot for each step of the iterative process.
    print_par_table (bool): If True, print table with eigenvalues from the parallel each step of the iterative process.
    par_k (int): Number of EFAs over a random matrix for parallel analysis.
    par_n_facs (int): Number of factors to display for parallel analysis.
    iterative (bool): NOT IMPLEMENTED YET. If True, run iterative process. If False, run EFA with all variables.
    auto_stop_par (bool): If True, stop the iterative process when the suggested number of factors from parallel analysis is lower than the requested number of factors. In that case, no EFA object or list of variables is returned.
    items_descr (pandas series): Series with item descriptions. If provided, the function will print the item description for each variable that is dropped from the analysis.
    do_det_check (bool): If True, check the determinant of the correlation matrix after the final solution is found.
    do_kmo_check (bool): If True, check the Kaiser-Meyer-Olkin measure of sampling adequacy after the final solution is found.
    kmo_dropna_thre (int): Threshold for the number of missing values. If the number of missing values is above this threshold, the function will drop the variable. If the SVD does not converge, try increasing this threshold.
    use_pca (bool): If True, use PCA instead of EFA. Default is False.
    never_exclude (list): List of variables that should never be excluded from the analysis. Default is None.
    par_percentile (int): Percentile to use for random data in parallel analysis. Default is 99.
    par_standard (float): Multiplier for random data threshold in parallel analysis. Default is 1.1.

    Returns:
    (analyzer, curr_vars): Tuple with analyzer object (EFA or PCA) and list of variables that were analyzed in the last step of the iterative process.
    """
    # Convert vars_analsis to a list if it's an Index object
    if isinstance(vars_analsis, pd.Index):
        vars_analsis = vars_analsis.tolist()
    
    # Initialize analyzer object (EFA or PCA)
    if use_pca:
        if 'PCA' not in globals():
            raise ImportError("PCA from sklearn.decomposition is required for PCA analysis. Please install scikit-learn.")
        analyzer = PCA(n_components=n_facs)
    else:
        analyzer = fa.FactorAnalyzer(n_factors=n_facs, rotation=rotation_method)

    # Initialize never_exclude list if None
    if never_exclude is None:
        never_exclude = []

    # Marker to indicate whether the final solution was found
    final_solution = False

    # List of variables used for current factor solution
    curr_vars = copy.deepcopy(vars_analsis)

    i = 1
    while not final_solution:
        # Fit analyzer
        if len(curr_vars) < 2:
            print(f"Not enough variables left (only {len(curr_vars)}). Stopping iteration.")
            return None, curr_vars

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            if use_pca:
                # Standardize data for PCA
                data_std = (data[curr_vars] - data[curr_vars].mean()) / data[curr_vars].std()
                analyzer.fit(data_std)
                # Calculate unrotated loadings from components
                unrotated_loadings = analyzer.components_.T * np.sqrt(analyzer.explained_variance_)
                # Apply rotation
                rotator = Rotator(method=rotation_method)
                rotated_loadings = rotator.fit_transform(unrotated_loadings)
                loadings = pd.DataFrame(rotated_loadings, index=curr_vars)
                # Calculate communalities (these don't change with rotation)
                comms = pd.Series(np.sum(unrotated_loadings**2, axis=1), index=curr_vars)
            else:
                analyzer.fit(data[curr_vars])
                loadings = pd.DataFrame(analyzer.loadings_, index=curr_vars)
                comms = pd.Series(analyzer.get_communalities(), index=curr_vars)

            if len(w) > 0:
                print("Warning during fitting:")
                print(w[-1].message)
                print("This may indicate high multicollinearity in the data.")

        print(f"Fitted solution #{i}\n")

        # print screeplot and/or table and/or check for auto-stopping for parallel analysis
        if print_par_plot or print_par_table or auto_stop_par:
            suggested_n_facs = parallel_analysis(
                data, curr_vars, k=par_k, facs_to_display=par_n_facs,
                print_graph=print_par_plot, print_table=print_par_table,
                extraction="components" if use_pca else "minres",
                percentile=par_percentile, standard=par_standard)

            if (suggested_n_facs < n_facs) and auto_stop_par:
                print("\nAuto-Stop based on parallel analysis: "
                      f"Parallel analysis suggests {suggested_n_facs} {'components' if use_pca else 'factors'}. "
                      f"That is less than the currently requested number ({n_facs})."
                      "Analysis stopped. No analyzer object or list of variables will be returned.")
                return

        # Check 1: Check communalities
        print("\nChecking for low communalities")
        mask_low_comms = comms < comm_thresh
        # Exclude never_exclude variables from mask
        mask_low_comms[never_exclude] = False

        if not any(mask_low_comms):
            print(f"All communalities above {comm_thresh}\n")
        else:
            bad_items = comms[mask_low_comms].index.tolist()
            print(
                f"Detected {len(bad_items)} items with low communality. Excluding them for next analysis.\n")
            for item in bad_items:
                if print_details:
                    print(f"\nRemoved item {item}\nCommunality: {comms[item]:.4f}\n")
                    if items_descr is not None:
                        print(f"Item description: {items_descr[item]}\n")
            curr_vars = [var for var in curr_vars if var not in bad_items]
            i += 1
            continue

        # Check 2: Check for low main loading
        print("Checking for low main loading")
        max_loadings = abs(loadings).max(axis=1)
        mask_low_main = max_loadings < main_thresh
        # Exclude never_exclude variables from mask
        mask_low_main[never_exclude] = False

        if not any(mask_low_main):
            print(f"All main loadings above {main_thresh}\n")
        else:
            bad_items = max_loadings[mask_low_main].index
            print(
                f"Detected {len(bad_items)} items with low main loading. Excluding them for next analysis.\n")
            for item in bad_items:
                if print_details:
                    print(f"\nRemoved item {item}\nMain (absolute) Loading: {abs(loadings.loc[item]).max():.4f}\n")
                    if items_descr is not None:
                        print(f"Item description: {items_descr[item]}\n")
                curr_vars.remove(item)
            i += 1
            continue

        # Check 3: Check for high cross loadings
        print("Checking for high cross loadings")

        crossloads_df = pd.DataFrame(index=curr_vars)

        crossloads_df["main_load"] = abs(loadings).max(axis=1)
        crossloads_df["cross_load"] = abs(loadings).apply(
            lambda row: row.nlargest(2).values[-1], axis=1)
        crossloads_df["diff"] = crossloads_df["main_load"] - crossloads_df["cross_load"]

        # For PCA, only use cross_thres, not load_diff_thresh
        if use_pca:
            mask_high_cross = crossloads_df["cross_load"] > cross_thres
        else:
            mask_high_cross = (crossloads_df["cross_load"] > cross_thres) | (
                crossloads_df["diff"] < load_diff_thresh)
        
        # Exclude never_exclude variables from mask
        mask_high_cross[never_exclude] = False

        if not any(mask_high_cross):
            if use_pca:
                print(f"All cross-loadings below {cross_thres}\n")
            else:
                print(
                    f"All cross-loadings below {cross_thres}"
                    f" and differences between main loading and crossloadings above {load_diff_thresh}.\n"
                )
        else:
            bad_items = crossloads_df[mask_high_cross].index
            print(
                f"Detected {len(bad_items)} items with high cross loading. Excluding them for next analysis.\n")
            for item in bad_items:
                if print_details:
                    print(f"Removed item {item}\nLoadings: \n{loadings.loc[item]}\n")
                    if items_descr is not None:
                        print(f"Item description: {items_descr[item]}\n")
                curr_vars.remove(item)
            i += 1
            continue

        print("Final solution reached.")
        final_solution = True

        if do_det_check:
            try:
                corrs = data[curr_vars].corr()
                det = np.linalg.det(corrs)
                print(f"\nDeterminant of correlation matrix: {det}")
                if det > 0.00001:
                    print("Determinant looks good!")
                else:
                    print("Determinant is smaller than 0.00001!")
                    print(
                        "Consider using stricter criteria and/or removing highly correlated vars")
            except Exception as e:
                print(f"Error during determinant calculation: {e}")

        if do_kmo_check:
            try:
                kmo_check(data[curr_vars], curr_vars, dropna_thre=kmo_dropna_thre, check_item_kmos=True, return_kmos=False, vars_descr=items_descr)
            except Exception as e:
                print(f"Error during KMO check: {e}")

        # Check for Heywood cases (only for EFA)
        if not use_pca:
            comms = analyzer.get_communalities()
            if comms.max() >= 1.0:
                print(f"Heywood case found for item {curr_vars[comms.argmax()]}. Communality: {comms.max()}")
            else:
                print("No Heywood case found.")

    return (analyzer, curr_vars)

# Function to print main loadings for each factor/component
def print_sorted_loadings(analyzer, item_labels, load_thresh=0.4, descr=None):
    """Print strongly loading variables for each factor/component. Will only print loadings above load_thresh for each factor/component.

    Parameters:
    analyzer (object): EFA or PCA object. For EFA, must be created with factor_analyzer package. For PCA, must be created with sklearn.decomposition.
    item_labels (list): List of item labels
    load_thresh (float): Threshold for main loadings. Only loadings above this threshold will be printed for each factor/component.
    descr (list or dict): List or dictionary of item descriptions. If provided, item descriptions will be printed.

    Returns:
    None
    """
    # Check if analyzer is a PCA object
    is_pca = isinstance(analyzer, PCA)

    if is_pca:
        # Calculate unrotated loadings from components
        unrotated_loadings = analyzer.components_.T * np.sqrt(analyzer.explained_variance_)
        # Apply rotation (using Oblimin by default)
        rotator = Rotator(method="Oblimin")
        rotated_loadings = rotator.fit_transform(unrotated_loadings)
        loadings = pd.DataFrame(rotated_loadings, index=item_labels)
    else:
        loadings = pd.DataFrame(analyzer.loadings_, index=item_labels)

    n_load = loadings.shape[1]

    if descr is not None:
        if isinstance(descr, list):
            loadings["descr"] = descr
        elif isinstance(descr, dict):
            loadings["descr"] = loadings.index.map(descr)

    for i in range(n_load):
        mask_relev_loads = abs(loadings[i]) > load_thresh
        sorted_loads = loadings[mask_relev_loads].sort_values(
            i, key=abs, ascending=False)
        print(f"Relevant loadings for {'component' if is_pca else 'factor'} {i}")
        if descr is not None:
            print(sorted_loads[[i, "descr"]].to_string(), "\n")
        else:
            print(sorted_loads[i].to_string(), "\n")

# Function to automatically reverse-code (Likert-scale) items where necessary
def rev_items_and_return(df, efa, item_labels, load_thresh=0.4, min_score=1, max_score=5):
    """Takes an EFA object and automatically reverse-codes (Likert-scale) items where necessary
    and adds the reverse-coded version to a new dataframe.
    Will only reverse-code items with main loadings above load_thresh for each factor.

    Parameters:
    df (pandas dataframe): Dataframe containing items to be reverse-coded
    efa (object): EFA object. Has to be created with factor_analyzer package.
    item_labels (list): List of item labels
    load_thresh (float): Threshold for main loadings. Only loadings above this threshold will be reverse-coded for each factor.
    min_score (int): Minimum possible score for items
    max_score (int): Maximum possible score for items

    Returns:
    (new_df, items_per_fact_dict): Tuple containing new dataframe with reverse-coded items and dictionary with a list of items per factor
    """
    new_df = df.copy()
    loadings = pd.DataFrame(efa.loadings_, index=item_labels)
    n_load = loadings.shape[1]

    items_per_fact_dict = {}

    # loop through n factors
    # determine relevant items that are positive (can just be used as is)
    # and items with negative loads (need to be reversed)
    for i in range(n_load):
        mask_pos_loads = loadings[i] > load_thresh
        mask_neg_loads = loadings[i] < -load_thresh
        pos_items = loadings[mask_pos_loads].index.tolist()
        neg_items = loadings[mask_neg_loads].index.tolist()

        # add items with positive items directly to dict
        items_per_fact_dict[i] = pos_items

        # create reverse-coded item in new_df for items with negative loadings
        for item in neg_items:
            rev_item_name = f"{item}_rev"
            new_df[rev_item_name] = (new_df[item] - (max_score + min_score)) * -1
            items_per_fact_dict[i].append(rev_item_name)

    return new_df, items_per_fact_dict

def factor_int_reliability(df, items_per_factor, measures=["cronbach", "omega_total", "omega_hier"], check_if_excluded=True, print_results=True, return_results=True):
    """Calculates and prints the internal reliability for each factor in a dataframe.
    Requires reliabilipy package.
    Available reliability measures are Cronbach's alpha, Omega Total and Omega Hierarchical.
    If a factor contains only 2 items, the reliability is calculated using the Spearman-Brown instead
    (see Eisinger, Grothenhuis & Pelzer, 2013: https://link.springer.com/article/10.1007/s00038-012-0416-3).

    Parameters:
    df (pandas dataframe): Dataframe containing items to compute reliability for
    items_per_factor (dict): Dictionary with a list of items per factor. Should have the structure {"factor_name_1": ["col_name_item_1", "col_name_item_2", ...]; "factor_name_2": ...}.
    measures (list): List of reliability measures to calculate. Possible values: "cronbach", "omega_total", "omega_hier". Default: ["cronbach", "omega_total", "omega_hier"]
    check_if_excluded (bool): If True, will also examine reliability when each item is excluded and print the results. Default is True.
    print_results (bool): If True, will print the results. Default is True.
    return_results (bool): If True, will return the results. Default is True.

    Returns:
    When check_if_excluded is False, returns:
    fac_reliab(pd.DataFrame): Dataframe with reliability estimates for each factor
    When check_if_excluded is True, returns a tuple with the following elements:
    fac_reliab(pd.DataFrame): Dataframe with reliability estimates for each factor
    fac_reliab_excl(dict): Dictionary with reliability estimates for each factor when each item is excluded. Keys are factor numbers, values are dataframes with reliability estimates. Each row gives reliability estimates for excluding one item from that factor.
    """

    # Create df to store measures for whole factors
    fac_reliab = pd.DataFrame(index=items_per_factor.keys(), columns=measures)
    # dict to store dfs to store measures for each item excluded
    if check_if_excluded:
        fac_reliab_excl = {}
    
    # Loop over factors
    for factor_n, items in items_per_factor.items():
        if len(items) > 2:
            ra = reliability_analysis(raw_dataset=df[items], is_corr_matrix=False, impute="median")

            # Check for Heywood case
            # reliabilipy runs into trouble when fa_g is a Heywood case
            # Will catch Warning and warn the user about the Heywood case
            # In general good idea to check for Heywood case though
            # will also check for Heywood case for fa_f and warn if there is one
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ra.fit()
                if len(w) > 0:
                    comms_g = ra.fa_g.get_communalities()
                    if comms_g.max() >= 1.0:
                        print(f"Heywood case found for item {items[comms_g.argmax()]} for the common factor of factor #{factor_n}! Communality: {comms_g.max()}")
                    else:
                        print(f"Warning for factor {factor_n}! Error: {w[-1].message}")

            # Also check fa_f for Heywood case
            comms_f = ra.fa_f.get_communalities()
            if comms_f.max() >= 1.0:
                print(f"Heywood case found for item {items[comms_f.argmax()]} for the group factors of factor #{factor_n}! Communality: {comms_f.max()}")
            
            if "cronbach" in measures:
                fac_reliab.loc[factor_n, "cronbach"] = ra.alpha_cronbach
            if "omega_total" in measures:
                fac_reliab.loc[factor_n, "omega_total"] = ra.omega_total
            if "omega_hier" in measures:
                fac_reliab.loc[factor_n, "omega_hier"] = ra.omega_hierarchical
        
            if check_if_excluded:
                if len(items) > 3:
                    fac_reliab_excl[factor_n] = pd.DataFrame(index=items, columns=measures)
                    # loop over items for current factor
                    # compute reliability measures by excluding one item at a time
                    for cur_item in items:
                        # create list with all items except current item
                        items_wo_cur_item = [item for item in items if item != cur_item]

                        ra_excl = reliability_analysis(raw_dataset=df[items_wo_cur_item], is_corr_matrix=False, impute="median", n_factors_f=2)

                        # Also check fa_g and fa_f for Heywood case here
                        with warnings.catch_warnings(record=True) as w:
                            warnings.simplefilter("always")
                            ra_excl.fit()
                            if len(w) > 0:
                                comms_excl = ra_excl.fa_g.get_communalities()
                                if comms_excl.max() >= 1.0:
                                    print(
                                        f"Heywood case found while excluding {cur_item} from factor # {factor_n}! " \
                                        f"Heywood case for {items_wo_cur_item[comms_excl.argmax()]} for the common factor. " \
                                        f"Communality: {comms_excl.max()}"
                                    )
                                else:
                                    print(
                                        f"Warning while excluding {cur_item} from factor # {factor_n}! " \
                                        f"Error: {w[-1].message}"
                                    )

                        # Also check fa_f for Heywood case
                        comms_f_excl = ra_excl.fa_f.get_communalities()
                        if comms_f_excl.max() >= 1.0:
                            print(
                                f"Heywood case found while excluding {cur_item} from factor # {factor_n}! " \
                                f"Heywood case for {items_wo_cur_item[comms_f_excl.argmax()]} for the group factors. " \
                                f"Communality: {comms_f_excl.max()}"
                            )
                        
                        if "cronbach" in measures:
                            fac_reliab_excl[factor_n].loc[cur_item, "cronbach"] = ra_excl.alpha_cronbach
                        if "omega_total" in measures:
                            fac_reliab_excl[factor_n].loc[cur_item, "omega_total"] = ra_excl.omega_total
                        if "omega_hier" in measures:
                            fac_reliab_excl[factor_n].loc[cur_item, "omega_hier"] = ra_excl.omega_hierarchical
                else:
                    print(f"Factor {factor_n} only has 3 items. Excluding items is not recommended. Will not compute reliability for excluding single items.")

        elif len(items) == 2:
            print(f"Factor {factor_n} only has two items, will use Spearman-Brown instead.")
            # For 2-item scales, the Spearman-Brown Formula can be simplified (given r):
            # S_B = 2 * r / (1 + r)
            corr = df[items].corr().iloc[0, 1]
            spear_brown_rel = 2*corr/(1+corr)
            fac_reliab.loc[factor_n, "Spearman-Brown"] = spear_brown_rel
        else:
            print(f"Factor {factor_n} has only one item, cannot compute reliability.")

    # print results
    if print_results:
        print("\nInternal reliability for factors:")
        print(fac_reliab.astype(float).round(3))
        if check_if_excluded:
            for fac in fac_reliab_excl:
                print(f"\nInternal reliability for factor {fac} for excluding one item at a time:")
                print(fac_reliab_excl[fac].astype(float).round(3))

    if check_if_excluded and return_results:
        return fac_reliab, fac_reliab_excl
    elif return_results:
        return fac_reliab
import copy
import warnings
import numpy as np
import pandas as pd
import factor_analyzer as fa

# optional imports
try:
    from reliabilipy import reliability_analysis
except:
    pass

try:
    import matplotlib.pyplot as plt
except:
    pass

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
except:
    pass

# Function to reduce multicollinearity
def reduce_multicoll(df, vars_li, det_thre=0.00001, vars_descr=None, print_details=True):
    """
    Function to reduce multicollinearity in a dataset (intended for EFA).
    Uses the determinant of the correlation matrix to determine if multicollinearity is present.
    If the determinant is below a threshold (0.00001 by default),
    the function will drop the variable with the highest VIF until the determinant is above the threshold.

    Parameters:
    df (pandas dataframe): dataframe containing the variables to be checked for multicollinearity
    vars_li (list): list of variables to be checked for multicollinearity
    det_thre (float): Threshold for the determinant of the correlation matrix. Default is 0.00001. If the determinant is below this threshold, the function will drop the variable with the highest VIF until the determinant is above the threshold.
    vars_descr (list): Dataframe or dictonary containing the variable descriptions (variable names as index/key). If provided, the function will also print the variable descriptions additionally to the variable names.
    print_details (bool): If True, the function will print a detailed report of the process. Default is True.

    Returns:
    reduced_vars(list): List of variables after multicollinearity reduction, i.e. variables that are not highly correlated with each other.
    """
    reduced_vars = copy.deepcopy(vars_li)
    print("Beginning check for multicollinearity")
    vars_corr = df[reduced_vars].corr()
    det = np.linalg.det(vars_corr)
    print(f"\nDeterminant of initial correlation matrix: {det}\n")

    if det > det_thre:
        print(
            f"Determinant is > {det_thre}. No issues with multicollinearity detected.")
        return (reduced_vars)

    print("Starting to remove redundant variables by acessing mutlicollinearity with VIF...\n")
    count_missing = len(df) - len(df.dropna(subset=vars_li))
    if count_missing > 0:
        print(
            f"This requries dropping missing values."
            f"The following procedure will ignore {count_missing} cases with missing values"
        )
    while det <= det_thre:
        # could implement pairwise dropping of missing here at some point
        # but until you have a case with lots of missing data, this will work fine
        x_df = df.dropna(subset=vars_li)[reduced_vars]
        vifs = [vif(x_df.values, i)
                for i in range(len(x_df.columns))]
        vif_data = pd.Series(vifs, index=x_df.columns)
        vif_max = (vif_data.idxmax(), vif_data.max())

        if print_details:
            print(f"Excluded item {vif_max[0]}. VIF: {vif_max[1]:.2f}")

            if vars_descr is not None:
                print(f"('{vars_descr[vif_max[0]]}')")
            print("")

        reduced_vars.remove(vif_max[0])

        vars_corr = df[reduced_vars].corr()
        det = np.linalg.det(vars_corr)

    print(f"Done! Determinant is now: {det:.6f}")
    count_removed = len(vars_li) - len(reduced_vars)
    print(
        f"I have excluded {count_removed} redunant items with {len(reduced_vars)} items remaining")

    return (reduced_vars)

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
    vars_descr (pandas dataframe or dictonary): Dataframe or dictonary containing the variable descriptions (variable names as index/key). If provided, the function will also print the variable descriptions additionally to the variable names.

    Returns:
    kmo_per_variable (numpy.ndarray) – The KMO score per item.
    kmo_total (float) – The overall KMO score.
    """
    # drop missing values
    if dropna_thre > 0:
        df.dropna(subset=vars_li, thresh=dropna_thre, inplace=True)

    # calculate KMO
    kmo = fa.factor_analyzer.calculate_kmo(df[vars_li])

    print(f"Overall KMO: {kmo[1]}")

    if check_item_kmos:
        # Check KMO for each variable
        i = 0
        low_item_kmo = False
        for item_kmo in kmo[0]:
            if item_kmo < .6:
                low_item_kmo = True
                print(f"Low KMO for {vars_li[i]} : {item_kmo}")
                if vars_descr is not None:
                    print(f"('{vars_descr[vars_li[i]]}')")
            i += 1

        if low_item_kmo == False:
            print("All item KMOs are >.6")

    if return_kmos:
        return (kmo)

# Function to conduct parallel analysis
def parallel_analysis(
    df, vars_li, k=200, facs_to_display=15, print_graph=True,
    print_table=True, return_rec_n=True, extraction="minres",
    percentile=99, standard=1.1):
    """Function to perform parallel analysis on a dataset.

    Parameters:
    df (pandas dataframe): dataframe containing the variables to be analyzed
    vars_li (list): list of variables to be analyzed
    k (int): number of EFAs to fit over a random dataset for parallel analysis. Default is 200.
    facs_to_display (int): number of factors to display in table and/or graph
    print_graph (bool): whether to print a graph of the results. Requires matplotlib package. Default is True.
    print_table (bool): whether to print a table of the results
    return_rec_n (bool): whether to return the recommended number of factors
    extraction (str): extraction method to use for the EFA/PCA. Default is "minres". Other options are "ml" (maximum likelihood), "principal" (principal factors), and "components" (principal components).
    percentile (int): against which percentile to compare the eigenvalues. Default is 99.
    standard (float): how much higher the eigenvalues should be compared to the random dataset. Default is 1.1 (10% higher).

    Returns:
    suggested_factors: number of factors suggested by parallel analysis
    """
    # EFA with no rotation to get EVs
    if extraction == "components":
        efa = fa.FactorAnalyzer(rotation=None)
        efa.fit(df[vars_li])
        # Eigenvalues are orignal eigenvalues for PCA
        evs = efa.get_eigenvalues()[0]
    else:
        efa = fa.FactorAnalyzer(rotation=None, method=extraction)
        efa.fit(df[vars_li])
        # Eigenvalues are common factor eigenvalues for EFA
        evs = efa.get_eigenvalues()[1]

    # Determine size of original dataset for creation of random dataset
    n, m = df[vars_li].shape

    # Prepare FactorAnalyzer object
    if extraction == "components":
        par_efa = fa.FactorAnalyzer(rotation=None)
    else:
        par_efa = fa.FactorAnalyzer(rotation=None, method=extraction)

    # Create df to store the eigenvalues
    ev_par_df = pd.DataFrame(columns=range(1, m+1))

    # Run the fit 'k' times over a random matrix
    for i in range(0, k):
        par_efa.fit(np.random.normal(size=(n, m)))
        if extraction == "components":
            cur_ev_series = pd.Series(par_efa.get_eigenvalues()[
                                      0], index=range(1, m+1))
        else:
            cur_ev_series = pd.Series(par_efa.get_eigenvalues()[
                                      1], index=range(1, m+1))

        ev_par_df = pd.concat(
            [pd.DataFrame(cur_ev_series).transpose(), ev_par_df], ignore_index=True)
        ev_par_df = ev_par_df.apply(pd.to_numeric)

    # get percentile for the evs
    par_per = ev_par_df.quantile(percentile/100)

    if print_graph:
        # Draw graph
        plt.figure(figsize=(10, 6))

        # Line for eigenvalue 1
        plt.plot([1, facs_to_display+1], [1, 1], 'k--', alpha=0.3)
        # For the random data (parallel analysis)
        plt.plot(range(1, len(par_per.iloc[:facs_to_display])+1),
                 par_per.iloc[:facs_to_display], 'b', label=f'EVs - random: {percentile}th percentile', alpha=0.4)
        # Markers and line for actual EFA eigenvalues
        plt.scatter(
            range(1, len(evs[:facs_to_display])+1), evs[:facs_to_display])
        plt.plot(range(1, len(evs[:facs_to_display])+1),
                 evs[:facs_to_display], label='EVs - survey data')

        plt.title('Parallel Analysis Scree Plots', {'fontsize': 20})
        if extraction == "components":
            plt.xlabel('Components', {'fontsize': 15})
        else:
            plt.xlabel('Factors', {'fontsize': 15})
        plt.xticks(ticks=range(1, facs_to_display+1),
                   labels=range(1, facs_to_display+1))
        plt.ylabel('Eigenvalue', {'fontsize': 15})
        plt.legend()
        plt.show()

    # Determine threshold    
    # Also print out table with EVs if requested
    last_factor_n = 0
    last_per_par = 0
    last_ev_efa = 0
    found_threshold = False
    suggested_factors = 1

    if print_table:
        # Create simple table with values for 95th percentile for random data and EVs for actual data
        print(
            f"Factor eigenvalues for the {percentile}th percentile of {k} random matricesand for survey data for first {facs_to_display} factors:\n")
        print(f"\033[1mFactor\tEV - random data {percentile}th perc.\tEV survey data\033[0m")

    # Loop through EVs to find threshold
    # If requested also print table with EV for each number of factors
    # Always print the row for the previous (!) factor -
    # that way when we reach threshold the suggested number of factors can be made bold
    for factor_n, cur_ev_par in par_per.iloc[:facs_to_display].items():
        # factor_n start with 1, ev_efa is a list and index start at 0
        # so the respective ev from ev_efa is factor_n - 1
        cur_ev_efa = evs[factor_n-1]

        # If Threshold not found yet:
        # Check if for current number factors the (EV from random data x standard) is >= EV from actual data
        # If so, threshold has been crossed and the suggested number of factors is the previous step
        if (factor_n > 1) & (cur_ev_par*standard >= cur_ev_efa) & (found_threshold == False):
            found_threshold = True
            suggested_factors = factor_n-1

            # if requested print EV for previous factor - make it BOLD
            if print_table:
                print(
                    f"\033[1m{last_factor_n}\t{last_per_par:.2f}\t\t\t\t{last_ev_efa:.2f}\033[0m")
            # the rest of the loop is only needed for printing the table
            # so if no table is requested we can exit the loop here
            else:
                break

        # if requested and this is not the threshold step, print previous factor EV
        elif (factor_n > 1) & (print_table):
            print(f"{last_factor_n}\t{last_per_par:.2f}\t\t\t\t{last_ev_efa:.2f}")

        # if this is the last factor, also print the current factor EV if requested
        if (print_table) & (factor_n == len(par_per.iloc[:facs_to_display])):
            print(f"{factor_n}\t{cur_ev_par:.2f}\t\t\t\t{cur_ev_efa:.2f}")

        last_factor_n = factor_n
        last_per_par = cur_ev_par
        last_ev_efa = cur_ev_efa

    if print_table:
        print(
            f"Suggested number of factors \n"
            f"based on parallel analysis and standard of {standard}: {suggested_factors}")

    if return_rec_n:
        return suggested_factors

# Function to run iterative EFA
def iterative_efa(data, vars_analsis, n_facs=4, rotation_method="Oblimin",
                  comm_thresh=0.2, main_thresh=0.4, cross_thres=0.3, load_diff_thresh=0.2,
                  print_details=False, print_par_plot=False, print_par_table=False,
                  par_k=100, par_n_facs=15, iterative=True, auto_stop_par=False,
                  items_descr=None):
    """Run EFA with iterative process, eliminating variables with low communality, low main loadings or high cross loadings in a stepwise process.

    Parameters:
    data (pandas dataframe): Dataframe with data to be analyzed
    vars_analsis (list): List of variables to be analyzed
    n_facs (int): Number of factors to extract
    rotation_method (str): Rotation method to be used. Default is "Oblimin". Has to be one of the methods supported by the factor_analyzer package.
    comm_thresh (float): Threshold for communalities. Variables with communality below this threshold will be dropped from analysis.
    main_thresh (float): Threshold for main loadings. Variables with main loadings below this threshold will be dropped from analysis.
    cross_thres (float): Threshold for cross loadings. Variables with cross loadings above this threshold will be dropped from analysis.
    load_diff_thresh (float): Threshold for difference between main and cross loadings. Variables with difference between main and cross loadings below this threshold will be dropped from analysis.
    print_details (bool): If True, print details for each step of the iterative process.
    print_par_plot (bool): If True, print parallel analysis scree plot for each step of the iterative process.
    print_par_table (bool): If True, print table with eigenvalues from the parallel each step of the iterative process.
    par_k (int): Number of EFAs over a random matric for parallel analysis.
    par_n_facs (int): Number of factors to display for parallel analysis.
    iterative (bool): NOT IMPLEMENTED YET. If True, run iterative process. If False, run EFA with all variables.
    auto_stop_par (bool): If True, stop the iterative process when the suggested number of factors from parallel analysis is lower than the requested number of factors. In that case, no EFA object or list of variables is returned.
    items_descr (pandas series): Series with item descriptions. If provided, the function will print the item description for each variable that is dropped from the analysis.

    Returns:
    (efa, curr_vars): Tuple with EFA object and list of variables that were analyzed in the last step of the iterative process.
    """
    # Initialize FactorAnalyzer object
    efa = fa.FactorAnalyzer(n_factors=n_facs, rotation=rotation_method)

    # Marker to indicate whether the final solution was found
    final_solution = False

    # List of variables used for current factor solution
    curr_vars = copy.deepcopy(vars_analsis)

    # Loop until final solution is found
    i = 1
    while final_solution == False:
        # Fit EFA
        efa.fit(data[curr_vars])
        print(f"Fitted solution #{i}\n")

        # print screeplot and/or table and/or check for auto-stopping for parallel analysis
        # (if respective option was chosen)
        if print_par_plot or print_par_table or auto_stop_par:
            suggested_n_facs = parallel_analysis(
                data, curr_vars, k=par_k, facs_to_display=par_n_facs,
                print_graph=print_par_plot, print_table=print_par_table)

            if (suggested_n_facs < n_facs) & auto_stop_par:
                print("\nAuto-Stop based on parallel analysis: "
                      f"Parallel analysis suggests {suggested_n_facs} factors. "
                      f"That is less than the currently requested number of factors ({n_facs})."
                      "Iterative Efa stopped. No EFA object or list of variables will be returned.")
                return

        # Check 1: Check communcalities
        print("\nChecking for low communalities")
        comms = pd.DataFrame(
            efa.get_communalities(),
            index=data[curr_vars].columns,
            columns=['Communality']
        )
        mask_low_comms = comms["Communality"] < comm_thresh

        if comms[mask_low_comms].empty:
            print(f"All communalities above {comm_thresh}\n")
        else:
            # save bad items and remove them
            bad_items = comms[mask_low_comms].index
            print(
                f"Detected {len(bad_items)} items with low communality. Excluding them for next analysis.\n")
            for item in bad_items:
                if print_details:
                    print(f"\nRemoved item {item}\nCommunality: {comms.loc[item, 'Communality']:.4f}\n")
                    if items_descr:
                        print(f"Item description: {items_descr[item]}\n")
                curr_vars.remove(item)
            i += 1
            continue

        # Check 2: Check for low main loading
        print("Checking for low main loading")
        loadings = pd.DataFrame(efa.loadings_, index=data[curr_vars].columns)
        max_loadings = abs(loadings).max(axis=1)
        mask_low_main = max_loadings < main_thresh
        if max_loadings[mask_low_main].empty:
            print(f"All main loadings above {main_thresh}\n")
        else:
            # save bad items and remove them
            bad_items = max_loadings[mask_low_main].index
            print(
                f"Detected {len(bad_items)} items with low main loading. Excluding them for next analysis.\n")
            for item in bad_items:
                if print_details:
                    print(f"\nRemoved item {item}\nMain (absolute) Loading: {abs(loadings.loc[item]).max():.4f}\n")
                    if items_descr:
                        print(f"Item description: {items_descr[item]}\n")
                curr_vars.remove(item)
            i += 1
            continue

        # check 3: Check for high cross loadings
        print("Checking high cross loadings")

        # create df that stores main_load, largest crossload and difference between the two
        crossloads_df = pd.DataFrame(index=curr_vars)

        crossloads_df["main_load"] = abs(loadings).max(axis=1)
        crossloads_df["cross_load"] = abs(loadings).apply(
            lambda row: row.nlargest(2).values[-1], axis=1)
        crossloads_df["diff"] = crossloads_df["main_load"] - \
            crossloads_df["cross_load"]

        mask_high_cross = (crossloads_df["cross_load"] > cross_thres) | (
            crossloads_df["diff"] < load_diff_thresh)

        if crossloads_df[mask_high_cross].empty:
            print(
                f"All cross-loadings below {cross_thres}"
                f" and differences between main loading and crossloadings above {load_diff_thresh}.\n"
            )
        else:
            # save bad items and remove them
            bad_items = crossloads_df[mask_high_cross].index
            print(
                f"Detected {len(bad_items)} items with high cross loading. Excluding them for next analysis.\n")
            for item in bad_items:
                if print_details:
                    print(f"Removed item {item}\nLoadings: \n{loadings.loc[item]}\n")
                    if items_descr:
                        print(f"Item description: {items_descr[item]}\n")
                curr_vars.remove(item)
            i += 1
            continue

        print("Final solution reached.")
        final_solution = True

        corrs = data[curr_vars].corr()
        det = np.linalg.det(corrs)
        print(f"\nDeterminant of correlation matrix: {det}")
        if det > 0.00001:
            print("Determinant looks good!")
        else:
            print("Determinant is smaller than 0.00001!")
            print(
                "Consider using stricer criteria and/or removing highly correlated vars")

        kmo_check(data[curr_vars], curr_vars, dropna_thre=0, check_item_kmos=True, return_kmos=False, vars_descr=items_descr)

        # Check for Heywood cases
        comms = efa.get_communalities()
        if comms.max() >= 1.0:
            print(f"Heywood case found for item {items[comms.argmax()]}. Communality: {comms.max()}")
        else:
            print("No Heywood case found.")

    return (efa, curr_vars)

# Function to print main loadings for each factor
def print_sorted_loadings(efa, item_lables, load_thresh=0.4, descr=[]):
    """Print strongly loading variables for each factor. Will only print loadings above load_thresh for each factor.

    Parameters:
    efa (object): EFA object. Has to be created with factor_analyzer package.
    item_lables (list): List of item labels
    load_thresh (float): Threshold for main loadings. Only loadings above this threshold will be printed for each factor.
    descr (list): List of item descriptions. If empty, no item description will be printed.

    Returns:
    None
    """

    loadings = pd.DataFrame(efa.loadings_, index=item_lables)
    n_load = loadings.shape[1]

    if len(descr) > 0:
        loadings["descr"] = loadings.apply(lambda x: descr[x.name], axis=1)

    for i in range(0, n_load):
        mask_relev_loads = abs(loadings[i]) > load_thresh
        sorted_loads = loadings[mask_relev_loads].sort_values(
            i, key=abs, ascending=False)
        print(f"Relevant loadings for factor {i}")
        if len(descr) > 0:
            print(sorted_loads[[i, "descr"]].to_string(), "\n")
        else:
            print(sorted_loads[i].to_string(), "\n")

# Function to automatically reverse-code (Likert-scale) items where necessary
def rev_items_and_return(df, efa, item_lables, load_thresh=0.4, min_score=1, max_score=5):
    """Takes an EFA object and automatically reverse-codes (Likert-scale) items where necessary
    and adds the reverse-coded version to a new dataframe.
    Will only reverse-code items with main loadings above load_thresh for each factor.

    Parameters:
    df (pandas dataframe): Dataframe containing items to be reverse-coded
    efa (object): EFA object. Has to be created with factor_analyzer package.
    item_lables (list): List of item labels
    load_thresh (float): Threshold for main loadings. Only loadings above this threshold will be reverse-coded for each factor.
    min_score (int): Minimum possible score for items
    max_score (int): Maximum possible score for items

    Returns:
    (new_df, items_per_fact_dict): Tuple containing new dataframe with reverse-coded items and dictionary with a list of items per factor
    """
    new_df = df.copy()
    loadings = pd.DataFrame(efa.loadings_, index=item_lables)
    n_load = loadings.shape[1]

    items_per_fact_dict = {}

    # loop through n factors
    # determine relevant items that are positive (can just be used as is)
    # and items with negative loads (need to be refersed)
    for i in range(0, n_load):
        mask_pos_loads = loadings[i] > load_thresh
        mask_neg_loads = loadings[i] < -load_thresh
        pos_items = loadings[mask_pos_loads].index.tolist()
        neg_items = loadings[mask_neg_loads].index.tolist()

        # add items with positive items directly to dict
        items_per_fact_dict[i] = pos_items

        # create reverse-coded item in new_df for items with negative loadings
        for item in neg_items:
            rev_item_name = item + "_rev"
            new_df[rev_item_name] = (new_df[item] - (max_score+min_score)) * -1
            items_per_fact_dict[i].append(rev_item_name)

    return (new_df, items_per_fact_dict)

def factor_int_reliability(df, items_per_factor, measures = ["cronbach", "omega_total", "omega_hier"], check_if_excluded = True, print_results = True, return_results = True):
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
    fac_reliab_excl(dict): Dictionary with reliability estimates for each factor when each item is excluded. Keys are factor numbers, values are dataframes with reliability estimates. Each row gives reliability estimates fore excluding one item from that factor.
    """

    # Create df to store measures for whole factors
    fac_reliab = pd.DataFrame(index=items_per_factor.keys(), columns=measures)
    # dict to store dfs to store measures for each item excluded
    if check_if_excluded:
        fac_reliab_excl = {}
    
    # Loop over factors
    for factor_n in items_per_factor:

        items = items_per_factor[factor_n]

        if len(items) > 2:
            
            ra = reliability_analysis(raw_dataset=df[items], is_corr_matrix=False, impute="median")

            # Check for Heywood case
            # relaiabilipy runs into trouble, when fa_g is a Heywood case
            # Will catch Warning and warn the user about the Heywood case
            # In general good idea to check for Heywood case though
            # will also check for Heywood case for fa_f and warn if there is one
            
            with warnings.catch_warnings(), np.errstate(invalid="warn"):
                warnings.filterwarnings('error', category=RuntimeWarning, message="invalid value encountered in double_scalars")
                try:
                    ra.fit()
                except Warning as err:
                    with np.errstate(invalid='ignore'):
                        ra.fit()
                        comms_g = ra.fa_g.get_communalities()
                        if comms_g.max() >= 1.0:
                            print(f"Heywood case found for item {items[comms_g.argmax()]} for the common factor of factor #{factor_n}! Communality: {comms_g.max()}")
                        else:
                            print(f"Warning for factor {factor_n}! Error: {err}")

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
                    fac_reliab_excl[factor_n] = pd.DataFrame(index=items_per_factor[factor_n], columns=measures)
                    # loop over items for current factor
                    # compute cronbach's alpha by excluding one item at a time
                    for cur_item in items:
                        # create list with all items except current item
                        items_wo_cur_item = copy.deepcopy(items)
                        items_wo_cur_item.remove(cur_item)

                        ra_excl = reliability_analysis(raw_dataset=df[items_wo_cur_item], is_corr_matrix=False, impute="median", n_factors_f=2)

                        # Also check fa_g and fa_f for Heywood case here
                        with warnings.catch_warnings(), np.errstate(invalid="warn"):
                            warnings.filterwarnings('error', category=RuntimeWarning, message="invalid value encountered in double_scalars")
                            try:
                                ra_excl.fit()
                            except Warning as err:
                                with np.errstate(invalid='ignore'):
                                    ra_excl.fit()
                                    comms_excl = ra_excl.fa_g.get_communalities()
                                    if comms_excl.max() >= 1.0:
                                        print(
                                            f"Heywood case found while excluding {cur_item} from factor # {factor_n}! " \
                                            f"Heywood case for {items_wo_cur_item[comms_excl.argmax()]} for the common factor. " \
                                            f"Communality: {comms_excl.max()}"
                                            )
                                    else:
                                        print(
                                            f"Warning while while excluding {cur_item} from factor # {factor_n}! " \
                                            f"Error: {err}"
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
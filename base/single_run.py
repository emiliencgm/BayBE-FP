import os
from base.benchmarking import load_data, create_campaign, create_search_space, create_recommender, generate_FP
import pandas as pd
from baybe.simulation import simulate_scenarios
from baybe.utils.random import set_random_seed
import numpy as np
import copy

def run_single_combination(args, param_dict, output_dir=None):
    """
    args: dataset, switch_after, mc_runs, n_iter, batch_size, seed, filepath, impute_mode.\n
    
    param_dict: FP_TYPES, KERNEL_PRIOR, THRESHOLD, ACQ_FUNC. INIT_METHOD, KERNEL_NAME\n
    
    output_dir = os.path.join("output", f"{args.dataset}_mc{args.mc_runs}_niter{args.n_iter}_batch{args.batch_size}_seed{args.seed}")\n
    
    file_prefix/scenario = "__".join([f"{k}_{v}" for k, v in param_dict.items()])
    """
    if output_dir is None:
        output_dir = os.path.join(
            "output",
            f"{args.dataset}_switch{args.switch_after}_mc{args.mc_runs}_niter{args.n_iter}_batch{args.batch_size}_seed{args.seed}"
        )
    os.makedirs(output_dir, exist_ok=True)
    
    file_prefix = "__".join([f"{k}_{v}" for k, v in param_dict.items()])
    campaign_name = "__".join([str(v) for k, v in param_dict.items()])
    file_path = os.path.join(output_dir, f"{file_prefix}.pkl")

    if os.path.exists(file_path):
        print(f"Skip {file_prefix}, already exists")
        record_path = os.path.join(output_dir, "all_records.pkl")
        df = pd.read_pickle(record_path)
        row = df.copy()
        for k, v in param_dict.items():
            row = row[row[k] == v]
        SCORE_AUC = row['score_AUC'].values[0]
        existing = True
        return file_path, SCORE_AUC, existing

    print(f"Run: {file_prefix}")

    data = load_data(dataset=args.dataset)
    lookup = data['lookup']
    F_BEST = data['F_BEST']
    objective = data['objective']
    numerical_params = data['numerical_params']
    discrete_data = data['discrete_data']

    fp_type = param_dict["FP_TYPES"]
    kernel_prior = param_dict["KERNEL_PRIOR"]
    threshold = param_dict["THRESHOLD"]
    acq_func = param_dict["ACQ_FUNC"]
    init_method = param_dict["INIT_METHOD"]
    kernel_name = param_dict["KERNEL_NAME"]

    PCA = threshold == "PCA"
    decorr_threshold = 0.0 if PCA else float(threshold)

    FP = generate_FP(discrete_data, fp_type=fp_type, PCA=PCA, decorr_threshold=decorr_threshold)
    
    searchspace = create_search_space(search_params_dict=FP, numeric_params_dict=numerical_params)
    
    feat_dim = len(searchspace.comp_rep_columns)

    recommender = create_recommender(kernel_prior=kernel_prior, switch_after=args.switch_after, acq_func=acq_func, searchspace=searchspace, init_method=init_method, feat_dim=feat_dim, kernel_name=kernel_name)
    
    # TODO to get model hyperparameters: surrogate_model._model.covar_module.lengthscale, see BoTorch.

    campaign = create_campaign(searchspace=searchspace, objective=objective, recommender=recommender)

    set_random_seed(args.seed)

    result = simulate_scenarios(
        {campaign_name: campaign},
        lookup,
        batch_size=args.batch_size,
        n_doe_iterations=args.n_iter,
        n_mc_iterations=args.mc_runs,
        impute_mode=args.impute_mode
    )

    for k, v in param_dict.items():
        result[k] = v

    result.to_pickle(file_path)
    
    SCORE_AUC = record_single_run(output_dir, param_dict, result, lookup, feat_dim)
    
    existing = False
    
    return file_path, SCORE_AUC, existing

def record_single_run(output_dir, param_dict, df_result, lookup, feat_dim):
    '''
    return: score_AUC
    '''
    csv_path = os.path.join(output_dir, 'all_records.csv')
    pkl_path = os.path.join(output_dir, 'all_records.pkl')
    
    if os.path.exists(pkl_path):
        df = pd.read_pickle(pkl_path)
    else:
        df = pd.DataFrame()
        
    F_BEST = lookup['yield'].max()
    
    top_perc_list = [0.01, 0.05, 0.10]
    score_top_coverage_list = top_coverage(lookup, df_result, top_perc_list)
    
    target_x_list = [20, 30, 40]
    target_y_list = [80, 85, 90]
    score_AUC, score_x_list, score_y_list = score_single_run(df_result, target_x_list=target_x_list, target_y_list=target_y_list, F_BEST=F_BEST)
    
    row_dict = copy.deepcopy(param_dict)
    
    row_dict['f_dim'] = feat_dim
    
    for i, top_perc in enumerate(top_perc_list):
        row_dict[f'top_{top_perc}_coverage'] = round(score_top_coverage_list[i], 3)
    
    row_dict['score_AUC'] = round(score_AUC, 3)
    for i, target_x in enumerate(target_x_list):
        row_dict[f'score_x={target_x}'] = round(score_x_list[i], 3)
    for i, target_y in enumerate(target_y_list):
        row_dict[f'score_y={target_y}'] = round(score_y_list[i], 3)
    
    new_row = pd.DataFrame([row_dict])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)
    
    return score_AUC
    

def score_single_run(df, metric_y='yield_CumBest', metric_x='Num_Experiments', target_x_list=None, target_y_list=None, std_penalty_coeff=0.0, F_BEST=100):
    """
    Convert curve into a scalar score.
    Score is normalized according to max(metric_x) and F_BEST (default=100 for yield_CumBest).
    """

    # mean-agg over all Monte Carlo runs or repetitions
    grouped = df.groupby(metric_x)[metric_y]
    xs = grouped.mean().index.values.astype(int)
    ys = grouped.mean().values
    stds = grouped.std().fillna(0.0).values

    # Ensure sorted by x
    idx_sorted = np.argsort(xs)
    xs, ys, stds = xs[idx_sorted], ys[idx_sorted], stds[idx_sorted]

    # (1) AUC scoring
    try:
        auc = np.trapz(ys, xs)
    except: # Numpy >= 2.0
        auc = np.trapezoid(ys, xs)
    # incorporate stability (negative penalty)
    penalty_area = std_penalty_coeff * np.mean(stds) * len(xs)
    SCORE_AUC =  (auc - penalty_area) / (F_BEST * xs[-1]) # higher is better

    # (2) Score by y value at target_x
    SCORE_X = []
    if target_x_list is None:
        raise ValueError("target_x must be provided for method='x'")
    for target_x in target_x_list:
        if target_x not in xs:
            # simply return the last y
            SCORE_X.append( (ys[-1] - std_penalty_coeff * stds[-1]) / F_BEST )
        else:
            i = np.where(xs == target_x)[0][0]
            SCORE_X.append( (ys[i] - std_penalty_coeff * stds[i]) / F_BEST )  # higher is better

    # (3) Score by x value where reaching target_y
    SCORE_Y = []
    if target_y_list is None:
        raise ValueError("target_y must be provided for method='y'")
    for target_y in target_y_list:
        idx = np.where(ys >= (target_y + std_penalty_coeff*np.mean(stds)))[0]
        if len(idx) > 0:
            i = idx[0]
            SCORE_Y.append( xs[i] / xs[-1] )  # smaller is better
        else:
            # never reach
            SCORE_Y.append( 1.0 )

    return SCORE_AUC, SCORE_X, SCORE_Y

def top_coverage(lookup, results, top_perc_list=None, x:int=None):
    '''
    top k% yield coverage during all Monte-Carlo runs.
    
    x: if given, truncated at x
    '''
    if x:
        results = results[results['Num_Experiments'] <= x].copy()
    SCORE_COVERAGE = []
    for top_perc in top_perc_list:
        lookup_len = len(lookup["yield"]) # search space
        
        top_k_perc = np.min(np.sort( lookup["yield"] )[ -int(top_perc*lookup_len): ]) # top-k% yield ==> threshold

        sumbest = np.sum( results['yield_IterBest']  >= top_k_perc ) # number of experiments above top-k% yield during MC_RUNS.

        # coverage of top-k% findings during all MC_RUNS
        SCORE_COVERAGE.append( sumbest / len(results) )
    return SCORE_COVERAGE
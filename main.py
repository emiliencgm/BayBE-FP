import os
import itertools
from base.single_run import run_single_combination
import pandas as pd

def is_valid_combination(param_dict, args):
    fp_type = param_dict.get("FP_TYPES")
    threshold = param_dict.get("THRESHOLD")
    kernel_name = param_dict.get("KERNEL_NAME")
    kernel_prior = param_dict.get("KERNEL_PRIOR")

    if fp_type in ["one_hot", 'OHE'] and threshold != 0.7:
        return False
    
    if fp_type in ['drfp', 'DRFP'] and not args.dataset in ['ni_catalyzed_1', 'ni_catalyzed_2', 'ni_catalyzed_3', 'ni_catalyzed_4']:
        return False
    
    if kernel_name in ['RBF', 'rbf'] and not kernel_prior in ["LogNormal_DSP", "SBO", 'adaptive_emilien']:
        return False

    return True

def run_high_throughput(args, param_grid):
    
    output_dir = os.path.join(
        "output",
        f"{args.dataset}_switch{args.switch_after}_mc{args.mc_runs}_niter{args.n_iter}_batch{args.batch_size}_seed{args.seed}"
    )
    os.makedirs(output_dir, exist_ok=True)

    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    valid_combinations = [comb for comb in all_combinations if is_valid_combination(comb, args)]
    print(f"{len(valid_combinations)}/{len(all_combinations)} combinations are valid.")

    results_path_pkl = os.path.join(output_dir, "all_results.pkl")
    results_path_csv = os.path.join(output_dir, "all_results.csv")

    new_dfs = []

    for param_dict in valid_combinations:
        file_path, score_auc, existing = run_single_combination(args, param_dict, output_dir=output_dir)
        if (not existing) and (file_path is not None) and os.path.exists(file_path):
            df = pd.read_pickle(file_path)
            new_dfs.append(df)

    if not new_dfs:
        print("No valid new results generated.")
        if os.path.exists(results_path_pkl):
            df_existing = pd.read_pickle(results_path_pkl)
            print(f"Return the existing results: {results_path_pkl}.")
            return df_existing
        else:
            return None

    df_new = pd.concat(new_dfs, ignore_index=False)

    if os.path.exists(results_path_pkl):
        df_existing = pd.read_pickle(results_path_pkl)
        df_all = pd.concat([df_existing, df_new], ignore_index=False)
        print(f"Loaded existing results: {results_path_pkl}")
    else:
        df_all = df_new
        print("No existing results found. Creating new results file.")

    df_all.to_pickle(results_path_pkl)
    df_all.to_csv(results_path_csv, index=False)
    print(f"Updated all results -> {results_path_pkl}")

    return df_all
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # Optimization loops:
    parser.add_argument("--impute_mode", type=str, default='error', help='[ignore] for missing combination.')
    parser.add_argument("--mc_runs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_iter", type=int, default=60)
    parser.add_argument("--switch_after", type=int, default=5)
    
    # Benchmarks:
    parser.add_argument("--dataset", type=str, default='shields', help='[shields, buchwald_hartwig, ni_catalyzed_1, ni_catalyzed_2, ni_catalyzed_3, ni_catalyzed_4, suzuki]')
    parser.add_argument("--seed", type=int, default=1337)
    
    args = parser.parse_args()
    
    PARAM_GRID_FULL = {
        "FP_TYPES": ['one_hot', 'mordred', 'chemeleon', 'chemberta_small', 'chemberta_large', 't5-base-chem', 't5-base', 'UAE-Large-V1', 'drfp'],
        "KERNEL_PRIOR": ['BayBE8D', 'BayBE75D', 'EDBO+', 'EDBO_MORDRED', 'EDBO_OHE', 'max_custom_0', 'BayBE_adaptive', 'LogNormal_DSP', 'SBO', 'adaptive_emilien'],
        "THRESHOLD": ['PCA', 0.7, 0.9],
        "ACQ_FUNC": ['qLogEI'],
        "INIT_METHOD": ['random'],
        "KERNEL_NAME": ['Matern', 'RBF'],
    }
    
    param_grid = {
        "FP_TYPES": ['mordred'],
        "KERNEL_PRIOR": ['adaptive_emilien'],
        "THRESHOLD": ['PCA'],
        "ACQ_FUNC": ['qLogEI'],
        "INIT_METHOD": ['random'],
        "KERNEL_NAME": ['Matern'],
    }
    
    df_all = run_high_throughput(args, param_grid)
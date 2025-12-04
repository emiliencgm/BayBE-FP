import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import copy
import numpy as np
    
PLOTARGS = {
    'linestyle': 'solid',
    'marker': 'o',
    'markersize': 6,
    'markeredgecolor': 'none'
}
FIGSIZE = (11,6)

def plot_subplots_by_two_factors(
    results='./all_results.pkl',
    factor_x="THRESHOLD",
    factor_y="FP_TYPES",
    color_factor="KERNEL_PRIOR",
    metric="yield_CumBest",
    F_BEST=100,
    dataset=''
):
    '''
    If y=_fp_type, x=_threshold, and color=_kernel_prior, then we plot the average/shift for all Monte_Carlo_Run [, _switch_after, _acqui_func].
    '''

    tmp = results.reset_index()
    unique_x = sorted(tmp[factor_x].unique(), key=str)
    unique_y = sorted(tmp[factor_y].unique(), key=str)

    ncols = len(unique_x)
    nrows = len(unique_y)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6*ncols, 5*nrows),
        sharey=True
    )
    axes = np.array(axes).reshape(nrows, ncols)

    legend_handles = None
    legend_labels = None

    for i, fy in enumerate(unique_y):
        for j, fx in enumerate(unique_x):

            ax = axes[i, j]
            sub_df = tmp[(tmp[factor_y] == fy) &
                        (tmp[factor_x] == fx)]
            if sub_df.empty:
                ax.set_visible(False)
                continue

            for kp in sorted(sub_df[color_factor].unique(), key=str):
                df_kp = sub_df[sub_df[color_factor] == kp]
                plotargs = copy.deepcopy(PLOTARGS)
                plotargs['markersize'] = 4

                line = sns.lineplot(
                    data=df_kp,
                    x="Num_Experiments",
                    y=metric,
                    label=kp,
                    ax=ax,
                    **plotargs,
                    legend=True,
                )

            ax.axhline(y=F_BEST, color='red', linestyle='--')
            ax.set_ylim(0, F_BEST+0.05*F_BEST)

            ax.set_title(f"{factor_y}={fy}, {factor_x}={fx}")
            if j == 0:
                ax.set_ylabel(metric)
            else:
                ax.set_ylabel("")
            if i == nrows-1:
                ax.set_xlabel("Num_Experiments")
            else:
                ax.set_xlabel("")

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

    # global legend
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=len(legend_labels),
            bbox_to_anchor=(0.5, 1.02),
            fontsize=12
        )

    plt.suptitle(f"Display by Three Factors: {dataset}", fontsize=16, y=1.04)
    plt.tight_layout()
    plt.savefig(f'SubPlots_{dataset}.pdf', dpi=1500, bbox_inches="tight")
    plt.close()
    
def plot_param_heatmap(df, col_x, col_y, metric, dataset=None):
    """
    df: from all_records.pkl
    """

    grouped = df.groupby([col_x, col_y])[metric].agg(
        Mean="mean",
        Std="std"
    ).reset_index()

    grouped["text"] = grouped.apply(
        lambda r: f"{r['Mean']:.2f}±{r['Std']:.2f}", axis=1
    )

    pivot_mean = grouped.pivot(index=col_y, columns=col_x, values="Mean")
    pivot_text = grouped.pivot(index=col_y, columns=col_x, values="text")

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        pivot_mean,
        annot=pivot_text,
        fmt="",
        cmap="YlGnBu",
        cbar_kws={'label': metric}
    )

    ax.set_title(f"Effect of {col_x} & {col_y} on {metric}")
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)

    plt.tight_layout()

    fig_name = f"heatmap_{dataset}_{col_x}_{col_y}_{metric}.pdf"
    plt.savefig(fig_name, dpi=600)
    plt.close()

    print(f"Saved heatmap -> {fig_name}")
    return grouped

def plot_kernel_prior_2d_scatter(pkl_file: str, dataset: str, metric='score_AUC', d_scale='sqrt', a=None, b=None, c=None):
    '''
    pkl_file: all_records.pkl
    '''
    lookup = {"BayBE8D": 1.1, "BayBE75D": 4.5, "EDBO+": 10.0, "EDBO_MORDRED": 20.0, "EDBO_OHE":3.0, "max_custom_0": 40.0}
    function_lookup = {
    "BayBE_adaptive": lambda f: np.interp(f, (8, 75), [1.2, 2.5]) / np.interp(f, (8, 75), [1.1, 0.55]),
    "adaptive_emilien": lambda f: 0.4 * np.sqrt(f) + 4.0,
    "SBO": lambda f: 1.0 * np.sqrt(f),
    # "LogNormal_DSP": lambda f: 18.17*np.sqrt(f),
    }

    
    df = pd.read_pickle(pkl_file)
    
    required_cols = {"KERNEL_PRIOR", "f_dim", metric}
    if not required_cols.issubset(df.columns):
        raise KeyError(f"missing: {required_cols - set(df.columns)}")
    
    df["f_dim_sqrt"] = np.sqrt(df["f_dim"])
    df["f_dim_log"] = np.log(df["f_dim"])
    
    # constant mapping
    df["KERNEL_PRIOR_FLOAT"] = df["KERNEL_PRIOR"].map(lookup)

    # function mapping
    if function_lookup is not None:
        for key, func in function_lookup.items():
            mask = df["KERNEL_PRIOR"] == key
            if mask.any():
                df.loc[mask, "KERNEL_PRIOR_FLOAT"] = func(df.loc[mask, "f_dim"].values)

    unmapped = df[df["KERNEL_PRIOR_FLOAT"].isnull()]
    if not unmapped.empty:
        print("unmapped values:", unmapped["KERNEL_PRIOR"].unique())
    
    y = df["KERNEL_PRIOR_FLOAT"].values
    if d_scale == 'sqrt':
        x = df["f_dim_sqrt"].values
    elif d_scale == 'log':
        x = df["f_dim_log"].values
    elif d_scale == 'x':
        x = df["f_dim"].values
    z = df[metric].values
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scatter = ax.scatter(x, y, c=z, cmap='inferno', s=80, edgecolor='k')
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(metric)
    
    if c is None: # y = ax + b
        if a is not None and b is not None:
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = a * x_line + b
            ax.plot(x_line, y_line, color='blue', linewidth=2, label=f'y={a}x+{b}')
            ax.legend()
    else: # y = ax^2 + bx + c
        if a is not None and b is not None:
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = a * x_line**2 + b * x_line + c
            ax.plot(x_line, y_line, color='green', linewidth=2, label=f'y={a}x^2+{b}x+{c}')
            ax.legend()
    
    ax.set_ylabel("KERNEL_PRIOR (float)")
    ax.set_xlabel(f"f_dim_{d_scale}")
    ax.set_title(f"2D Scatter Plot Colored by {metric} — {dataset}")
    
    plt.tight_layout()
    plt.savefig(f'dim_lengthscale_{dataset}_{metric}_{d_scale}.pdf', dpi=600)
    plt.close()

if __name__ == "__main__":
    pass
    # plot_subplots_by_two_factors(results=pd.read_pickle('/Users/guanmingchen/Desktop/BO/BayBE-FP/output/shields_switch5_mc2_niter30_batch1_seed1337/all_results.pkl'), dataset='shields')
    
    # plot_param_heatmap(df=pd.read_pickle('/Users/guanmingchen/Desktop/BO/BayBE-FP/output/shields_switch5_mc2_niter30_batch1_seed1337/all_records.pkl'), col_x='FP_TYPES', col_y='KERNEL_PRIOR', metric='score_AUC', dataset='shields')

    for dataset in ['shields', 'ni_catalyzed_2']:
        for metric in ['score_AUC', 'top_0.05_coverage']:
            plot_kernel_prior_2d_scatter(pkl_file=f'/Users/guanmingchen/Desktop/BO/BayBE-FP/output/{dataset}_switch5_mc2_niter30_batch1_seed1337/all_records.pkl', dataset=dataset, metric=metric, a=0.4, b=4.0,  d_scale='sqrt')
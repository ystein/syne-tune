# %%
import logging
from argparse import ArgumentParser
from benchmarking.nursery.benchmark_kdd.baselines import Methods
from datetime import datetime

from benchmarking.nursery.benchmark_kdd.results_analysis.utils import method_styles, load_and_cache, plot_results, \
    print_rank_table

if __name__ == '__main__':
    date_min = datetime.fromisoformat("2022-01-04")
    date_max = datetime.fromisoformat("2023-01-04")

    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_tag", type=str, required=False, default="purple-akita",
        help="the experiment tag that was displayed when running the experiment"
    )
    args, _ = parser.parse_known_args()
    experiment_tag = args.experiment_tag
    logging.getLogger().setLevel(logging.INFO)

    load_cache_if_exists = True

    # benchmarks_to_df = {bench: df[] for bench, df in benchmarks_to_df.items()}
    methods_to_show = list(method_styles.keys())
    print(methods_to_show)
    benchmarks_to_df = load_and_cache(load_cache_if_exists=load_cache_if_exists, experiment_tag=experiment_tag, methods_to_show=methods_to_show)

    for bench, df_ in benchmarks_to_df.items():
        df_methods = df_.algorithm.unique()
        for x in methods_to_show:
            if x not in df_methods:
                logging.warning(f"method {x} not found in {bench}")

    for benchmark in ['fcnet', 'nas201']:
        n = 0
        for key, df in benchmarks_to_df.items():
            if benchmark in key:
                n += len(df[df.algorithm == Methods.RS])
        print(f"number of hyperband evaluations for {benchmark}: {n}")

    methods_to_show = [
        Methods.RS,
        Methods.TPE,
        Methods.REA,
        Methods.GP,
        Methods.MSR,
        Methods.ASHA,
        Methods.BOHB,
        #Methods.MOBSTER,
        Methods.ZEROSHOT,
        Methods.ASHA_BB,
        Methods.ASHA_CTS,
        Methods.SGPT,
        Methods.RUSH,
        Methods.RUSH5
    ]
    print_rank_table(benchmarks_to_df, methods_to_show)

    plot_results(benchmarks_to_df, method_styles)

    import numpy as np
    import matplotlib.pyplot as plt


    def delatexfy(string):
        rename_dict = {
            'RSMSR': 'RS-MSR',
            'ASHABB': 'ASHA-BB',
            'ASHACTS': 'ASHA-CTS'
        }
        delatexfied = string.replace("\\", "").replace("{", "").replace("}", "")
        return rename_dict.get(delatexfied, delatexfied)

    max_time = {task_name: task_df['st_tuner_time'].max() for task_name, task_df in benchmarks_to_df.items()}
    results = list()
    for time_fraction in np.arange(1, 0, -0.1):
        for task_name, task_df in benchmarks_to_df.items():
            benchmarks_to_df[task_name] = task_df[task_df['st_tuner_time'] < time_fraction * max_time[task_name]]
        results.append(print_rank_table(benchmarks_to_df, methods_to_show))
    ranking = dict()
    for result in reversed(results):
        for method, benchmark_results in result.to_dict().items():
            method = delatexfy(method)
            for benchmark, rank in benchmark_results.items():
                benchmark = delatexfy(benchmark)
                if benchmark not in ranking:
                    ranking[benchmark] = dict()
                if method not in ranking[benchmark]:
                    ranking[benchmark][method] = list()
                ranking[benchmark][method].append(rank)
    for benchmark, method_ranking in ranking.items():
        plt.figure(facecolor='w')
        for method, avg_rank in method_ranking.items():
            method_style = method_styles[method]
            plt.plot(np.arange(0.1, 1.1, 0.1), avg_rank, label=method, color=method_style.color,
                     linestyle=method_style.linestyle, marker=method_style.marker,)
        plt.title(benchmark)
        plt.xlabel('Fraction of Total Wallclock Time')
        plt.ylabel('Normalized Average Rank')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"figures/rank-{benchmark}.pdf")
        plt.close()

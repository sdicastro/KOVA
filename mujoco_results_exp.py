import numpy as np
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from glob import glob
import os.path as osp
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')  # Can change to 'Agg' for non-interactive mode
plt.rcParams['svg.fonttype'] = 'none'

EXT = "preogress.csv"
X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 4
COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'magenta', 'lavender', 'cyan', 'yellow', 'black', 'pink',
          'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'turquoise', 'darkgreen', 'tan', 'salmon',
          'gold', 'lightpurple', 'darkred', 'darkblue']


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def window_func(x, y, window, func, alg, env):
    yw = rolling_window(y, window)
    yw_func = func(yw, axis=-1)
    x = x[window - 1:]
    return x, yw_func


def xtimesteps_by_alg(alg):
    xtimesteps = ''
    if alg == 'ppo' or alg == 'acktr' or alg == 'sac':
        xtimesteps = 'total_timesteps'
    elif alg == 'trpo':
        xtimesteps = 'TimestepsSoFar'

    return xtimesteps


def xepisodes_by_alg(alg):
    xepisodes = ''
    if alg == 'ppo' or alg == 'acktr' or alg == 'sac':
        xepisodes = 'total_timesteps'
    elif alg == 'trpo':
        xepisodes = 'EpisodesSoFar'

    return xepisodes


def label_by_alg(pol_alg, vf_alg):
    start = ''
    if pol_alg == 'ppo':
        start = 'PPO'
    elif pol_alg == 'trpo':
        start = 'TRPO'
    elif pol_alg == 'acktr':
        start = 'ACKTR'
    elif pol_alg == 'sac':
        start = 'SAC'

    if vf_alg == 'kalman':
        end = 'KOVA'
    elif vf_alg == 'acktr':
        return start
    else:
        end = 'Adam'
    return start + ' with ' + end


def comb_by_alg_env(alg, env):
    combs = []
    if alg == 'ppo':
        if env == 'Swimmer-v2':
            combs = ['99', '13']
        elif env == 'Hopper-v2':
            combs = ['99', '15']
        elif env == 'HalfCheetah-v2':
            combs = ['99', '12']
        elif env == 'Ant-v2':
            combs = ['99', '102']
        elif env == 'Walker2d-v2':
            combs = ['99', '13']
    elif alg == 'trpo':
        # '96' is ACKTR
        if env == 'Swimmer-v2':
            combs = ['99', '12', '96']
        elif env == 'Hopper-v2':
            combs = ['99', '102', '96']
        elif env == 'HalfCheetah-v2':
            combs = ['99', '105', '96']
        elif env == 'Walker2d-v2':
            combs = ['99', '107', '96']
        elif env == 'Ant-v2':
            combs = ['99', '108', '96']
    elif alg == 'sac':
        if env == 'Swimmer-v2':
            combs = ['80', '22']
        elif env == 'Hopper-v2':
            combs = ['98', '2']
        elif env == 'HalfCheetah-v2':
            combs = ['98', '5']
        elif env == 'Walker2d-v2':
            combs = ['98', '1']
        elif env == 'Ant-v2':
            combs = ['98', '0']
    return combs


def df2xy(df, xaxis, yaxis, alg):
    if xaxis == X_TIMESTEPS:
        xtimesteps = xtimesteps_by_alg(alg)
        x_index = df.columns.get_loc(xtimesteps)
    elif xaxis == X_EPISODES:
        xepisodes = xepisodes_by_alg(alg)
        x_index = df.columns.get_loc(xepisodes)
    else:
        raise NotImplementedError
    x = np.squeeze(df.as_matrix(columns=[df.columns[x_index]]))
    try:
        y_index = df.columns.get_loc(yaxis)
    except:
        y_index = 0
    y = np.squeeze(df.as_matrix(columns=[df.columns[y_index]]))
    return x, y


def plot_curves(xy_list, xaxis, exp_args_vals=None, ax=None):
    policy_alg = exp_args_vals[1]["pol_alg"][4:]
    combs = comb_by_alg_env(alg=policy_alg, env=exp_args_vals[0]["env"])

    for i, comb in enumerate(combs):
        env = exp_args_vals[0]["env"]
        info = [inf for j, inf in enumerate(exp_args_vals) if inf["comb"] == comb]

        pol_alg = info[0]["pol_alg"][4:]
        xy_window = [window_func(x, y, EPISODES_WINDOW, np.mean, pol_alg, env) for idx, (x, y) in enumerate(xy_list)
                     if exp_args_vals[idx]["comb"] == comb]
        maxx = max(xy[0][-1] for xy in xy_window)
        minx = 0
        data = np.array(xy_window)

        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        pol_alg = info[0]["pol_alg"][4:]
        vf_alg = info[0]["vf_alg"][3:]
        label = label_by_alg(pol_alg, vf_alg)

        if vf_alg == 'kalman':
            color = 'blue'
            line = '-'
        elif vf_alg == 'acktr':
            color = 'green'
            line = '-.'
        else:
            color = 'red'
            line = '--'

        ax.plot(data_mean[0], data_mean[1], line, color=color, linewidth=2, label=label)
        ax.fill_between(data_mean[0], data_mean[1] + data_std[1], data_mean[1] - data_std[1],
                        facecolor=color, alpha=0.2)
        ax.yaxis.set_tick_params(labelsize=14)

        if xaxis == X_TIMESTEPS:
            ax.set_xlim(minx, maxx)
            ax.set_xticks((minx, (minx+maxx)/2, maxx))
            ax.set_xticklabels((str(minx), "{:.0e}".format(round((minx+maxx)/2, -5)),
                                "{:.0e}".format(round(maxx, -5))))
            ax.xaxis.set_tick_params(labelsize=16)


class LoadMonitorResultsError(Exception):
    pass


def load_results_progress(dir, alg):
    import pandas
    progress_files = (
        glob(osp.join(dir, "*progress.csv")))
    if not progress_files:
        raise LoadMonitorResultsError("no monitor files of the form *%s found in %s" % (EXT, dir))
    dfs = []
    for fname in progress_files:
        with open(fname, 'rt') as fh:
            df = pandas.read_csv(fh, header=0, index_col=None)
        dfs.append(df)
    df = pandas.concat(dfs)
    xtimesteps = xtimesteps_by_alg(alg)
    df.sort_values(xtimesteps, inplace=True)

    df.reset_index(inplace=True)
    return df


def plot_loss(dirs, algorithms, xaxis, exp_args_vals=None):
    figsize = (18., 6.)
    margins = {
        "left": 1.5 / figsize[0],
        "bottom": 0.8 / figsize[1],
        "right": 0.8,
        "top": 0.9,
        "hspace": 0.25}

    fig, axs = plt.subplots(len(dirs), len(dirs[0]), figsize=figsize)
    fig.subplots_adjust(**margins)

    with PdfPages(dirs[0][0][0] + '/train_reward.pdf') as pdf:
        yaxis = 'eprewmean'
        for alg_index, algorithm_data in enumerate(dirs):
            for i, env in enumerate(algorithm_data):
                dflist = []
                algs = []
                print("alg", algorithms[alg_index])
                for k, dir in enumerate(env):
                    alg_temp = exp_args_vals[alg_index][i][k]["pol_alg"][4:]
                    algs.append(alg_temp)
                    df = load_results_progress(dir, alg_temp)
                    dflist.append(df)
                ax = axs[alg_index, i]
                xy_list = [df2xy(df, xaxis, yaxis, algs[k]) for k, df in enumerate(dflist)]
                plot_curves(xy_list, xaxis, exp_args_vals[alg_index][i], ax=ax)

                # placing the legend at the right side of the subplots in every row
                if i == len(algorithm_data) - 1:
                    handles, labels = ax.get_legend_handles_labels()
                    if alg_index == 0:
                        ax.legend(handles, labels, fontsize=20, fancybox=True, shadow=True, bbox_to_anchor=(1.04, 0.55),
                                  loc="center left", borderaxespad=0)
                    elif alg_index == 1:
                        ax.legend(handles, labels, fontsize=20, fancybox=True, shadow=True, bbox_to_anchor=(1.04, 0.5),
                                  loc="center left", borderaxespad=0)
                    elif alg_index == 2:
                        ax.legend(handles, labels, fontsize=20, fancybox=True, shadow=True, bbox_to_anchor=(1.04, 0.45),
                                  loc="center left", borderaxespad=0)

                # placing the title of environments at the top of the figure
                if alg_index == 0:
                    ax.set_title(exp_args_vals[alg_index][i][0]["env"], fontsize=20)

                # placing ylabel at the left side of the figure
                if i == 0:
                    ax.set_ylabel('Mean\nepisode\nreward', fontsize=18, rotation=0, ha='right', va='center', ma='left',
                                  labelpad=5)

                # placing xlabel at the botttom of the figure
                if alg_index == len(dirs) - 1:
                    ax.set_xlabel("Time steps", fontsize=18)
                else:
                    ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
                plt.subplots_adjust(wspace=0.5)

                ax.grid(which='both', linestyle='--')
        pdf.savefig(dpi=200)


def main():
    import argparse
    import os

    algs = ['ppo', 'trpo', 'sac']
    envs = ['Swimmer-v2', 'Hopper-v2', 'HalfCheetah-v2', 'Walker2d-v2', 'Ant-v2']

    list_of_dirs = ["/home/path/{}".format(alg) for alg in algs]
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--results_dir', default=list_of_dirs)
    parser.add_argument('--exp_dirs', help='List of log directories', nargs='*', default=envs)
    parser.add_argument('--xaxis', help='Varible on X-axis', default=X_TIMESTEPS)  # X_TIMESTEPS X_EPISODES X_WALLTIME

    args = parser.parse_args()
    args.dirs = [[os.path.join(alg_list, dir) for dir in args.exp_dirs] for alg_list in args.results_dir]
    subdirs = [[os.listdir(env) for env in alg_list] for alg_list in args.dirs]
    total_data = []
    total_exp_args_vals = []
    for alg_idx, alg in enumerate(subdirs):
        temp2 = []
        exp_args_vals = []
        for idx, env in enumerate(alg):
            # This is a list of all results dir inside a apecific env in a specific algorithm
            temp = [os.path.join(args.dirs[alg_idx][idx], dir) for dir in env]
            parse_dir = [string.split('_') for string in env]
            if algs[alg_idx] == 'ppo':
                arguments_names = ["exp", "date", "env", "pol_alg", "vf_alg", "learning_rate", "onv_coeff", "eta",
                                   "onv_type", "body_mass", "batch_size", "seed", "comb", "last_layer", "separateVars",
                                   "eval"]
            else:
                arguments_names = ["exp", "date", "env", "pol_alg", "vf_alg", "learning_rate", "onv_coeff", "eta",
                                   "onv_type", "body_mass", "seed", "comb", "last_layer", "separateVars", "eval"]

            temp3 = [dict(zip(arguments_names, p)) for p in parse_dir]
            for d in temp3:
                d['plot_mean'] = True
            temp2.append(temp)
            exp_args_vals.append(temp3)
        total_data.append(temp2)
        total_exp_args_vals.append(exp_args_vals)

    args.dirs = total_data
    plot_loss(args.dirs, algs, args.xaxis, exp_args_vals=total_exp_args_vals)
    plt.show()


if __name__ == '__main__':
    main()

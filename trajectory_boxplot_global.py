import os
from os.path import join, basename, dirname, normpath
import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Cardo']})
rc('text', usetex=True)

def process_dir(path):
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    trans_data = np.concatenate([v['rel_trans_perc'] for v in data.values()])
    rot_data = np.concatenate([v['rel_rot_deg_per_m'] for v in data.values()])
    return rot_data, trans_data

def method_from_path(path):
    path = normpath(path)
    return basename(dirname(dirname(dirname(dirname(path)))))

def key(x):
    return np.median(trans_data[x])


if __name__ == '__main__':
    parser = argparse.ArgumentParser("""""")
    parser.add_argument("--results", default="/home/dani/Documents/projects/estimation_vs_resolution/data/results/pose_tracking_water1_var_contrast_threshold/")
    args = parser.parse_args()

    fig, ax = plt.subplots(ncols=2, figsize=(20,10))
    methods = {}
    for path, dirs, files in os.walk(args.results):
        if "cached_rel_err.pickle" in files:
            method = method_from_path(path)
            methods[method] = join(path, "cached_rel_err.pickle")

    all_methods = sorted(methods.keys())
    rot_data = []
    trans_data = []
    for i, m in enumerate(all_methods):
        rot_datum, trans_datum = process_dir(methods[m])
        rot_data += [rot_datum]
        trans_data += [trans_datum]

    n_methods = len(all_methods)
    ax[0].set_xlim([0,n_methods])
    ax[1].set_xlim([0,n_methods])

    index = sorted(np.arange(n_methods), key=key)
    rot_data = [rot_data[i] for i in index]
    trans_data = [trans_data[i] for i in index]
    all_methods = [all_methods[i] for i in index]


    ax[0].boxplot(rot_data, positions=np.arange(n_methods), labels=all_methods)
    ax[0].set_xticklabels(all_methods, rotation=90)
    ax[0].legend()
    ax[0].set_ylabel(r"Translation Error [\%]")
    ax[0].set_yscale("log")

    ax[1].boxplot(trans_data, positions=np.arange(n_methods), labels=all_methods)
    ax[1].set_xticklabels(all_methods, rotation=90)
    ax[1].legend()
    ax[1].set_ylabel(r"Rotation Error [deg / m]")
    ax[1].set_yscale("log")

    fig.savefig(join(args.results, "boxplots_all.pdf"), bbox_inches="tight")

    plt.show()


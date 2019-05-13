# License: MIT
# Author: Karl Stelzner

import csv
import os

import visdom
import matplotlib.pyplot as plt
import numpy as np

vis = visdom.Visdom()


def read_csv(filename):
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        rows = []
        for row in reader:
            rows.append(row)

    return rows


def rows_to_matrix(rows):
    data = np.zeros((len(rows), 3))
    for i, row in enumerate(rows):
        data[i, :] = row[:3]
    return data


def process_perf(data, max_t):
    spacing = min(15, max_t // 10)
    bins = np.linspace(0, max_t + spacing, spacing)
    digitized = np.digitize(data[:, 1], bins)
    means = [data[digitized == i, 2].mean() for i in range(1, len(bins))]
    stdev = [data[digitized == i, 2].std() for i in range(1, len(bins))]
    offset = (bins[1] - bins[0]) / 2
    bins -= offset
    bins = np.clip(bins, 0.0, 2 * max_t)
    return bins[1:], np.array(means), np.array(stdev)


def make_perf(spair_file, air_file, output_file, max_t,
              add_annotation=None, directory='./plots', no_bg_file=None):
    spair_file = os.path.join(directory, spair_file)
    air_file = os.path.join(directory, air_file)
    output_file = os.path.join(directory, output_file)

    spn_data = rows_to_matrix(read_csv(spair_file))
    air_data = rows_to_matrix(read_csv(air_file))

    air_bins, air_means, air_stdev = process_perf(air_data, max_t + 20)
    spn_bins, spn_means, spn_stdev = process_perf(spn_data, max_t + 20)

    print(spair_file, 'AIR:', air_means[-2:], 'SuPAIR', spn_means[-2:])

    fig, plot = plt.subplots(figsize=[3.8, 2.44])
    plt.subplots_adjust(top=0.97, bottom=0.21,
                        left=0.19, right=0.95,
                        wspace=0.1, hspace=0.1)
    plot.fill_between(spn_bins, spn_means + spn_stdev, spn_means - spn_stdev, alpha=0.5)
    plot.plot(spn_bins, spn_means, linewidth=2.0, label='SuPAIR')

    if no_bg_file is not None:
        no_bg_file = os.path.join(directory, no_bg_file)
        nobg_data = rows_to_matrix(read_csv(no_bg_file))
        nobg_bins, nobg_means, nobg_stdev = process_perf(nobg_data, max_t + 20)
        plot.fill_between(nobg_bins, nobg_means + nobg_stdev, nobg_means - nobg_stdev, alpha=0.5)
        plot.plot(nobg_bins, nobg_means, linewidth=2.0, label='SuPAIR w/o bg')

    plot.fill_between(air_bins, air_means + air_stdev, air_means - air_stdev, alpha=0.5)
    plot.plot(air_bins, air_means, linewidth=2.0, label='AIR')
    plot.set_xlim(min(air_bins), max_t)
    plot.set_ylim(-0.002, 1.0)
    plot.set_xlabel('time (s)', fontsize=14)
    plot.set_ylabel('count accuracy', fontsize=14)
    plot.tick_params(labelsize=12)
    plot.legend(loc='lower right')
    # if add_annotation is not None:
    #     add_annotation(plot)

    vis.matplot(plt)
    plt.savefig(output_file)


def add_mnist_annotation(plot):
    plot.annotate('SuPAIR', xy=(93, 0.92), xytext=(103, 0.72), fontsize=14,
                  horizontalalignment='center',
                  arrowprops=dict(facecolor='tab:blue', shrink=0.05,
                                  width=3, headwidth=10))

    plot.annotate('AIR', xy=(220, 0.57), xytext=(250, 0.37), fontsize=14,
                  horizontalalignment='center',
                  arrowprops=dict(facecolor='tab:orange', shrink=0.05,
                                  width=3, headwidth=10))


def add_spair_annotation(plot):
    plot.annotate('SuPAIR', xy=(63, 0.92), xytext=(71, 0.72), fontsize=14,
                  horizontalalignment='center',
                  arrowprops=dict(facecolor='tab:blue', shrink=0.05,
                                  width=3, headwidth=10))

    plot.annotate('AIR', xy=(220, 0.57), xytext=(230, 0.37), fontsize=14,
                  horizontalalignment='center',
                  arrowprops=dict(facecolor='tab:orange', shrink=0.05,
                                  width=3, headwidth=10))


make_perf('spair_mnist.csv', 'air_mnist.csv', 'mnist_perf.pdf',
          380, add_mnist_annotation, directory='./performance_data', no_bg_file='no_bg_mnist.csv')
make_perf('spair_sprites.csv', 'air_sprites.csv', 'sprite_perf.pdf',
          250, add_spair_annotation, directory='./performance_data', no_bg_file='no_bg_sprites.csv')
make_perf('spair_noisy_mnist.csv', 'air_noisy_mnist.csv', 'noisy_mnist_perf.pdf',
          450, add_spair_annotation, directory='./performance_data', no_bg_file='no_bg_noise.csv')
make_perf('spair_struc_noisy_mnist.csv', 'air_struc_mnist.csv', 'struc_noisy_mnist_perf.pdf',
          450, add_spair_annotation, directory='./performance_data', no_bg_file='no_bg_struc.csv')


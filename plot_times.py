#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Plot cuda vs eigen blas performance"
    )
    parser.add_argument("filename")
    args = parser.parse_args()

    df = pd.read_csv(args.filename, sep='\t')

    facet_kws=dict(sharey=False)
    ax = sns.relplot(
        data=df, x='size', y='time', col='algo', hue='algo', style='type', kind='line',
        facet_kws=facet_kws
    )
    ax.set_xlabels("Problem Size (n)")
    ax.set_ylabels("Time (s)")
    #plt.savefig('perf-results.png')
    plt.show()

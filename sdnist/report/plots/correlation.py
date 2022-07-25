import os
from typing import List
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd

plt.style.use('seaborn-deep')


class CorrelationDifferencePlot:
    def __init__(self,
                 synthetic: pd.DataFrame,
                 target: pd.DataFrame,
                 output_directory: Path,
                 features: List[str]):
        """
        Computes and plots features correlation difference between
        synthetic and target data

        Parameters
        ----------
            synthetic : pd.Dataframe
                synthetic dataset
            target : pd.Dataframe
                target dataset
            output_directory: pd.Dataframe
                path of the directory to which plots will will be saved
            features: List[str]
                List of names of features for which to compute correlation
        """
        self.syn = synthetic
        self.tar = target
        self.o_dir = output_directory
        self.plots_path = Path(self.o_dir, 'correlation_difference')
        self.features = features
        self._setup()

    def _setup(self):
        if not self.o_dir.exists():
            raise Exception(f'Path {self.o_dir} does not exist. Cannot save plots')

        os.mkdir(self.plots_path)

    def save(self) -> List[Path]:
        corr_df = correlation_difference(self.syn, self.tar, self.features)
        return save_correlation_difference_plot(corr_df, self.plots_path)


def correlations(data: pd.DataFrame, features: List[str]):
    corr_list = []

    for f_a in features:
        f_a_corr = []
        for f_b in features:
            c_val = data[f_a].corr(data[f_b], method='kendall')
            f_a_corr.append(c_val)
        corr_list.append(f_a_corr)

    return pd.DataFrame(corr_list, columns=features, index=features)


def correlation_difference(synthetic: pd.DataFrame,
                           target: pd.DataFrame,
                           features: List[str]) -> pd.DataFrame:

    syn_corr = correlations(synthetic, features)
    tar_corr = correlations(target, features)

    diff = syn_corr - tar_corr

    return diff


def save_correlation_difference_plot(correlation_data: pd.DataFrame,
                                     output_directory: Path) -> List[Path]:
    cd = correlation_data[reversed(correlation_data.columns)].abs()
    plt.imshow(cd, cmap='Blues', interpolation='none')
    plt.colorbar()
    plt.xticks(range(cd.shape[1]), cd.columns)
    plt.yticks(range(cd.shape[0]), cd.index)
    file_path = Path(output_directory, 'corr_diff.jpg')
    plt.title('Correlation Diff. between Synthetic and Target')
    plt.savefig(file_path)
    plt.close()

    return [file_path]


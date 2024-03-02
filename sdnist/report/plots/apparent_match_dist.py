import os

from typing import List
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sdnist.metrics.apparent_match_dist import cellchange
from sdnist.utils import *
from sdnist.report.column_combs.column_combs import ColumnCombs

# plt.style.use('seaborn-v0_8-deep')


def plot_apparent_match_dist(match_percentages: pd.Series,
                             output_directory: Path) -> Path:
    fig = plt.figure(figsize=(6, 6), dpi=100)

    if len(match_percentages):
        df = pd.DataFrame(match_percentages, columns=['perc'])
    else:
        df = pd.DataFrame([200], columns=['perc'])
    df.hist(width=1.5, align='mid')
    plt.xlim(0, 100)
    ax = plt.gca()
    ax.grid(False)
    ax.locator_params(axis='y', integer=True)
    plt.xlabel('Match Percentage', fontsize=14)
    plt.ylabel('Record Counts', fontsize=14)
    plt.title(
        'Percentage of Matched Records')
    out_file = Path(output_directory, f'apparent_match_distribution.jpg')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1 - 2, x2 + 2, y1, y2 + 0.05))
    fig.tight_layout()
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    return out_file


class ApparentMatchDistributionPlot:
    def __init__(self,
                 synthetic: pd.DataFrame,
                 target: pd.DataFrame,
                 output_directory: Path,
                 quasi_features: List[str],
                 exclude_features: List[str],
                 col_comb: Optional[ColumnCombs] = None,
                 ):
        """
        Computes and plots apparent records match distribution between
        synthetic and target data

        Parameters
        ----------
            synthetic : pd.Dataframe
                synthetic dataset
            target : pd.Dataframe
                target dataset
            output_directory: pd.Dataframe
                path of the directory to which plots will will be saved
            quasi_features: List[str]
                Subset of features for which to find apparent record matches
            exclude_features:
                features to exclude from matching between dataset
        """
        self.syn = synthetic
        self.tar = target
        self.o_dir = output_directory
        self.o_path = Path(self.o_dir, 'apparent_match_distribution')
        self.quasi_features = quasi_features
        self.exclude_features = exclude_features
        self.quasi_matched_df = pd.DataFrame()
        self.report_data = dict()
        self.col_comb = col_comb
        self._setup()

    def _setup(self):
        if not self.o_dir.exists():
            raise Exception(f'Path {self.o_dir} does not exist. Cannot save plots')

        os.mkdir(self.o_path)

    def save(self) -> List[Path]:
        percents, u1, u2, mu = cellchange(self.syn, self.tar,
                                          self.quasi_features,
                                          self.exclude_features)
        self.quasi_matched_df = mu

        save_file_path = plot_apparent_match_dist(percents,
                                                  self.o_path)
        mu['percent_match'] = percents
        self.report_data['unique_matched_percents'] = \
            relative_path(save_data_frame(mu, self.o_path, 'unique_matched_percents'))
        self.report_data['plot'] = relative_path(save_file_path)

        '''
        The above code is backwards compatible with legacy SDNIST. Now
        we want to consider that, in a query-based setting, an attacker
        would query for only the quasi-identifiers and a single additional
        feature that the attacker wanted to infer. We do that here.
        Note that we use 'c_' tables
        '''
        self.report_data['query_unique_matches'] = {}
        self.report_data['quasi_identifiers'] = self.quasi_features
        non_qi_features = list(set(self.quasi_features) ^ set(self.tar.columns))
        for non_qi_feature in non_qi_features:
            if non_qi_feature in self.exclude_features:
                continue
            all_features = self.quasi_features + [non_qi_feature]
            df_syn = self.col_comb.getDataframeByColumns(all_features, version = 'c_')
            percents, u1, u2, mu = cellchange(df_syn[all_features],
                                              self.tar[all_features],
                                              self.quasi_features,
                                              self.exclude_features)
            unique_matches = len(percents[percents == 100.0])
            if len(percents) > 0:
                percent = unique_matches / len(percents)
            else:
                percent = 0
            self.report_data['query_unique_matches'][non_qi_feature] = \
                {'matches': unique_matches,
                 'unique_quasi_identifiers':len(percents),
                 'percent': percent,
                }
        return [save_file_path]

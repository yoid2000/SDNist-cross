from typing import Optional, List
import pandas as pd

from scipy.stats import pearsonr
from sdnist.report.column_combs.column_combs import ColumnCombs


class PearsonCorrelationDifference:
    def __init__(self,
                 target: pd.DataFrame,
                 synthetic: pd.DataFrame,
                 features: Optional[List[str]] = None,
                 col_comb: Optional[ColumnCombs] = None,
                 wpf_values: Optional[List] = None,
                 wpf_feature: Optional[str] = None):
        self.features = features if features \
            else target.columns.tolist()
        self.target = target[self.features]
        self.synthetic = synthetic[self.features]
        self.col_comb = col_comb
        self.wpf_values = wpf_values
        self.wpf_feature = wpf_feature

        # pair-wise pearson correlation difference of given features
        self.pp_corr_diff = pd.DataFrame()  # pair-wise pearson correlation difference
        self.target_corr = pd.DataFrame()  # pair-wise correlations in target data
        self.synthetic_corr = pd.DataFrame()  # pair-wise correlations in synthetic data

    def compute(self):
        self.pp_corr_diff = self.pair_wise_difference()

    def pair_wise_difference(self) -> pd.DataFrame:
        self.target_corr = self.pair_wise_correlations(self.target)
        self.synthetic_corr = self.pair_wise_correlations(self.synthetic,
                                                          col_comb=self.col_comb,
                                                          wpf_feature = self.wpf_feature,
                                                          wpf_values = self.wpf_values)

        diff = self.synthetic_corr - self.target_corr
        return diff

    def pair_wise_correlations(self, data: pd.DataFrame,
                               col_comb: Optional[ColumnCombs] = None,
                               wpf_values: Optional[List] = None,
                               wpf_feature: Optional[str] = None) -> pd.DataFrame:
        data = data.copy()
        corr_list = []
        for f_a in self.features:
            f_corr = []
            for f_b in self.features:
                if col_comb is not None:
                    data = col_comb.getDataframeByColumns([f_a, f_b],
                                                          wpf_values = wpf_values,
                                                          wpf_feature = wpf_feature,
                                                          version='t_')
                corr, _ = pearsonr(data[f_a], data[f_b])
                f_corr.append(corr)
            corr_list.append(f_corr)
        return pd.DataFrame(corr_list, columns=self.features, index=self.features)

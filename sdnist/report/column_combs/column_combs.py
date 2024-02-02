import pandas as pd
import os
from pathlib import Path
from typing import List, Optional
from sdnist.report import Dataset
from sdnist.report.dataset.validate import validate
from sdnist.report.dataset.binning import *
from sdnist.report.dataset.transform import transform
from sdnist.utils import SimpleLogger
from sdnist.load import TestDatasetName

def _makeColumnsKey(columns):
    columns.sort()
    return '.'.join(columns)

class ColumnCombs:
    def __init__(self,
                 synthetic_filepath: Path,
                 dataset_name: TestDatasetName,
                 data_root: Path,
                 exact_matches_only: Optional[bool] = True 
                 ):
        """
        Reads in all of the synthetic tables (for each column combination)
        Remembers which synthetic table has the most columns. This table is returned
        when there is no exact column combination match.

        Parameters
        ----------
            synthetic_filepath: Path,
                Path to the synthetic datafile with all columns
                All synthetic datafiles must be in the same directory
            data_root: Path,
                The data_root of `report/__main__.py`
            exact_matches_only: bool
                Set to True if only exact column matches should be used. 
                When True, throws an exception if exact match not found.
                When False, returns table with all combinations if exact match
                    not found.
        """
        self.exact_matches_only = exact_matches_only
        self.col_combs_dir = synthetic_filepath.parent
        self.encountered_combs = []
        self.missing_combs = []
        csv_files = [f for f in os.listdir(self.col_combs_dir) if f.endswith('.csv')]
        self.comb_dataframes = {}
        max_num_columns = 0
        self.default_col_key = ''
        for csv_file in csv_files:
            csv_path = os.path.join(self.col_combs_dir, csv_file)
            df = pd.read_csv(csv_path)
            log = SimpleLogger()
            log.disabled = True
            comb_dataset = Dataset(csv_path, log, dataset_name, data_root, False)
            columns = comb_dataset.synthetic_data.columns.tolist()
            col_key = _makeColumnsKey(columns)
            if len(columns) > max_num_columns:
                self.default_col_key = col_key
                max_num_columns = len(columns)
            if comb_dataset.d_synthetic_data is None:
                raise Exception(f'Missing d_synthetic_data for {col_key}')
            if comb_dataset.t_synthetic_data is None:
                raise Exception(f'Missing t_synthetic_data for {col_key}')
            if comb_dataset.synthetic_data is None:
                raise Exception(f'Missing synthetic_data for {col_key}')
            self.comb_dataframes[col_key] = comb_dataset

    def getDataframeByColumns(self,
                              columns: List[str],
                              wpf_values: Optional[List] = None,
                              wpf_feature: Optional[str] = None, 
                              version: Optional[str] = 'initial') -> pd.DataFrame:
        """
        Returns the synthetic dataframe with the corresponding columns
        """
        if wpf_feature is not None:
            columns += [wpf_feature]
        # Remove duplicates (can happen if for instance correlation between
        # the same column is being computed)
        columns = list(set(columns))
        columns.sort()
        self.encountered_combs.append([version, columns])
        col_key = _makeColumnsKey(columns)
        if col_key not in self.comb_dataframes:
            if self.exact_matches_only:
                raise Exception(f'Could not find {col_key} in comb_dataframes')
            else:
                col_key = self.default_col_key
        if version == 'd_':
            df_syn = self.comb_dataframes[col_key].d_synthetic_data
        elif version == 't_':
            df_syn = self.comb_dataframes[col_key].t_synthetic_data
        elif version == 'initial':
            df_syn = self.comb_dataframes[col_key].synthetic_data
        elif version == 'c_':
            df_syn = self.comb_dataframes[col_key].c_synthetic_data
        else:
            raise Exception(f'Unexpected col_comb version {version}')
        # make a copy, cause I'm not 100% sure that the calling code won't modify df_syn
        df_syn = df_syn.copy()
        if wpf_feature:
            # Select subset of rows where column wpf_feature matches wpf_values
            # TODO: here we assume we need the initial dataframe, but cleaner if this
            # knowledge is handed to us from the caller
            df_syn_initial = self.comb_dataframes[col_key].synthetic_data
            df_syn = df_syn[df_syn_initial[wpf_feature].isin(wpf_values)]
        return df_syn

    def saveEncounteredColumns(self):
        ''' This is simply for the purpose of learning what combinations
        have been requested by SDNIST. It is otherwise not operational.
        '''
        import json

        all_columns = [i[1] for i in self.encountered_combs]
        distinct_columns = [list(x) for x in set(tuple(x) for x in all_columns) ]
        distinct_columns.sort(key=len)
        print(distinct_columns)
        print(f"{len(distinct_columns)} distinct columns")
        with open("columns.json", 'w') as f:
            json.dump(distinct_columns, f, indent=4)


#
#            df = pd.read_csv(csv_path)
#            columns = list(df.columns)
#            columns.sort()
#            col_key = self._makeColumnsKey(columns)
#            self.comb_dataframes['initial'][col_key] = df
#            # The following mimics the steps in Dataset __post_init__
#            self.comb_dataframes['c_'][col_key], _ = \
#                        validate(df, dataset.data_dict, columns)
#            self.comb_dataframes['initial'][col_key] = self.comb_dataframes['c_'][col_key]
#            self.comb_dataframes['initial'][col_key] = self.comb_dataframes['initial'][col_key].reindex(sorted(self.comb_dataframes['initial'][col_key].columns), axis=1)
#            self.comb_dataframes['c_'][col_key] = self.comb_dataframes['c_'][col_key].reindex(sorted(self.comb_dataframes['c_'][col_key].columns), axis=1)
#            if 'DENSITY' in columns:
#                self.comb_dataframes['initial'][col_key] = bin_density(self.comb_dataframes['c_'][col_key], dataset.data_dict, update=True)
#            self.comb_dataframes['t_'][col_key] = transform(self.comb_dataframes['c_'][col_key], dataset.schema)
#            numeric_features = ['AGEP', 'POVPIP', 'PINCP', 'PWGTP', 'WGTP']
#            numeric_features = list(set(numeric_features) & set(columns))
#            if len(numeric_features) > 0:
#                print(f"Work on {numeric_features} for {columns}")
#                self.comb_dataframes['d_'][col_key] = \
#                    percentile_rank_synthetic(self.comb_dataframes['c_'][col_key],
#                                              dataset.c_target_data[numeric_features],
#                                              dataset.d_target_data[numeric_features],
#                                              numeric_features)
#                self.comb_dataframes['d_'][col_key] = \
#                    add_bin_for_NA(self.comb_dataframes['d_'][col_key],
#                                   self.comb_dataframes['c_'][col_key],
#                                   numeric_features)
#
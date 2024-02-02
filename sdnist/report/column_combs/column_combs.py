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
    return '.'.join(columns)

class ColumnCombs:
    def __init__(self,
                 dataset: Dataset,
                 synthetic_filepath: Path,
                 dataset_name: TestDatasetName,
                 data_root: Path
                 ):
        """
        Reads in all of the synthetic tables (for each column combination)

        Parameters
        ----------
            dataset: Dataset,
                The dataset created by the main program
            synthetic_filepath: Path,
                Path to the synthetic datafile with all columns
                All synthetic datafiles must be in the same directory
        """
        self.col_combs_dir = synthetic_filepath.parent
        self.encountered_combs = []
        self.missing_combs = []
        csv_files = [f for f in os.listdir(self.col_combs_dir) if f.endswith('.csv')]
        self.comb_dataframes = {}
        for csv_file in csv_files:
            csv_path = os.path.join(self.col_combs_dir, csv_file)
            df = pd.read_csv(csv_path)
            log = SimpleLogger()
            log.disabled = True
            comb_dataset = Dataset(csv_path, log, dataset_name, data_root, False)
            columns = comb_dataset.synthetic_data.columns.tolist()
            col_key = _makeColumnsKey(columns)
            if comb_dataset.d_synthetic_data is None:
                raise Exception(f'Missing d_synthetic_data for {col_key}')
            if comb_dataset.t_synthetic_data is None:
                raise Exception(f'Missing t_synthetic_data for {col_key}')
            if comb_dataset.synthetic_data is None:
                raise Exception(f'Missing synthetic_data for {col_key}')
            self.comb_dataframes[col_key] = comb_dataset

    def getDataframeByColumns(self,
                              columns: List[str],
                              version: Optional[str]='initial') -> pd.DataFrame:
        """
        Returns the synthetic dataframe with the corresponding columns
        """
        # Remove duplicates (can happen if for instance correlation between
        # the same column is being computed)
        columns = list(set(columns))
        columns.sort()
        self.encountered_combs.append([version, columns])
        col_key = _makeColumnsKey(columns)
        if col_key not in self.comb_dataframes:
            raise Exception(f'Could not find {col_key} in comb_dataframes')
        if version == 'd_':
            return self.comb_dataframes[col_key].d_synthetic_data
        elif version == 't_':
            return self.comb_dataframes[col_key].t_synthetic_data
        elif version == 'initial':
            return self.comb_dataframes[col_key].synthetic_data
        elif version == 'c_':
            return self.comb_dataframes[col_key].c_synthetic_data
        else:
            raise Exception(f'Unexpected col_comb version {version}')

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
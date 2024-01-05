from pathlib import Path
from typing import List, Optional

class ColumnCombs:
    def __init__(self,
                 synthetic_filepath: Path,
                 ):
        """
        Reads in all of the synthetic tables (for each column combination)

        Parameters
        ----------
            synthetic_filepath: Path,
                Path to the synthetic datafile with all columns
                All synthetic datafiles must be in the same directory
        """
        self.col_combs_dir = synthetic_filepath.parent
        self.encountered_combs = []
        pass

    def getDataframeByColumns(self,
                              columns: List[str],
                              version: Optional[str]='initial'):
        """
        Returns the synthetic dataframe with the corresponding columns
        """
        columns.sort()
        self.encountered_combs.append([version, columns])
        pass

    def saveEncounteredColumns(self):
        import json

        all_columns = [i[1] for i in self.encountered_combs]
        distinct_columns = [list(x) for x in set(tuple(x) for x in all_columns) ]
        distinct_columns.sort(key=len)
        print(distinct_columns)
        print(f"{len(distinct_columns)} distinct columns")
        with open("columns.json", 'w') as f:
            json.dump(distinct_columns, f, indent=4)
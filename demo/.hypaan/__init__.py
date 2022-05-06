import functools
from collections import OrderedDict

from .dataset_a import get_dataset_a_result_struct
from .dataset_b import get_dataset_b_result_struct

class DatasetDef:

    def __init__(self):
        self.dataset_structs = OrderedDict(
            dataset_a=get_dataset_a_result_struct,
            dataset_b=get_dataset_b_result_struct,
        )
    def get_names(self):
        """
        Returns list of possible datasets to load (will be passed to get_folder_struct)
        :return: list of str
        """
        return list(self.dataset_structs.keys())

    def get_folder_struct(self, name):
        """
        Returns list of strings or dict that represent the folder structure for searching. All possible
        combination of subfolders will be explored. Each subfolder is defined either by string (name of subfolder) or
        a dictionary where key is name of subfolder and values are list of attributes,

        The last element should be the name of the file containing metrics.

        :param str name: dataset name
        :return: list of str/dict
        """
        return self.dataset_structs[name]()

    def get_experiment_types(self):
        """
        Return list of available experiment types.
        Each type can have one or more strings that will be used to replace the
        first occurrence of {} in the provided folder structure.
        :return: dictionary with keys as name of experiment and values as str or list of str
        """
        return OrderedDict(TRAIN='train_',
                           TEST=['', 'test_'])

    def get_metrics_defintion(self):
        """
        Returns definition of metrics where that will be read from metrics file:
          - best_direction=1 indicates that high value is best, while for best_direction=-1, lowest value is best
          - best_val defines max best value
          - mod_fn applies modification of original read value
        :return: dictionary with keys as name of metrics and values as its options
        """
        import numpy as np
        return OrderedDict(  AP=dict(best_direction=1,  best_val=100, mod_fn=lambda x: x*100),
                             AR=dict(best_direction=1,  best_val=100, mod_fn=lambda x: x*100),
                             F1=dict(best_direction=1,  best_val=100, mod_fn=lambda x: x*100),
                             Re=dict(best_direction=-1, best_val=0,   mod_fn=lambda x: x*100),
                            mae=dict(best_direction=-1, best_val=0,   mod_fn=lambda x: x),
                           rmse=dict(best_direction=-1, best_val=0,   mod_fn=np.sqrt))


    def get_metrics_file_parser_fn(self):
        """
        Optional member that provides custom parsers of metrics file (default version loads metrics from json file)
        Parser function must return two dictionaries:
          - a dictionary with metrics
          - a dictionary with additional attributes (can be empty)
        :return: pointer to function that must return two dictionaries
        """
        def json_parser_fn(filename):
            import json
            with open(filename) as f: metrics = json.load(f)
            return metrics, dict(extra_attribute="custom_parser")
        return json_parser_fn

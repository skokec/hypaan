## haypaan: Hyper-Parameter Analyzer

Hyper-Parameter Analyzer is a tool for analyzing performance of hyper-parameters on machine-learning models. 
Hypaan provides a web interface for visualizing results over various hyper-parameter values. 

Features:
 * [hiplot](https://github.com/facebookresearch/hiplot) visualization
 * individual hyper-parameter impact analyzer (comparing effect of specific value against other values)
 * decision-tree visualization for analyzing the influence between different combinations of hyper-parameters
 * table and line plotting based on attribute selection and grouping  
 * custom selection/filtering and grouping of results based on user-defined attributes (e.g., selection of the best epoch or 
   the best threshold within specific hyper-parameter groups)
   
Loading of experiment runs:
 * a experiment run is defined with attributes and metrics (both as key-value pairs)
 * loading from user-defined multi-subfolder structures (with assigned attributes for each potential subfolder)
 * automatic searching of experiments by traversing all possible combination of subfolders  
 * capable of handling over 50.000 experiment runs
 * customizable loading:
  * user-defined metrics types
  * user-defined metrics file parsing (defaults to json files)

## Installation and dependencies 

Install as pip package:
```bash
pip install git+https://github.com/skokec/hypaan.git
```

Dependencies:
 * Python>=3.6
 * streamlit>=1.1.0 
 * hiplot>=0.1.31
 * others:
   * numpy, scipy, matplotlib, scikit_learn, tqdm, Pillow, dill

## Run web interface

Run using from installed package:
```bash
hypaan [PATH_TO_ROOT_DIR] [PATH_TO_DATASET_CONFIG] [streamlit_args]
```
Run provided demo by:
```bash
hypaan demo --server.port=6006
```
View results in browser at localhost:6006.

## Running from source

Checkout source from git and run using streamlit:
```bash
# clone source from git
git clone https://github.com/skokec/hypaan.git . 

# install dependencies
pip install -r requirements.txt

# using streamlit lanuch app `hypaan/run_web.py`
streamlit run hypaan/run_web.py [PATH_TO_ROOT_DIR] [PATH_TO_DATASET_CONFIG] [streamlit_args]
```

## How to load experiments

Each experiment is internally defined with attributes, i.e., (key,value) pairs, and metrics (metric is attribute as well).

Experiments need to be provided in folder/subfolder structure consisting of multiple levels of potential subfolders that
hypaan will traverse over. Each level can be defined with multiple potential subfolders, where each potential 
subfolder can have unique attributes assigned to them. 

Hypaan traverses over all levels of subfolders and generating every potential combination of paths where result can 
be found. Attributes from each level are collected together to form the final set of attributes for one experiment.

Last string in folder/subfolder structure MUST be filename from which metrics are loaded. By default, hypaan uses json
file as metrics, but this can be customized with user-provided loading function (see below).

## Dataset config

You must provide `DatasetDef` class containing list of all possible datasets (with search folders and their 
attributes as hyper-parameters). This must be provided as python package folder either in `[PATH_TO_ROOT_DIR]/.hypaan/` 
or in separately supplied `[PATH_TO_DATASET_CONFIG]` folder.

`DatasetDef` class must have the following members:
 * `get_names()`: list of datasets 
 * `get_folder_struct(name: str)`: folder structure with attributes
 * `get_experiment_types()`: types of experiments (e.g., TEST, TRAIN, VAL)
 * `get_metrics_defintion()`: types of metrics (e.g., AP, mAP, AR, F1, mae, ...)
 * (optional) `get_metrics_file_parser_fn()`: optional function to load/parse matrics file from provided folders

Example of the required dataset definition file is shown below. See provided [demo](/demo/.hypaan/__init__.py) example 
for more information.

```python
from collections import OrderedDict

class DatasetDef:

    def __init__(self):
        pass

    def get_folder_struct(self, name):
        """
        Returns list of strings or dict that represent the folder structure for searching. All possible
        combination of subfolders will be explored. Each subfolder is defined either by string (name of subfolder) or
        a dictionary where key is name of subfolder and values are list of attributes,

        The last element should be the name of the file containing metrics.

        :param str name: dataset name
        :return: list of str/dict
        """
        return [
           'exp',
           'dataset_A',
           {
               'subfolder1' : ['subfolder=1','attribute2=abc'],
               'subfolder2' : ['subfolder=2','attribute2=xyz']
           },
           'subsubfolder',        
           {
               '{}results_type1' : ['type=1'],
               '{}results_type2' : ['type=2']
           },
           'results.json'
       ]

    def get_names(self):
        """
        Returns list of possible datasets to load (will be passed to get_folder_struct)
        :return: list of str
        """
        return ['datasetA',]

    def get_experiment_types(self):
        """
        Return list of available experiment types.
        Each type can have one or more strings that will be used to replace the
        first occurrence of {} in the provided folder structure.
        :return: dictionary with keys as name of experiment and values as str or list of str
        """
        return OrderedDict(TRAIN='train_',
                           VAL='val_',
                           TEST=['', 'test_'])

    def get_metrics_defintion(self):
        """
        Returns definition of metrics where that will be read from metrics file:
          - best_direction=1 indicates that high value is best, while for best_direction=-1, lowest value is best
          - best_val defines max best value
          - mod_fn applies modification of original read value
        :return: dictionary with keys as name of metrics and values as its options
        """
        return OrderedDict(AP=dict(best_direction=1,  best_val=100, mod_fn=lambda x: x*100),
                           AR=dict(best_direction=1,  best_val=100, mod_fn=lambda x: x*100),
                           F1=dict(best_direction=1,  best_val=100, mod_fn=lambda x: x*100))
    # OPTIONAL 
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
            return metrics, dict(extra_attribute="my_additional_attr")
        return json_parser_fn
```


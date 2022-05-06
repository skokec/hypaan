import os, sys, random

import parser
import metric

import views.hyperplot
import views.lineplot
import views.hyperparam_impact
import views.decision_trees
import views.tableview

from streamlit import legacy_caching
import streamlit as st

import time
from multiprocessing import Pool

import importlib
import importlib.util

import logging

import pandas as pd
import numpy as np


@st.experimental_singleton
def get_plot_mutex_obj():
    return dict(lock=False)

class PlotMutex:
    def __init__(self):
        self.lock = get_plot_mutex_obj()

    def __enter__(self):
        while self.lock['lock']:
            time.sleep(0.1)

        self.lock['lock'] = True

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock['lock'] = False

class StreamlitHyperParamAnalyzer:

    def __init__(self, root_dir, dataset_def):
        self.parser = parser.ResultsParser(datasets)

        self.dataset_def = dataset_def
        self.all_dataset_names = dataset_def.get_names()
        self.root_dir = root_dir

    def run(self):

        @st.experimental_singleton
        def get_pool():
            return Pool(20)

        @st.cache
        def get_random_hash():
            return str(random.randint(-2**30,2**30))

        @st.experimental_memo(suppress_st_warning=True)
        def get_experiment(db, db_struct, db_type, rand_hash, _parser, _pool):
            exp_results, output_csv_file, traversing_details = _parser.parse_results_list(db, db_struct, db_type, _pool)
            param_list = _parser.get_available_params(exp_results)
            return exp_results, param_list, output_csv_file, traversing_details, rand_hash

        st.set_page_config(layout="wide",page_title="Hyper-Parameter Analyzer")

        all_db_types = list(self.dataset_def.get_experiment_types().keys())

        param_delim = ";"
        query_params = st.experimental_get_query_params()

        # set default settings based on URL query params
        logging.debug(query_params)
        settings = dict(db=self.all_dataset_names.index(query_params['db'][0])+1 if 'db' in query_params else 0,
                        db_type=all_db_types.index(query_params['db_type'][0]) if 'db_type' in query_params else 0)

        selected_dataset = st.sidebar.selectbox("Select dataset", ['None'] + self.all_dataset_names, index=settings['db'], key="db_select" )
        selected_dataset_type = st.sidebar.selectbox("Select dataset type", all_db_types, index=settings['db_type'], key="db_type_select")

        if st.sidebar.button('Re-run / clear cache'):
            legacy_caching.clear_cache()

        if selected_dataset != 'None':
            with st.spinner('Loading results ...'):
                start_t = time.time()
                pool = get_pool()
                rand_hash = get_random_hash()

                # selected_dataset and results_structures must be manually passed as args for caching to properly rerun
                # when it changes
                results_structures = self.dataset_def.get_folder_struct(selected_dataset)

                # add root folder as first part of results_structures
                results_structures = [self.root_dir] + results_structures

                exp_results, param_list, output_csv_file, traversing_details, _ = \
                    get_experiment(selected_dataset, results_structures, selected_dataset_type, rand_hash,
                                   self.parser, pool)

                with st.expander("Folder traversing info"):
                    df = pd.DataFrame(np.array([[len(l) if type(l) == dict else 1 for l in results_structures],
                                                traversing_details['found_level_counts']]),
                                      columns=['Level %d' % i for i in range(len(results_structures))],
                                      index=['Num defined sub-folders','Found total combinations'])
                    st.write(df)

                if len(exp_results) == 0:
                    first_empty_level = traversing_details['first_empty_level']
                    if type(first_empty_level) == str and len(first_empty_level) > 1024:
                        first_empty_level = first_empty_level[:1024] + "..."
                    elif type(first_empty_level) == dict:
                        first_empty_level = sorted(list(first_empty_level.keys()))
                        if len(first_empty_level) > 16:
                            first_empty_level = first_empty_level[:16] + ['...']
                        first_empty_level = " ".join(first_empty_level)

                    st.warning("WARNING: No metrics found for dataset '%s' of type '%s'! No matches in subfolders: %s" % (selected_dataset,
                                                                                                          selected_dataset_type,
                                                                                                          first_empty_level))
                    return

                display_param_list = sorted(list(param_list.keys()))

                all_variables_str = sorted(["%s=%s" % (k, v) for k, v_list in param_list.items() for v in v_list])

                end_t = time.time()
                st.success('Successfully loaded %d experiments with %d unique variables (in %.2f sec)!' % (len(exp_results), len(param_list), end_t-start_t))



            # set default settings based on URL query params
            settings.update(dict(groupby_attr=query_params['groupby_attr'][0].split(param_delim) if 'groupby_attr' in query_params else [],
                                 groupby_metric=query_params['groupby_metric'][0].split(param_delim) if 'groupby_metric' in query_params else [],
                                 groupby_dbtype=all_db_types.index(query_params['groupby_dbtype'][0]) if 'groupby_dbtype' in query_params else all_db_types.index(selected_dataset_type)))

            # override settings with state values if they exist
            for state_key, param_key in [('sidebar_groupby_attr','groupby_attr'),
                                         ('sidebar_groupby_metric','groupby_metric')]:
                if state_key in st.session_state and st.session_state[state_key] is not None:
                    settings[param_key] = st.session_state[state_key]

            ########################################################################################
            # apply additional grouping by attribute
            with st.sidebar.form(key='groupby_form'):
                st.write('Select within attribute group by best metric')
                groupby_attr = st.multiselect("Group by attribute", display_param_list, default=[a for a in settings['groupby_attr'] if a in display_param_list],
                                              key="sidebar_groupby_attr",)
                groupby_metric = st.multiselect("Metric:", self.parser.metric_list, default=[m for m in settings['groupby_metric'] if m in self.parser.metric_list],
                                                key="sidebar_groupby_metric",)
                groupby_dbtype = st.selectbox("Use metrics from dataset type (will ignore if same exp is not present):",
                                              all_db_types, key="sidebar_groupby_dbtype", index=settings['groupby_dbtype'])
                submit_button = st.form_submit_button(label='Apply')

            if groupby_attr is not None and groupby_metric is not None:
                with st.spinner('Filtering results ...'):
                    start_t = time.time()

                    if groupby_dbtype != selected_dataset_type:
                        exp_results_for_op_metrics, _, _, _, _ = \
                            get_experiment(selected_dataset, results_structures, groupby_dbtype, rand_hash,
                                           self.parser, pool)
                    else:
                        exp_results_for_op_metrics = None

                    exp_results = self.parser.experiment_group_by(exp_results, groupby_attr, groupby_metric,
                                                                  exp_results_for_op_metrics=exp_results_for_op_metrics)
                    end_t = time.time()

                    st.success('Successfully filtered to only %d experiments (in %.2f sec)!' % (len(exp_results), end_t-start_t))


            # save settings for URL query params
            settings = dict(db=selected_dataset,
                            db_type=selected_dataset_type,
                            groupby_attr=param_delim.join(groupby_attr),
                            groupby_metric=param_delim.join(groupby_metric),
                            groupby_dbtype=str(groupby_dbtype))

            plot_mutex = PlotMutex()

            #######################################################################################
            with st.expander("Hiplot"):
                logging.info('Ploting hiplot')
                cfg = views.hyperplot.display_hiplot(self.parser, exp_results, output_csv_file, param_list, groupby_attr)
                settings.update(cfg)

            #######################################################################################
            with st.expander("Table-view"):
                logging.info('Show table')
                cfg = views.tableview.display_tableview(self.parser, exp_results, param_list, all_variables_str)
                settings.update(cfg)

            ########################################################################################
            with st.expander("Line-plot"):
                logging.info('Plot lines')
                cfg = views.lineplot.display_lineplot(self.parser, exp_results, param_list, all_variables_str, groupby_attr, plot_mutex)
                settings.update(cfg)

            ########################################################################################
            with st.expander("Hyper-param impact"):
                logging.info('Calculating per hyper-parameter impact')
                cfg = views.hyperparam_impact.display_hyperparam_impact(self.parser, exp_results, param_list, all_variables_str, groupby_attr, plot_mutex)
                settings.update(cfg)

            ########################################################################################
            with st.expander("Hyper-param decision-tree"):
                logging.info('Finding useful decision-trees in hyper-param')
                cfg = views.decision_trees.display_hyperparam_decisiontrees(self.parser, exp_results, param_list, all_variables_str, groupby_attr, plot_mutex)
                settings.update(cfg)

            st.experimental_set_query_params(**{k:v for k,v in settings.items() if v is not None and len(v) > 0})

def load_user_datasets(dataset_path):
    # create local temporary root where link to 'datasets_struct' definition folder will be added
    # (this is needed to ensure folder has predefined name for proper importing)
    HYPAAN_TEMP_DIR = os.environ.get('HYPAAN_TEMP_DIR', './')
    os.makedirs(HYPAAN_TEMP_DIR, exist_ok=True)

    sys.path.append(HYPAAN_TEMP_DIR)
    local_dataset_path = os.path.join(HYPAAN_TEMP_DIR, 'datasets_struct')

    # create symlink from fixed local name to dataset_path
    if os.path.exists(local_dataset_path) or os.path.islink(local_dataset_path):
        os.remove(local_dataset_path)
    os.symlink(os.path.abspath(dataset_path), local_dataset_path, target_is_directory=True)

    # reload all functions/submodules of 'datasets_struct' that were previously loaded
    for k, m in sys.modules.items():
        if k.startswith('datasets_struct'):
            importlib.reload(m)


    # load datasets_struct as module
    datasets_spec = importlib.util.spec_from_file_location("datasets_struct",
                                                           os.path.join(local_dataset_path, '__init__.py'))
    datasets_struct = importlib.util.module_from_spec(datasets_spec)
    datasets_spec.loader.exec_module(datasets_struct)
    # manually add module to datasets_struct for pickle support
    sys.modules['datasets_struct'] = datasets_struct

    # ensure dataset module contains 'DatasetDef' class
    assert 'DatasetDef' in dir(datasets_struct), "Missing 'DatasetDef' class in '%s'" % dataset_path

    # extract user-provided definition of all datasets (folder structures, experiment types and metrics)
    datasets = datasets_struct.DatasetDef()

    # ensure datasets obj has all required members
    assert all([n in dir(datasets) for n in ['get_folder_struct', 'get_names',
                                             'get_experiment_types', 'get_metrics_defintion']]), \
        "ERROR: User-provided 'DatasetDef' class from '%s' is missing one of the following function: " + \
        " 'get_folder_struct','get_names', 'get_experiment_types' or 'get_metrics_defintion' !"

    return datasets

if __name__ == "__main__":

    # setup logging and excpetion handling
    logging.basicConfig(level=logging.INFO)

    def excepthook(*args):
        logging.getLogger().error('Uncaught exception:', exc_info=args)
    sys.excepthook = excepthook

    # disable GUI for pylab (using non-display/interactive mode)
    import pylab as plt

    plt.ioff()
    plt.switch_backend("agg")

    # force reload of all modules otherwise old version will be used on each re-run
    importlib.reload(parser)
    importlib.reload(metric)
    importlib.reload(views.hyperplot)
    importlib.reload(views.lineplot)
    importlib.reload(views.hyperparam_impact)
    importlib.reload(views.decision_trees)
    importlib.reload(views.tableview)

    root_dir = sys.argv[1] if len(sys.argv) > 1 else './'
    dataset_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root_dir,'.hypaan')

    datasets = load_user_datasets(dataset_dir)

    # run analyzer
    st_analizer = StreamlitHyperParamAnalyzer(root_dir, datasets)
    st_analizer.run()
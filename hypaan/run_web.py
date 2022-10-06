import os, sys, random

import parser
import metric

import views.hyperplot
import views.lineplot
import views.hyperparam_impact
import views.decision_trees
import views.tableview
import views.prefiltering


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

class ProgressPlaceholder:
    def __init__(self, placeholer, num_steps=0):
        self.placeholder = placeholer
        self.num_steps = num_steps
        self.current_step = 0

    def set_num_steps(self, n):
        self.num_steps = n

    def increase(self):
        with self.placeholder: st.progress(self.current_step / self.num_steps)
        self.current_step += 1

    def clear(self):
        with self.placeholder: st.write("")

class StatusPlaceholder:
    def __init__(self, placeholer, prefix="", suffix=""):
        self.placeholder = placeholer
        self.prefix = prefix
        self.suffix = suffix
    def update(self, text):
        with self.placeholder: st.caption(self.prefix + text + self.suffix)

class StreamlitHyperParamAnalyzer:

    def __init__(self, root_dir, dataset_def):
        self.parser = parser.ResultsParser(datasets)

        self.dataset_def = dataset_def
        self.all_dataset_names = dataset_def.get_names()
        self.root_dir = root_dir

        self.sidebar_update_placeholder = None
        self.sidebar_progress_placeholder = None

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
                        db_type=all_db_types.index(query_params['db_type'][0]) if 'db_type' in query_params else 0,
                        display_views=query_params['display_views'] if 'display_views' in query_params else ['hiplot'])

        sidebar_update = StatusPlaceholder(st.sidebar.empty(), prefix="STATUS: ")
        sidebar_progress = ProgressPlaceholder(st.sidebar.empty())

        sidebar_update.update("Please select dataset and views")

        with st.sidebar.form(key="db_submit"):
            selected_dataset = st.selectbox("Select dataset", ['None'] + self.all_dataset_names, index=settings['db'], key="db_select" )
            selected_dataset_type = st.selectbox("Select dataset type", all_db_types, index=settings['db_type'], key="db_type_select")

            st.write("Hyper-param visualizers:")
            col11, col12, col13 = st.columns(3)
            with col11: show_hiplot = st.checkbox("HiPlot",value='hiplot' in settings['display_views'])
            with col12: show_tableview = st.checkbox("Table",value='table' in settings['display_views'])
            with col13: show_lineplot = st.checkbox("LinePlot",value='line' in settings['display_views'])

            st.write("Detailed hyper-param tools:")
            col21, col22 = st.columns(2)

            with col21: show_hyper_impact = st.checkbox("Impact",value='impact' in settings['display_views'])
            with col22: show_hyper_tree = st.checkbox("DecisionTree",value='tree' in settings['display_views'])

            prefilter_exps = st.checkbox("Pre-filter experiments before loading", value='prefilter' in settings['display_views'])

            col31, col32 = st.columns(2)
            with col31: db_run_button = st.form_submit_button(label='Apply')

            with col32:
                if st.form_submit_button('Clear cache'):
                    legacy_caching.clear_cache()

            display_views = (['hiplot'] if show_hiplot else []) + (['table'] if show_tableview else []) + \
                            (['line'] if show_lineplot else []) + (['impact'] if show_hyper_impact else []) + \
                            (['tree'] if show_hyper_tree else [])

        if selected_dataset != 'None':
            # save settings for URL query params
            new_settings = dict(db=selected_dataset,
                                db_type=selected_dataset_type,
                                display_views=(['hiplot'] if show_hiplot else []) +
                                          (['table'] if show_tableview else []) +
                                          (['line'] if show_lineplot else []) +
                                          (['impact'] if show_hyper_impact else []) +
                                          (['tree'] if show_hyper_tree else []) +
                                          (['prefilter'] if prefilter_exps else []))

            results_structures = self.dataset_def.get_folder_struct(selected_dataset)

            loading_placeholder = st.empty()
            status_placeholder = st.empty()

            if prefilter_exps:
                with st.expander("Pre-filtering of experiments before loading"):
                    results_structures, cfg = views.prefiltering.display_prefiltering(results_structures)
                    new_settings.update(cfg)

            if len(results_structures) > 0:
                loadings_stats = dict()

                sidebar_update.update("Loading results ...")
                try:
                    sidebar_progress.set_num_steps(len(display_views) + 2)
                    sidebar_progress.increase()
                    #with loading_placeholder:
                    with st.spinner('Loading results ...'):
                        start_t = time.time()
                        pool = get_pool()
                        rand_hash = get_random_hash()

                        # add root folder as first part of results_structures
                        results_structures = [self.root_dir] + results_structures

                        # selected_dataset and results_structures must be manually passed as args for caching to properly rerun
                        # when it changes
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

                            with status_placeholder:
                                st.warning("WARNING: No metrics found for dataset '%s' of type '%s'! No matches in subfolders: %s"
                                           % (selected_dataset, selected_dataset_type, first_empty_level))
                            return

                        display_param_list = sorted(list(param_list.keys()))

                        all_variables_str = sorted(["%s=%s" % (k, v) for k, v_list in param_list.items() for v in v_list])

                        end_t = time.time()
                        loadings_stats['num_total_exp'] = len(exp_results)
                        loadings_stats['num_total_params'] = len(param_list)
                        loadings_stats['time_experiments'] = end_t-start_t

                    with status_placeholder:
                        st.success('Successfully loaded %d experiments with %d unique variables (in %.2f sec)!' %
                                   (loadings_stats['num_total_exp'],loadings_stats['num_total_params'],loadings_stats['time_experiments']))

                        sidebar_progress.increase()

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
                        groupby_attr = st.multiselect("Group by ayttribute", display_param_list, default=[a for a in settings['groupby_attr'] if a in display_param_list],
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
                            loadings_stats['num_filtered_exp'] = len(exp_results)
                            loadings_stats['time_filtering'] = end_t-start_t

                        with status_placeholder:
                            st.success('Successfully loaded and retained %d/%d experiments with %d unique variables (in %.2f + %.2f sec)!' %
                                       (loadings_stats['num_filtered_exp'],loadings_stats['num_total_exp'],
                                        loadings_stats['num_filtered_exp'],loadings_stats['time_experiments'],
                                        loadings_stats['time_filtering']))

                    new_settings.update(dict(groupby_attr=param_delim.join(groupby_attr),
                                             groupby_metric=param_delim.join(groupby_metric),
                                             groupby_dbtype=str(groupby_dbtype)))
                    sidebar_progress.increase()

                    plot_mutex = PlotMutex()

                    #######################################################################################
                    if show_hiplot:
                        with st.expander("Hiplot"):
                            sidebar_update.update("Loading HiPlot View ... ")
                            logging.info('Ploting hiplot')
                            cfg = views.hyperplot.display_hiplot(self.parser, exp_results, output_csv_file, param_list, groupby_attr)
                            new_settings.update(cfg)
                            sidebar_progress.increase()

                    #######################################################################################
                    if show_tableview:
                        with st.expander("Table-view"):
                            sidebar_update.update("Loading Table View ... ")
                            logging.info('Show table')
                            cfg = views.tableview.display_tableview(self.parser, exp_results, param_list, all_variables_str)
                            new_settings.update(cfg)
                            sidebar_progress.increase()

                    ########################################################################################
                    if show_lineplot:
                        with st.expander("Line-plot"):
                            sidebar_update.update("Loading Lineplot View ... ")
                            logging.info('Plot lines')
                            cfg = views.lineplot.display_lineplot(self.parser, exp_results, param_list, all_variables_str, groupby_attr, plot_mutex)
                            new_settings.update(cfg)
                            sidebar_progress.increase()

                    ########################################################################################
                    if show_hyper_impact:
                        with st.expander("Hyper-param impact"):
                            sidebar_update.update("Loading Hyper-param Impact View ... ")
                            logging.info('Calculating per hyper-parameter impact')
                            cfg = views.hyperparam_impact.display_hyperparam_impact(self.parser, exp_results, param_list, all_variables_str, groupby_attr, plot_mutex)
                            new_settings.update(cfg)
                            sidebar_progress.increase()

                    ########################################################################################
                    if show_hyper_tree:
                        with st.expander("Hyper-param decision-tree"):
                            sidebar_update.update("Loading DecisionTree View ... ")
                            logging.info('Finding useful decision-trees in hyper-param')
                            cfg = views.decision_trees.display_hyperparam_decisiontrees(self.parser, exp_results, param_list, all_variables_str, groupby_attr, plot_mutex)
                            new_settings.update(cfg)
                            sidebar_progress.increase()

                finally:
                    num_exp = loadings_stats['num_filtered_exp'] if 'num_filtered_exp' in loadings_stats else \
                              loadings_stats['num_total_exp'] if 'num_total_exp' in loadings_stats else 0
                    sidebar_update.update("Ready%s" % (" (using %d experiments)" % num_exp if num_exp > 0 else ""))
                    sidebar_progress.clear()

            st.experimental_set_query_params(**{k: v for k, v in new_settings.items() if v is not None and len(v) > 0})

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
    importlib.reload(views.prefiltering)

    root_dir = sys.argv[1] if len(sys.argv) > 1 else './'
    dataset_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root_dir,'.hypaan')

    datasets = load_user_datasets(dataset_dir)

    # run analyzer
    st_analizer = StreamlitHyperParamAnalyzer(root_dir, datasets)
    st_analizer.run()
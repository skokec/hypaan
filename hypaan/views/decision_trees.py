import streamlit as st

import numpy as np
import pylab as plt

import logging
from tqdm import tqdm

import copy

from sklearn.feature_extraction import DictVectorizer
from sklearn import tree

from collections import OrderedDict

import importlib, settings_manager
importlib.reload(settings_manager)

#@st.cache
def calc_hyperparam_impact(_exp_results, param_key, selected_metrics, param_list, metric_list, metric_list_direction):
    selected_metric_directions = [metric_list_direction[m] for m in selected_metrics]
    param_result = {}

    for val in param_list[param_key]:
        # find experiments that include this val
        positive_exp = [exp for exp in _exp_results if exp[param_key] == val]
        # find experiments that do NOT include this val
        negative_exp = [exp for exp in _exp_results if exp[param_key] != val]

        out_key = '%s=%s' % (param_key, val)

        result_diff = []

        # for every positive exp find matching exp with the same hyperparameter except for selected param
        for pos_exp in positive_exp:
            selected_keys = {k: v for k, v in pos_exp.items() if
                             k != param_key and k[1:] not in metric_list}

            for neg_exp in negative_exp:
                if all([neg_exp[k] == v for k, v in selected_keys.items()]):

                    # copy experiment attributes - except metrics, which are replaced with diff values
                    exp = {k:v for k, v in neg_exp.items() if k[1:] not in metric_list}

                    for d, m in zip(selected_metric_directions, selected_metrics):
                        exp['.' + m] = d * (pos_exp['.' + m] - neg_exp['.' + m])

                    result_diff.append(exp)

        if len(result_diff) > 0:
            param_result[out_key] = result_diff

    return param_result


def compute_and_plot_decision_tree(exp_results, parser_obj, split_ignore_attr, split_metric, max_depth, y_fn, plot_mutex):
    if len(exp_results) <= 0:
        return

    # get list of all params that will be used for decision making
    params = parser_obj.get_available_params(exp_results)

    # ignore those requested by user
    decision_params = [k for k in params.keys() if split_ignore_attr is None or k not in split_ignore_attr]

    metric_keys = ["." + m for m in split_metric]

    # create feature keys from metrics and decision params
    feature_keys = metric_keys + decision_params

    # extract only needed keys from each experiment
    exp_results = [{k:exp[k] for k in feature_keys} for exp in exp_results]

    ### Vectorize
    # conver hyper-params to one-hot encoding numpy array
    vectorize = DictVectorizer(sparse=False)
    dictVector = vectorize.fit_transform(exp_results)

    feature_names = vectorize.get_feature_names()

    Y_raw = np.stack([dictVector[:, feature_names.index('.'+m)] for m in split_metric], axis=1)
    Y = y_fn(Y_raw)

    # remove metrics from final features
    X = dictVector[:, len(metric_keys):]
    feature_names = feature_names[len(metric_keys):]

    ### Retain only one attribute from binary keys
    retained_feat_keys = []
    retained_feat_idx = []
    for i, f in enumerate(feature_names):
        if "=" in f:
            key = f.split("=")[0]
            if len(params[key]) > 2 or key not in retained_feat_keys:
                retained_feat_idx.append(i)
                retained_feat_keys.append(key)

    X = X[:, retained_feat_idx]
    feature_names = [feature_names[i] for i in retained_feat_idx]

    ### Run decision tree
    clf = tree.DecisionTreeRegressor(max_depth=max_depth, criterion="mse")
    clf = clf.fit(X, Y)

    ### Collect mean/std deviations from supporting samples for every decision node
    decision_paths = clf.decision_path(X)

    tree_group = {}
    for p, y in zip(decision_paths, Y_raw):
        for node_ix in p.nonzero()[1]:
            if node_ix not in tree_group:
                tree_group[node_ix] = []
            tree_group[node_ix].append(y)

    tree_group_mean = {}
    tree_group_std = {}
    tree_group_min = {}
    tree_group_max = {}

    for node_id, vals in tree_group.items():
        tree_group_mean[node_id] = {m: v for m, v in zip(split_metric, np.mean(vals, axis=0))}
        tree_group_std[node_id] = {m: v for m, v in zip(split_metric, np.std(vals, axis=0))}
        tree_group_min[node_id] = {m: v for m, v in zip(split_metric, np.min(vals, axis=0))}
        tree_group_max[node_id] = {m: v for m, v in zip(split_metric, np.max(vals, axis=0))}

    tree_group_stat = dict(mean=tree_group_mean,
                           std=tree_group_std,
                           min=tree_group_min,
                           max=tree_group_max)
    ## Plot all results
    st.subheader('Feature importance (gini information):')

    for idx in np.argsort(clf.feature_importances_)[::-1]:
        f = feature_names[idx].split(vectorize.separator)[0]
        st.write('%s' % f, clf.feature_importances_[idx])

    st.subheader('Decision tree:')

    dpi = 100
    #figsize = (2 ** max_depth, (max_depth) * 1.5)

    #fig = plt.figure(figsize=figsize)

    #fig.subplots_adjust(0, 0, 0.98, 0.98)
    #fig.tight_layout()
    with plot_mutex:
        plot_tree(clf, tree_group_stat, filled=True, feature_names=feature_names, fontsize=8)
        fig = plt.gcf()
        fig.subplots_adjust(0, 0, 0.98, 0.98)
        fig.tight_layout()
        figsize = fig.get_size_inches()

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)

        from PIL import Image
        I = Image.open(buf)
        logging.debug("image size=", I.size)
        st.image(buf, width=int(figsize[0] * dpi))



def display_hyperparam_decisiontrees(parser_obj, exp_results, param_list, all_variables_str, existing_group_by, plot_mutex):

    metric_list = parser_obj.metric_list
    metric_list_best_val = parser_obj.metric_list_best_val
    metric_list_direction = parser_obj.metric_list_direction

    display_param_list = sorted(list(param_list.keys()))
    display_groupby_op = ['None','mean','median','max','min']

    query_params = st.experimental_get_query_params()

    settings = settings_manager.InputSettingsManager(param_prefix='tree')

    # set default settings as specified in URL query params
    # first val == default value, second val = parser func from/to str, third val = key in state session, fourth val = valid values
    settings.add_definition(max_depth=(4, settings.INT_FN, 'decisiontree_form_max_depth', None),
                            metrics=([], settings.LIST_FN, 'decisiontree_form_split_metrics', metric_list),
                            conditional_attr=([], settings.LIST_FN, 'decisiontree_form_split_conditional_attr', display_param_list),
                            ignore_attr=([], settings.LIST_FN, 'decisiontree_form_split_ignore_attr', display_param_list),
                            varin=([], settings.LIST_FN, 'decisiontree_form_var_include', all_variables_str),
                            varout_num=(1, settings.INT_FN, 'decisiontree_form_num_exclude_combo', None),
                            varout_exact=(False, settings.BOOL_FN, 'decisiontree_form_var_exclude_exact', None),
                            groupby_attr=([], settings.LIST_FN, 'decisiontree_form_groupby_attr', display_param_list),
                            groupby_metric=([], settings.LIST_FN, 'decisiontree_form_groupby_metric', metric_list),
                            groupby_op=(display_groupby_op.index('None'), settings.CHOICE_FN, 'decisiontree_form_groupby_op', display_groupby_op))

    settings.parse(query_params, st.session_state)

    # there is variable number of varout params so we need to handle them manually
    for i in range(int(settings['varout_num'])):
        settings.add_definition(**{'varout_%d' % i: ([], settings.LIST_FN, "tableview_form_var_exclude_%d" % i, all_variables_str)})

    # parse again with additional values for varout
    settings.parse(query_params, st.session_state)

    ###########################################################################################################
    ## HI-plot
    with st.spinner('Displaying decision trees ...'):
        with st.form(key='decisiontree_form'):

            col01, col02 = st.columns(2)
            col11, col12 = st.columns(2)
            col21, col22 = st.columns(2)
            col31, col32, col33 = st.columns(3)

            with col01:
                max_depth = st.number_input("MAX tree depth:", min_value=1, step=1, **settings.as_streamlit_args('max_depth',value_name='value'))
            with col02:
                split_metrics = st.multiselect("Decision tree metrics:", metric_list, **settings.as_streamlit_args('metrics'))

            with col11:
                split_conditional_attr = st.multiselect("Conditional on specific attribute:", display_param_list, **settings.as_streamlit_args('conditional_attr'))
            with col12:
                split_ignore_attr = st.multiselect("Ignore attributes in decisions:", display_param_list, **settings.as_streamlit_args('ignore_attr'))

            with col21:
                var_include = st.multiselect("Should INCLUDE all those variables:", all_variables_str, **settings.as_streamlit_args('varin'))
            with col22:
                num_exclude_combo = st.number_input("Number combinations", min_value=1, step=1, **settings.as_streamlit_args('varout_num',value_name='value'))

                var_exclude_list = []
                for i in range(int(num_exclude_combo)):
                    var_exclude_list.append(st.multiselect("Should EXCLUDE any of those variables (%d):" % i, all_variables_str,  **settings.as_streamlit_args('varout_%d' % i)))
                var_exclude_exact = st.checkbox("EXCLUDE exact combination only", **settings.as_streamlit_args('varout_exact',value_name='value'))

            with col31:
                groupby_attr = st.multiselect("Group by attribute", display_param_list, **settings.as_streamlit_args('groupby_attr'))
            with col32:
                groupby_metric = st.multiselect("Metric:", metric_list, **settings.as_streamlit_args('groupby_metric'))
            with col33:
                groupby_op = st.selectbox("Modify grouped metrics to:", display_groupby_op, **settings.as_streamlit_args('groupby_op',value_name='index'))

            submit_button = st.form_submit_button(label='RUN')

        # save settings for URL query params
        new_settings = dict(max_depth=max_depth,
                            metrics=split_metrics,
                            conditional_attr=split_conditional_attr,
                            ignore_attr=split_ignore_attr,
                            varin=var_include,
                            varout_num=int(num_exclude_combo),
                            varout_exact=var_exclude_exact,
                            groupby_attr=groupby_attr,
                            groupby_metric=groupby_metric,
                            groupby_op=groupby_op)

        # manually add varout since there can be multiple vars
        new_settings.update({'varout_%d' % i: var_exclude_list[i] for i in range(int(num_exclude_combo))})

        # finalize new settings
        new_settings = settings.compile_new_settings(**new_settings)

        if submit_button:
            exp_results = copy.deepcopy(exp_results)

            # manually remove any vars that were previously grouped
            if existing_group_by is not None:
                exp_results = [{k: v for k,v in exp.items() if k not in existing_group_by} for exp in exp_results]

            # remove experiments that contain explicitly removed variables
            if var_exclude_list is not None :
                for var_exclude in var_exclude_list:
                    if len(var_exclude) > 0:
                        exp_results = parser_obj.experiment_exclude_vars(exp_results, var_exclude, only_exact_combination=var_exclude_exact)

            # retain experiments that contain only explicitly requested variables
            if len(var_include) > 0:
                # split var_include into dictionary
                exp_results = parser_obj.experiment_retain_only_vars(exp_results, var_include)

            if groupby_attr is not None and groupby_metric is not None:
                best_by_modify_op = None
                if groupby_op != 'None':
                    if groupby_op == 'mean':
                        best_by_modify_op = np.mean
                    elif groupby_op == 'median':
                        best_by_modify_op = np.median
                    elif groupby_op == 'max':
                        best_by_modify_op = np.max
                    elif groupby_op == 'min':
                        best_by_modify_op = np.min

                exp_results = parser_obj.experiment_group_by(exp_results, groupby_attr, groupby_metric, best_by_modify_op=best_by_modify_op,
                                                             remove_gruped_attributes=True)

            st.header("Unconditional on all attributes (filtered by %s) :" % "  ".join(var_include))

            get_y_fn = lambda Y_raw: np.sqrt(np.sum((Y_raw -  np.array([metric_list_best_val[m] for m in split_metrics]).reshape(1,-1)) ** 2, axis=1))

            compute_and_plot_decision_tree(exp_results, parser_obj, split_ignore_attr, split_metrics, max_depth, get_y_fn, plot_mutex)

            if split_conditional_attr is not None and len(split_conditional_attr) > 0:
                get_y_fn = lambda Y_raw: Y_raw[:, 0]

                for cond_attr in split_conditional_attr:
                    exp_results_list = calc_hyperparam_impact(exp_results, cond_attr, split_metrics,
                                                         param_list, metric_list, metric_list_direction)
                    if len(exp_results_list) == 2:
                        retain_keys = [k for k in exp_results_list.keys() if k.split("=")[1] not in ['off','false','False','0']]

                        exp_results_list = {k:v for k,v in exp_results_list.items() if k in retain_keys}

                    for conditional_attr, exp_results_cond in exp_results_list.items():
                        st.header("Conditional on %s (filtered by %s)" % (conditional_attr, "  ".join(var_include)))
                        compute_and_plot_decision_tree(exp_results_cond, parser_obj, split_ignore_attr, split_metrics, max_depth, get_y_fn, plot_mutex)

            st.success('Done!')

    return new_settings

from numbers import Integral

from sklearn.tree import _criterion
from sklearn.tree import _tree
from sklearn.tree._reingold_tilford import buchheim, Tree

from sklearn.utils.validation import check_is_fitted

from sklearn.tree._export import _color_brew, SENTINEL

import warnings


def plot_tree(decision_tree, tree_group_stat, *, max_depth=None, feature_names=None,
              class_names=None, label='all', filled=False,
              impurity=True, node_ids=False,
              proportion=False, rotate='deprecated', rounded=False,
              precision=3, ax=None, fontsize=None):

    check_is_fitted(decision_tree)

    if rotate != 'deprecated':
        warnings.warn(("'rotate' has no effect and is deprecated in 0.23. "
                       "It will be removed in 1.0 (renaming of 0.25)."),
                      FutureWarning)

    exporter = _MPLTreeExporter(
        max_depth=max_depth, feature_names=feature_names,
        class_names=class_names, label=label, filled=filled,
        impurity=impurity, node_ids=node_ids,
        proportion=proportion, rotate=rotate, rounded=rounded,
        precision=precision, fontsize=fontsize)
    exporter.characters[3] = ''
    return exporter.export(decision_tree, tree_group_stat, ax=ax)


class _BaseTreeExporter:
    def __init__(self, max_depth=None, feature_names=None,
                 class_names=None, label='all', filled=False,
                 impurity=True, node_ids=False,
                 proportion=False, rotate=False, rounded=False,
                 precision=3, fontsize=None):
        self.max_depth = max_depth
        self.feature_names = feature_names
        self.class_names = class_names
        self.label = label
        self.filled = filled
        self.impurity = impurity
        self.node_ids = node_ids
        self.proportion = proportion
        self.rotate = rotate
        self.rounded = rounded
        self.precision = precision
        self.fontsize = fontsize

    def get_color(self, value):
        # Find the appropriate color & intensity for a node
        if self.colors['bounds'] is None:
            # Classification tree
            color = list(self.colors['rgb'][np.argmax(value)])
            sorted_values = sorted(value, reverse=True)
            if len(sorted_values) == 1:
                alpha = 0
            else:
                alpha = ((sorted_values[0] - sorted_values[1])
                         / (1 - sorted_values[1]))
        else:
            # Regression tree or multi-output
            color = list(self.colors['rgb'][0])
            alpha = ((value - self.colors['bounds'][0]) /
                     (self.colors['bounds'][1] - self.colors['bounds'][0]))
        # unpack numpy scalars
        alpha = float(alpha)
        # compute the color as alpha against white
        color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
        # Return html color code in #RRGGBB format
        return '#%2x%2x%2x' % tuple(color)

    def get_fill_color(self, tree, node_id):
        # Fetch appropriate color for node
        if 'rgb' not in self.colors:
            # Initialize colors and bounds if required
            self.colors['rgb'] = _color_brew(tree.n_classes[0])
            if tree.n_outputs != 1:
                # Find max and min impurities for multi-output
                self.colors['bounds'] = (np.min(-tree.impurity),
                                         np.max(-tree.impurity))
            elif (tree.n_classes[0] == 1 and
                  len(np.unique(tree.value)) != 1):
                # Find max and min values in leaf nodes for regression
                self.colors['bounds'] = (np.min(tree.value),
                                         np.max(tree.value))
        if tree.n_outputs == 1:
            node_val = (tree.value[node_id][0, :] /
                        tree.weighted_n_node_samples[node_id])
            if tree.n_classes[0] == 1:
                # Regression
                node_val = tree.value[node_id][0, :]
        else:
            # If multi-output color node by impurity
            node_val = -tree.impurity[node_id]
        return self.get_color(node_val)

    def node_to_str(self, tree, tree_group_stat, node_id, criterion):
        # Generate the node content string
        if tree.n_outputs == 1:
            value = tree.value[node_id][0, :]
        else:
            value = tree.value[node_id]

        # Should labels be shown?
        labels = (self.label == 'root' and node_id == 0) or self.label == 'all'

        characters = self.characters
        node_string = characters[-1]

        # Write node ID
        if self.node_ids:
            if labels:
                node_string += 'node '
            node_string += characters[0] + str(node_id) + characters[4]

        # Write decision criteria
        if tree.children_left[node_id] != _tree.TREE_LEAF:
            # Always write node decision criteria, except for leaves
            if self.feature_names is not None:
                feature = self.feature_names[tree.feature[node_id]]
            else:
                feature = "X%s%s%s" % (characters[1],
                                       tree.feature[node_id],
                                       characters[2])
            if "=" in feature:
                node_string += feature + characters[4]
            else:
                node_string += '%s %s %s%s' % (feature,
                                                characters[3],
                                               round(tree.threshold[node_id],
                                                     self.precision),
                                               characters[4])

        # Write impurity
        if self.impurity:
            if isinstance(criterion, _criterion.FriedmanMSE):
                criterion = "friedman_mse"
            elif not isinstance(criterion, str):
                criterion = "impurity"
            if labels:
                node_string += '%s = ' % criterion
            node_string += (str(round(tree.impurity[node_id], self.precision))
                            + characters[4])

        # Write node sample count
        if labels:
            node_string += 'samples = '
        if self.proportion:
            percent = (100. * tree.n_node_samples[node_id] /
                       float(tree.n_node_samples[0]))
            node_string += (str(round(percent, 1)) + '%' +
                            characters[4])
        else:
            node_string += (str(tree.n_node_samples[node_id]) +
                            characters[4])

        # Write node class distribution / regression value
        if self.proportion and tree.n_classes[0] != 1:
            # For classification this will show the proportion of samples
            value = value / tree.weighted_n_node_samples[node_id]
        if labels:
            node_string += 'value = '
        if tree.n_classes[0] == 1:
            # Regression
            value_text = np.around(value, self.precision)
        elif self.proportion:
            # Classification
            value_text = np.around(value, self.precision)
        elif np.all(np.equal(np.mod(value, 1), 0)):
            # Classification without floating-point weights
            value_text = value.astype(int)
        else:
            # Classification with floating-point weights
            value_text = np.around(value, self.precision)
        # Strip whitespace
        value_text = str(value_text.astype('S32')).replace("b'", "'")
        value_text = value_text.replace("' '", ", ").replace("'", "")
        if tree.n_classes[0] == 1 and tree.n_outputs == 1:
            value_text = value_text.replace("[", "").replace("]", "")
        value_text = value_text.replace("\n ", characters[4])
        node_string += value_text + characters[4]

        # Write node majority class
        if (self.class_names is not None and
                tree.n_classes[0] != 1 and
                tree.n_outputs == 1):
            # Only done for single-output classification trees
            if labels:
                node_string += 'class = '
            if self.class_names is not True:
                class_name = self.class_names[np.argmax(value)]
            else:
                class_name = "y%s%s%s" % (characters[1],
                                          np.argmax(value),
                                          characters[2])
            node_string += class_name

        if tree_group_stat is not None:
            node_string += "---------------------------------%s" % characters[4]
            for k in tree_group_stat['mean'][node_id].keys():
                node_string += "%s = %.2f+/-%.2f (%.2f,%.2f)%s" % (k,
                                                                 tree_group_stat['mean'][node_id][k],
                                                                 tree_group_stat['std'][node_id][k],
                                                                 tree_group_stat['min'][node_id][k],
                                                                 tree_group_stat['max'][node_id][k],
                                                                 characters[4])

        # Clean up any trailing newlines
        if node_string.endswith(characters[4]):
            node_string = node_string[:-len(characters[4])]

        return node_string + characters[5]


class _DOTTreeExporter(_BaseTreeExporter):
    def __init__(self, out_file=SENTINEL, max_depth=None,
                 feature_names=None, class_names=None, label='all',
                 filled=False, leaves_parallel=False, impurity=True,
                 node_ids=False, proportion=False, rotate=False, rounded=False,
                 special_characters=False, precision=3):

        super().__init__(
            max_depth=max_depth, feature_names=feature_names,
            class_names=class_names, label=label, filled=filled,
            impurity=impurity,
            node_ids=node_ids, proportion=proportion, rotate=rotate,
            rounded=rounded,
            precision=precision)
        self.leaves_parallel = leaves_parallel
        self.out_file = out_file
        self.special_characters = special_characters

        # PostScript compatibility for special characters
        if special_characters:
            self.characters = ['&#35;', '<SUB>', '</SUB>', '&le;', '<br/>',
                               '>', '<']
        else:
            self.characters = ['#', '[', ']', '<=', '\\n', '"', '"']

        # validate
        if isinstance(precision, Integral):
            if precision < 0:
                raise ValueError("'precision' should be greater or equal to 0."
                                 " Got {} instead.".format(precision))
        else:
            raise ValueError("'precision' should be an integer. Got {}"
                             " instead.".format(type(precision)))

        # The depth of each node for plotting with 'leaf' option
        self.ranks = {'leaves': []}
        # The colors to render each node with
        self.colors = {'bounds': None}

    def export(self, decision_tree):
        # Check length of feature_names before getting into the tree node
        # Raise error if length of feature_names does not match
        # n_features_ in the decision_tree
        if self.feature_names is not None:
            if len(self.feature_names) != decision_tree.n_features_:
                raise ValueError("Length of feature_names, %d "
                                 "does not match number of features, %d"
                                 % (len(self.feature_names),
                                    decision_tree.n_features_))
        # each part writes to out_file
        self.head()
        # Now recurse the tree and add node & edge attributes
        if isinstance(decision_tree, _tree.Tree):
            self.recurse(decision_tree, 0, criterion="impurity")
        else:
            self.recurse(decision_tree.tree_, 0,
                         criterion=decision_tree.criterion)

        self.tail()

    def tail(self):
        # If required, draw leaf nodes at same depth as each other
        if self.leaves_parallel:
            for rank in sorted(self.ranks):
                self.out_file.write(
                    "{rank=same ; " +
                    "; ".join(r for r in self.ranks[rank]) + "} ;\n")
        self.out_file.write("}")

    def head(self):
        self.out_file.write('digraph Tree {\n')

        # Specify node aesthetics
        self.out_file.write('node [shape=box')
        rounded_filled = []
        if self.filled:
            rounded_filled.append('filled')
        if self.rounded:
            rounded_filled.append('rounded')
        if len(rounded_filled) > 0:
            self.out_file.write(
                ', style="%s", color="black"'
                % ", ".join(rounded_filled))
        if self.rounded:
            self.out_file.write(', fontname=helvetica')
        self.out_file.write('] ;\n')

        # Specify graph & edge aesthetics
        if self.leaves_parallel:
            self.out_file.write(
                'graph [ranksep=equally, splines=polyline] ;\n')
        if self.rounded:
            self.out_file.write('edge [fontname=helvetica] ;\n')
        if self.rotate:
            self.out_file.write('rankdir=LR ;\n')

    def recurse(self, tree, node_id, criterion, parent=None, depth=0):
        if node_id == _tree.TREE_LEAF:
            raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        # Add node with description
        if self.max_depth is None or depth <= self.max_depth:

            # Collect ranks for 'leaf' option in plot_options
            if left_child == _tree.TREE_LEAF:
                self.ranks['leaves'].append(str(node_id))
            elif str(depth) not in self.ranks:
                self.ranks[str(depth)] = [str(node_id)]
            else:
                self.ranks[str(depth)].append(str(node_id))

            self.out_file.write(
                '%d [label=%s' % (node_id, self.node_to_str(tree, node_id,
                                                            criterion)))

            if self.filled:
                self.out_file.write(', fillcolor="%s"'
                                    % self.get_fill_color(tree, node_id))
            self.out_file.write('] ;\n')

            if parent is not None:
                # Add edge to parent
                self.out_file.write('%d -> %d' % (parent, node_id))
                if parent == 0:
                    # Draw True/False labels if parent is root node
                    angles = np.array([45, -45]) * ((self.rotate - .5) * -2)
                    self.out_file.write(' [labeldistance=2.5, labelangle=')
                    if node_id == 1:
                        self.out_file.write('%d, headlabel="True"]' %
                                            angles[0])
                    else:
                        self.out_file.write('%d, headlabel="False"]' %
                                            angles[1])
                self.out_file.write(' ;\n')

            if left_child != _tree.TREE_LEAF:
                self.recurse(tree, left_child, criterion=criterion,
                             parent=node_id, depth=depth + 1)
                self.recurse(tree, right_child, criterion=criterion,
                             parent=node_id, depth=depth + 1)

        else:
            self.ranks['leaves'].append(str(node_id))

            self.out_file.write('%d [label="(...)"' % node_id)
            if self.filled:
                # color cropped nodes grey
                self.out_file.write(', fillcolor="#C0C0C0"')
            self.out_file.write('] ;\n' % node_id)

            if parent is not None:
                # Add edge to parent
                self.out_file.write('%d -> %d ;\n' % (parent, node_id))


class _MPLTreeExporter(_BaseTreeExporter):
    def __init__(self, max_depth=None, feature_names=None,
                 class_names=None, label='all', filled=False,
                 impurity=True, node_ids=False,
                 proportion=False, rotate=False, rounded=False,
                 precision=3, fontsize=None):

        super().__init__(
            max_depth=max_depth, feature_names=feature_names,
            class_names=class_names, label=label, filled=filled,
            impurity=impurity, node_ids=node_ids, proportion=proportion,
            rotate=rotate, rounded=rounded, precision=precision)
        self.fontsize = fontsize

        # validate
        if isinstance(precision, Integral):
            if precision < 0:
                raise ValueError("'precision' should be greater or equal to 0."
                                 " Got {} instead.".format(precision))
        else:
            raise ValueError("'precision' should be an integer. Got {}"
                             " instead.".format(type(precision)))

        # The depth of each node for plotting with 'leaf' option
        self.ranks = {'leaves': []}
        # The colors to render each node with
        self.colors = {'bounds': None}

        self.characters = ['#', '[', ']', '<=', '\n', '', '']
        self.bbox_args = dict()
        if self.rounded:
            self.bbox_args['boxstyle'] = "round"

        self.arrow_args = dict(arrowstyle="<-")

    def _make_tree(self, node_id, et, tree_group_stat, criterion, depth=0):
        # traverses _tree.Tree recursively, builds intermediate
        # "_reingold_tilford.Tree" object
        name = self.node_to_str(et, tree_group_stat, node_id, criterion=criterion)
        if (et.children_left[node_id] != _tree.TREE_LEAF
                and (self.max_depth is None or depth <= self.max_depth)):
            children = [self._make_tree(et.children_left[node_id], et, tree_group_stat,
                                        criterion, depth=depth + 1),
                        self._make_tree(et.children_right[node_id], et, tree_group_stat,
                                        criterion, depth=depth + 1)]
        else:
            return Tree(name, node_id)
        return Tree(name, node_id, *children)

    def export(self, decision_tree, tree_group_stat, ax=None):
        import matplotlib.pyplot as plt
        from matplotlib.text import Annotation

        my_tree = self._make_tree(0, decision_tree.tree_, tree_group_stat,
                                  decision_tree.criterion)
        draw_tree = buchheim(my_tree)

        if ax is None:
            max_x, max_y = draw_tree.max_extents() + 1
            figsize = (max_x * 2, (max_y) * 1.5)

            fig = plt.figure(figsize=figsize)

            fig.subplots_adjust(0, 0, 0.98, 0.98)
            fig.tight_layout()

            ax = plt.gca()

        ax.clear()
        ax.set_axis_off()

        # important to make sure we're still
        # inside the axis after drawing the box
        # this makes sense because the width of a box
        # is about the same as the distance between boxes
        max_x, max_y = draw_tree.max_extents() + 1
        ax_width = ax.get_window_extent().width
        ax_height = ax.get_window_extent().height

        scale_x = ax_width / max_x
        scale_y = ax_height / max_y

        self.recurse(draw_tree, decision_tree.tree_, ax,
                     scale_x, scale_y, ax_height)

        anns = [ann for ann in ax.get_children()
                if isinstance(ann, Annotation)]

        # update sizes of all bboxes
        renderer = ax.figure.canvas.get_renderer()

        for ann in anns:
            ann.update_bbox_position_size(renderer)

        if self.fontsize is None:
            # get figure to data transform
            # adjust fontsize to avoid overlap
            # get max box width and height
            extents = [ann.get_bbox_patch().get_window_extent()
                       for ann in anns]
            max_width = max([extent.width for extent in extents])
            max_height = max([extent.height for extent in extents])
            # width should be around scale_x in axis coordinates
            size = anns[0].get_fontsize() * min(scale_x / max_width,
                                                scale_y / max_height)
            for ann in anns:
                ann.set_fontsize(size)

        return anns

    def recurse(self, node, tree, ax, scale_x, scale_y, height, depth=0):
        import matplotlib.pyplot as plt
        kwargs = dict(bbox=self.bbox_args.copy(), ha='center', va='center',
                      zorder=100 - 10 * depth, xycoords='axes pixels',
                      arrowprops=self.arrow_args.copy())
        kwargs['arrowprops']['edgecolor'] = plt.rcParams['text.color']

        if self.fontsize is not None:
            kwargs['fontsize'] = self.fontsize

        # offset things by .5 to center them in plot
        xy = ((node.x + .5) * scale_x, height - (node.y + .5) * scale_y)

        if self.max_depth is None or depth <= self.max_depth:
            if self.filled:
                kwargs['bbox']['fc'] = self.get_fill_color(tree,
                                                           node.tree.node_id)
            else:
                kwargs['bbox']['fc'] = ax.get_facecolor()

            if node.parent is None:
                # root
                ax.annotate(node.tree.label, xy, **kwargs)
            else:
                xy_parent = ((node.parent.x + .5) * scale_x,
                             height - (node.parent.y + .5) * scale_y)
                ax.annotate(node.tree.label, xy_parent, xy, **kwargs)
            for child in node.children:
                self.recurse(child, tree, ax, scale_x, scale_y, height,
                             depth=depth + 1)

        else:
            xy_parent = ((node.parent.x + .5) * scale_x,
                         height - (node.parent.y + .5) * scale_y)
            kwargs['bbox']['fc'] = 'grey'
            ax.annotate("\n  (...)  \n", xy_parent, xy, **kwargs)
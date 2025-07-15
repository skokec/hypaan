import copy
import time

import streamlit as st

import numpy as np

import pandas as pd
import plotly.express as px

from stqdm import stqdm as tqdm

import importlib, settings_manager
importlib.reload(settings_manager)

@st.cache
def calc_hyperparam_impact(_exp_results, param_key, selected_metrics, param_list, metric_list, metric_list_direction,
                           ignore_attr_list=[]):
    selected_metric_directions = [metric_list_direction[m] for m in selected_metrics]
    param_result = {}
    param_vals = param_list[param_key]

    for val in param_vals:
        # find experiments that include this val
        positive_exp = [exp for exp in _exp_results if param_key in exp and exp[param_key] == val]
        # find experiments that do NOT include this val
        negative_exp = [exp for exp in _exp_results if param_key in exp and exp[param_key] != val]

        out_key = '%s=%s' % (param_key, val)

        result_diff = {m: [] for m in selected_metrics}
        result_diff['selected_keys'] = []

        # for every positive exp find matching exp with the same hyperparameter except for the selected param
        for pos_exp in positive_exp:
            selected_keys = {k: v for k, v in pos_exp.items() if
                             k != param_key and k[1:] not in metric_list and k not in ignore_attr_list}

            for neg_exp in negative_exp:
                if all([neg_exp[k] == v for k, v in selected_keys.items()]):
                    for d, m in zip(selected_metric_directions, selected_metrics):
                        diff = d * (pos_exp['.' + m] - neg_exp['.' + m])

                        result_diff[m].append(diff)
                    added_selected_keys = {param_key: neg_exp[param_key]}
                    added_selected_keys.update(selected_keys)
                    result_diff['selected_keys'].append(added_selected_keys)

        if any([len(diff) for diff in result_diff.values()]):
            param_result[out_key] = result_diff

    return param_result

def display_hyperparam_impact(parser_obj, exp_results, param_list, all_variables_str, existing_group_by, plot_mutex):

    metric_list = parser_obj.metric_list
    metric_list_direction = parser_obj.metric_list_direction

    display_param_list = sorted(list(param_list.keys()))
    display_groupby_op = ['None','mean','median','max','min']

    display_param_list_plus_all = ['All'] + display_param_list
    display_param_list_plus_none = ['-----'] + display_param_list

    query_params = st.experimental_get_query_params()

    settings = settings_manager.InputSettingsManager(param_prefix='hyperimpact')

    # set default settings as specified in URL query params
    # first val == default value, second val = parser func from/to str, third val = key in state session, fourth val = valid values
    settings.add_definition(vars=([], settings.LIST_FN, 'hyperparam_form_plot_variables',display_param_list_plus_all),
                            metrics=([], settings.LIST_FN, 'hyperparam_form_show_metrics', metric_list),
                            ignore_attr=([], settings.LIST_FN, 'hyperparam_form_ignore_attr', display_param_list),
                            varin=([], settings.LIST_FN, 'hyperparam_form_var_include', all_variables_str),
                            varout_num=(1, settings.INT_FN, 'hyperparam_form_num_exclude_combo', None),
                            varout_exact=(False, settings.BOOL_FN, 'hyperparam_form_var_exclude_exact', None),
                            colors=(0, settings.CHOICE_FN, 'hyperparam_form_colors_by', display_param_list_plus_none),
                            marker=(0, settings.CHOICE_FN, 'hyperparam_form_marker_by', display_param_list_plus_none),
                            marker_size=(0, settings.CHOICE_FN, 'hyperparam_form_marker_size_by', display_param_list_plus_none),
                            groupby_attr=([], settings.LIST_FN, 'hyperparam_form_groupby_attr', display_param_list),
                            groupby_metric=([], settings.LIST_FN, 'hyperparam_form_groupby_metric', metric_list),
                            groupby_op=(display_groupby_op.index('None'), settings.CHOICE_FN, 'hyperparam_form_groupby_op', display_groupby_op))

    settings.parse(query_params, st.session_state)

    # there is variable number of varout params so we need to handle them manually
    for i in range(int(settings['varout_num'])):
        settings.add_definition(**{'varout_%d' % i: ([], settings.LIST_FN, "decisiontree_form_var_exclude_%d" % i, all_variables_str)})

    # parse again with additional values for varout
    settings.parse(query_params, st.session_state)

    ###########################################################################################################
    ## Hyper-param impact
    with st.form(key='hyperparam_form'):
        col11, col12 = st.columns(2)
        col21, col22 = st.columns(2)
        col31, col32, col33 = st.columns(3)
        with col11: plot_variables = st.multiselect("Show individual hyper-parameter impact", display_param_list_plus_all, **settings.as_streamlit_args('vars'))
        with col12: show_metrics = st.multiselect("Metrics:", metric_list, **settings.as_streamlit_args('metrics'))
        with col21:
            ignore_attr = st.multiselect("Ignore attributes for comparison:", display_param_list, **settings.as_streamlit_args('ignore_attr'))
            var_include = st.multiselect("Should INCLUDE all those variables:", all_variables_str, **settings.as_streamlit_args('varin'))
        with col22:
            num_exclude_combo = st.number_input("Number combinations", min_value=1, step=1, **settings.as_streamlit_args('varout_num',value_name='value'))

            var_exclude_list = []
            for i in range(int(num_exclude_combo)):
                var_exclude_list.append(
                    st.multiselect("Should EXCLUDE any of those variables (%d):" % i, all_variables_str, **settings.as_streamlit_args('varout_%d' % i)))
            var_exclude_exact = st.checkbox("EXCLUDE exact combination only", **settings.as_streamlit_args('varout_exact',value_name='value'))

        with col31: colors_by = st.selectbox("Select color by:", display_param_list_plus_none, **settings.as_streamlit_args('colors',value_name='index'))
        with col32: marker_by = st.selectbox("Select marker by:", display_param_list_plus_none, **settings.as_streamlit_args('marker',value_name='index'))
        with col33: marker_size_by = st.selectbox("Select marker size by:", display_param_list_plus_none, **settings.as_streamlit_args('marker_size',value_name='index'))

        with col31: groupby_attr = st.multiselect("Group by attribute", display_param_list, **settings.as_streamlit_args('groupby_attr'))
        with col32: groupby_metric = st.multiselect("Metric for grouping by attribute:", metric_list, **settings.as_streamlit_args('groupby_metric'))
        with col33: groupby_op = st.selectbox("Modify grouped metrics to:", display_groupby_op, **settings.as_streamlit_args('groupby_op',value_name='index'))

        submit_button = st.form_submit_button(label='Analyze hyper-param impact')

    # save settings for URL query params
    new_settings = dict(vars=plot_variables,
                        metrics=show_metrics,
                        ignore_attr=ignore_attr,
                        varin=var_include,
                        varout_num=int(num_exclude_combo),
                        varout_exact=var_exclude_exact,
                        colors=colors_by,
                        marker=marker_by,
                        marker_size=marker_size_by,
                        groupby_attr=groupby_attr,
                        groupby_metric=groupby_metric,
                        groupby_op=groupby_op)

    # manually add varout since there can be multiple vars
    new_settings.update({'varout_%d' % i: var_exclude_list[i] for i in range(int(num_exclude_combo))})

    # finalize new settings
    new_settings = settings.compile_new_settings(**new_settings)

    if plot_variables is not None and len(plot_variables) > 0:
        with st.spinner('Calculating hyper-parameter impact ...'):
            param_result = {}
            if 'All' in plot_variables:
                plot_variables = sorted(list(param_list.keys()))

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
                    elif groupby_op ==  'min':
                        best_by_modify_op = np.min
                exp_results = parser_obj.experiment_group_by(exp_results, groupby_attr, groupby_metric, best_by_modify_op,
                                                             remove_gruped_attributes=True)

            # manually remove any vars that were previously grouped
            if existing_group_by is not None:
                exp_results = [{k: v for k,v in exp.items() if k not in existing_group_by} for exp in exp_results]

            selected_metrics = [m for m in metric_list if m in show_metrics]

            # remove attributes that are always the same
            df = pd.DataFrame(exp_results)
            nunique = df.nunique()
            cols_to_drop = nunique[nunique == 1].index
            exp_results = df.drop(cols_to_drop, axis=1).to_dict(orient='records')
            unique_params = [l for l in list(nunique[nunique > 1].index) if l[1:] not in metric_list]

            # for every hyper-parameter value find performance impact
            all_p = []

            for ii, param_key in enumerate(tqdm(plot_variables)):
                p = calc_hyperparam_impact(exp_results, param_key, selected_metrics,
                                       param_list, metric_list, metric_list_direction,
                                       ignore_attr)
                param_result.update(p)

                for k,e in p.items():
                    keys = pd.DataFrame(e['selected_keys'])

                    item_p = []

                    for m, v in e.items():
                        if m != 'selected_keys':
                            data = {'diff':v,
                                    'metric':[m] * len(v),
                                    'key': [k] * len(v),
                                    'default_size': [3] * len(v)}
                            item_p.append(pd.concat([pd.DataFrame(data),keys], axis=1))
                    all_p.append(pd.concat(item_p,axis=0))

            all_p = pd.concat(all_p,axis=0)

            all_mean = all_p[['diff', 'key', 'metric']].groupby(['key','metric'], as_index=False).mean()
            all_std = all_p[['diff', 'key', 'metric']].groupby(['key', 'metric'], as_index=False).std()

            num_param_inspect = len(param_result)
            if num_param_inspect > 0:
                colors_by = colors_by if colors_by != display_param_list_plus_none[0] else None
                marker_by = marker_by if marker_by != display_param_list_plus_none[0] else None
                marker_size_by = marker_size_by if marker_size_by != display_param_list_plus_none[0] else None

                progres_res = tqdm(param_result.items())
                cols = st.columns(2)
                i = 0
                for k, m in progres_res:
                    progres_res.set_postfix(attribute="'%s'" % k)
                    with cols[i%len(cols)]:
                        if marker_size_by is not None:
                            marker_size_vals = np.array(all_p.loc[all_p['key'] == k][marker_size_by])
                            if marker_size_vals.dtype.type is np.string_:
                                marker_size_vals = marker_size_vals.astype(np.float32)
                            else:
                                marker_key_vals = sorted(np.unique(marker_size_vals))
                                marker_size_vals = [(marker_key_vals.index(m)+1) for m in marker_size_vals]
                        else:
                            marker_size_vals = 'default_size'
                        colors_by_curr = colors_by if colors_by is not None else None
                        marker_by_curr = marker_by if marker_by is not None else None
                        fig = px.scatter(all_p.loc[all_p['key'] == k], x='metric', y='diff',
                                         color=colors_by_curr, symbol=marker_by_curr, size=marker_size_vals, size_max=10 if marker_size_vals == 'default_size' else 25,
                                         hover_data=list(set(unique_params)-{'default_size'}),
                                         color_discrete_sequence=px.colors.qualitative.D3,
                                         title="<b>%s</b><br>%s" % (k, "  ".join(var_include)), height=786)
                        fig.add_bar(x=all_mean.loc[all_mean['key'] == k]['metric'], y=all_mean.loc[all_mean['key'] == k]['diff'],
                                    showlegend=False, marker=dict(color=px.colors.qualitative.Pastel2[0]),
                                    error_y=dict(array=all_std.loc[all_std['key'] == k]['diff'],type='data',visible=True,
                                                 thickness=4,width=20))
                        fig.update_layout(
                            # legend=dict(
                            #     orientation="h",
                            #     yanchor="bottom",
                            #     y=1.02,
                            #     xanchor="right",
                            #     x=1
                            # ),
                        title={
                                'y': 0.92,
                                'x': 0.5,
                                'xanchor': 'center',
                                'yanchor': 'top',
                                'font': dict(size=20)
                            },
                            yaxis_title='',
                            yaxis_tickfont_size=16,
                            xaxis_tickfont_size=16,
                            yaxis_titlefont_size=22,
                            xaxis_titlefont_size=18,
                        )
                        st.plotly_chart(fig,use_container_width=True)
                    i += 1

            st.success('Done!')

    return new_settings
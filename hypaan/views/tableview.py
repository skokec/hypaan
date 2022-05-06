import copy

import streamlit as st

import numpy as np
import pylab as plt

import parser

import importlib
importlib.reload(parser)

import numpy as np
from scipy.ndimage.filters import uniform_filter1d

def moving_average(x, w):
    return uniform_filter1d(x, size=w)

import importlib, settings_manager
importlib.reload(settings_manager)

def display_tableview(parser_obj, exp_results, param_list, all_variables_str):

    metric_list = parser_obj.metric_list

    display_param_list = sorted(list(param_list.keys()))

    query_params = st.experimental_get_query_params()

    settings = settings_manager.InputSettingsManager(param_prefix='tableview')

    # set default settings as specified in URL query params
    # first val == default value, second val = parser func from/to str, third val = key in state session, fourth val = valid values
    settings.add_definition(varin=([], settings.LIST_FN, 'tableview_form_var_include', all_variables_str),
                            varout_num=(1, settings.INT_FN, 'tableview_form_num_exclude_combo', None),
                            varout_exact=(False, settings.BOOL_FN, 'tableview_var_exclude_exact', None),
                            groupby_attr=([], settings.LIST_FN, 'tableview_form_groupby_attr', display_param_list),
                            groupby_metric=([], settings.LIST_FN, 'tableview_form_groupby_metric', metric_list),)

    settings.parse(query_params, st.session_state)

    # there is variable number of varout params so we need to handle them manually
    for i in range(int(settings['varout_num'])):
        settings.add_definition(**{'varout_%d' % i: ([], settings.LIST_FN, "tableview_form_var_exclude_%d" % i, all_variables_str)})

    # parse again with additional values for varout
    settings.parse(query_params, st.session_state)

    ###########################################################################################################
    ## Line plot
    with st.form(key='tableview_form'):
        col21, col22 = st.columns(2)
        col31, col32 = st.columns(2)

        with col21:
            var_include = st.multiselect("Should INCLUDE all those variables:", all_variables_str, **settings.as_streamlit_args('varin'))
        with col22:
            num_exclude_combo = st.number_input("Number combinations", min_value=1, step=1, **settings.as_streamlit_args('varout_num',value_name='value'))

            var_exclude_list = []
            for i in range(int(num_exclude_combo)):
                var_exclude_list.append( st.multiselect("Should EXCLUDE any of those variables (%d):" % i, all_variables_str, key="tableview_form_var_exclude_%d" % i,
                                                        default=[v for v in settings['varout_%d' % i] if v in all_variables_str]))
            var_exclude_exact = st.checkbox("EXCLUDE exact combination only", **settings.as_streamlit_args('varout_exact',value_name='value'))

        with col31:
            groupby_attr = st.multiselect("Group by attribute", display_param_list, **settings.as_streamlit_args('groupby_attr'))
        with col32:
            groupby_metric = st.multiselect("Metric:", metric_list, **settings.as_streamlit_args('groupby_metric'))

        submit_button = st.form_submit_button(label='Plot selected ')

    # save settings for URL query params
    new_settings = dict(varin=var_include,
                    varout_num=int(num_exclude_combo),
                    varout_exact=var_exclude_exact,
                    groupby_attr=groupby_attr,
                    groupby_metric=groupby_metric)

    # manually add varout since there can be multiple vars
    new_settings.update({'varout_%d' % i: var_exclude_list[i] for i in range(int(num_exclude_combo))})

    # finalize new settings
    new_settings = settings.compile_new_settings(**new_settings)

    with st.spinner('Preparing table-view ...'):

        if any([len(v) > 0 for v in var_exclude_list]) or len(var_include) > 0:
            # remove experiments that contain explicitly removed variables
            if var_exclude_list is not None:
                for var_exclude in var_exclude_list:
                    if len(var_exclude) > 0:
                        exp_results = parser_obj.experiment_exclude_vars(exp_results, var_exclude,
                                                                         only_exact_combination=var_exclude_exact)

            # retain experiments that contain only explicitly requested variables
            if len(var_include) > 0:
                # split var_include into dictionary
                exp_results = parser_obj.experiment_retain_only_vars(exp_results, var_include)

            if groupby_attr is not None and groupby_metric is not None:

                exp_results = parser_obj.experiment_group_by(exp_results, groupby_attr, groupby_metric)

            # retain only attributes that are not the same
            unique_params = parser_obj.get_available_params(exp_results)
            exp_results = [{k:v for k,v in exp.items() if k in unique_params or k.startswith(parser_obj.METRICS_SUFFIX)} for exp in exp_results]

            st.table(exp_results)

            st.success('Done!')

    return new_settings
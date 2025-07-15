import copy

import streamlit as st

import logging
import numpy as np
import pylab as plt

import parser

import importlib
importlib.reload(parser)

import numpy as np
import pandas as pd
import plotly.express as px

from scipy.ndimage.filters import uniform_filter1d

from stqdm import stqdm as tqdm

def moving_average(x, w):
    return uniform_filter1d(x, size=w)

import importlib, settings_manager
importlib.reload(settings_manager)

def display_lineplot(parser_obj, exp_results, param_list, all_variables_str, existing_group_by, plot_mutex):

    metric_list = parser_obj.metric_list

    display_param_list = sorted(list(param_list.keys()))
    display_x_axis = ["---"]+display_param_list

    DEFAULT_X_AXIS = 'epoch_eval'
    # revert to empty x-axis if default is missing
    if DEFAULT_X_AXIS not in display_x_axis:
        DEFAULT_X_AXIS = "---"

    query_params = st.experimental_get_query_params()

    settings = settings_manager.InputSettingsManager(param_prefix='lineplot')

    # set default settings as specified in URL query params
    # first val == default value, second val = parser func from/to str, third val = key in state session
    settings.add_definition(x=(display_x_axis.index(DEFAULT_X_AXIS), settings.CHOICE_FN, 'lineplot_form_x_axis',display_x_axis),
                            y=([], settings.LIST_FN, 'lineplot_form_y_axis_list', metric_list),
                            y_smooth=(1,settings.INT_FN,'lineplot_form_y_smooth',None),
                            colors=([], settings.LIST_FN,'lineplot_form_colors_by',display_param_list),
                            dashes=([], settings.LIST_FN,'lineplot_form_dashes_by',display_param_list),
                            varin=([], settings.LIST_FN,'lineplot_form_var_include',all_variables_str),
                            varout_num=(1, settings.INT_FN,'lineplot_form_num_exclude_combo',None),
                            varout_exact=(False, settings.BOOL_FN,'lineplot_var_exclude_exact',None),
                            groupby_attr=([], settings.LIST_FN,'lineplot_form_groupby_attr',display_param_list),
                            groupby_metric=([], settings.LIST_FN,'lineplot_form_groupby_metric',metric_list))

    settings.parse(query_params, st.session_state)

    # there is variable number of varout params so we need to handle them manually
    for i in range(int(settings['varout_num'])):
        settings.add_definition(**{'varout_%d' % i: ([], settings.LIST_FN, "lineplot_form_var_exclude_%d" % i, all_variables_str)})

    # parse again with additional values for varout
    settings.parse(query_params, st.session_state)

    ###########################################################################################################
    ## Line plot
    with st.form(key='lineplot_form'):
        col11, col12, col13 = st.columns(3)
        col21, col22 = st.columns(2)
        col31, col32 = st.columns(2)

        with col11:
            x_axis = st.selectbox("Select X-axis group:", display_x_axis, **settings.as_streamlit_args('x',value_name='index'))
        with col12:
            y_axis_list = st.multiselect("Select Y-axis metrics:", metric_list, **settings.as_streamlit_args('y'))
            y_smooth = st.slider("Apply line smoothing:", min_value=1, step=1, max_value=10, **settings.as_streamlit_args('y_smooth',value_name='value'))

        with col13:
            colors_by = st.multiselect("Select color by:", display_param_list, **settings.as_streamlit_args('colors'))
            dashes_by = st.multiselect("Select dashes by:", display_param_list, **settings.as_streamlit_args('dashes'))
        with col21:
            var_include = st.multiselect("Should INCLUDE all those variables:", all_variables_str, **settings.as_streamlit_args('varin'))
            logging.debug("using var_include: ", var_include)
        with col22:
            num_exclude_combo = st.number_input("Number combinations", min_value=1, step=1, **settings.as_streamlit_args('varout_num', value_name='value'))

            var_exclude_list = []
            for i in range(int(num_exclude_combo)):
                var_exclude_list.append(
                    st.multiselect("Should EXCLUDE any of those variables (%d):" % i, all_variables_str, **settings.as_streamlit_args('varout_%d' % i)))
            var_exclude_exact = st.checkbox("EXCLUDE exact combination only", **settings.as_streamlit_args('varout_exact', value_name='value'))

        with col31:
            groupby_attr = st.multiselect("Group by attribute", display_param_list, **settings.as_streamlit_args('groupby_attr'))
        with col32:
            groupby_metric = st.multiselect("Metric:", metric_list, **settings.as_streamlit_args('groupby_metric'))

        submit_button = st.form_submit_button(label='Plot selected ')

    # save settings for URL query params
    new_settings = dict(x=x_axis,
                        y=y_axis_list,
                        y_smooth=y_smooth,
                        colors=colors_by,
                        dashes=dashes_by,
                        varin=var_include,
                        varout_num=int(num_exclude_combo),
                        varout_exact=var_exclude_exact,
                        groupby_attr=groupby_attr,
                        groupby_metric=groupby_metric)

    # manually add varout since there can be multiple vars
    new_settings.update({'varout_%d' % i: var_exclude_list[i] for i in range(int(num_exclude_combo))})

    # finalize new settings
    new_settings = settings.compile_new_settings(**new_settings)

    with st.spinner('Preparing line-plot ...'):

        if x_axis != "---" and y_axis_list is not None and len(y_axis_list) > 0:
            color_by_attr = True if colors_by is not None and len(colors_by) > 0 else False
            dashes_by_attr = True if dashes_by is not None and len(dashes_by) > 0 else False

            exp_results = copy.deepcopy(exp_results)

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
                exp_results = parser_obj.experiment_group_by(exp_results, groupby_attr, groupby_metric)

            forbidden_keys = [x_axis]
            if groupby_attr is not None and groupby_metric is not None:
                forbidden_keys += groupby_attr
            if existing_group_by is not None:
                forbidden_keys += existing_group_by

            x_labels = sorted(param_list[x_axis])

            grouped_exp = {}

            for exp in tqdm(exp_results, desc='Generating lines from experiments 1/2'):
                keys = ["%s=%s" % (k, v) for k, v in exp.items() if
                        k.startswith(".") is False and k not in forbidden_keys]
                exp_str = ";".join(keys)

                if exp_str not in grouped_exp:
                    grouped_exp[exp_str] = dict(keys=[], x=[], color_keys=[], dash_keys=[])
                    grouped_exp[exp_str].update({lineplot_y_axis:[] for lineplot_y_axis in y_axis_list})

                grouped_exp[exp_str]['keys'] = keys
                grouped_exp[exp_str]['x'].append(exp[x_axis])

                for lineplot_y_axis in y_axis_list:
                    grouped_exp[exp_str][lineplot_y_axis].append(exp["." + lineplot_y_axis])

                if color_by_attr:
                    grouped_exp[exp_str]['color_keys'].append([exp[c] for c in colors_by])
                if dashes_by_attr:
                    grouped_exp[exp_str]['dash_keys'].append([exp[d] for d in dashes_by])

            common_keys = set.intersection(*[set(exp_group['keys']) for exp_group in grouped_exp.values()])
            legend_keys = ["  ".join(sorted(list(set(exp_group['keys']) - common_keys))) for exp_group in
                           grouped_exp.values()]

            common_keys_str = "  ".join([k for k in sorted(list(common_keys)) if k in var_include])
            common_title = common_keys_str + " - " if len(common_keys_str) else ""

            # using interactive Plotly
            pd_color_k = ",".join(colors_by) if len(colors_by) > 0 else None
            pd_dash_k = ",".join(dashes_by) if len(dashes_by) > 0 else None

            progress_grouped_exp = tqdm(grouped_exp.items(), desc='Generating lines from experiments 2/2')

            exp_for_display = []
            for group_attr,(k, exp_group) in zip(legend_keys,progress_grouped_exp):
                len_data = len(exp_group['x'])
                exp_data = {x_axis: exp_group['x'],
                            'x': [x_labels.index(x_) for x_ in exp_group['x']],
                            'group_attrs': [group_attr] * len_data}

                if pd_color_k is not None:
                    exp_data[pd_color_k] = [",".join(map(str,c)) for c in exp_group['color_keys']]
                if pd_dash_k is not None:
                    exp_data[pd_dash_k] = [",".join(map(str,d)) for d in exp_group['dash_keys']]

                exp_data.update({lineplot_y_axis: exp_group[lineplot_y_axis] for lineplot_y_axis in y_axis_list})
                exp_data.update({k.split("=")[0]: [k.split("=")[1]]*len_data for k in exp_group['keys']})

                df = pd.DataFrame(exp_data)
                df.sort_values(by=['x'])
                if y_smooth > 1:
                    for lineplot_y_axis in y_axis_list:
                        df[lineplot_y_axis] = moving_average(df[lineplot_y_axis], y_smooth)

                exp_for_display.append(df)

            df = pd.concat(exp_for_display,ignore_index=True)
            df.sort_values(by=['group_attrs','x'])

            progress_y_axis = tqdm(y_axis_list, desc='Plotting graph')
            for lineplot_y_axis in progress_y_axis:
                progress_y_axis.set_postfix(metric=lineplot_y_axis)

                fig = px.line(df, x=x_axis, y=lineplot_y_axis, line_group='group_attrs', hover_name='group_attrs',
                              color=pd_color_k, line_dash=pd_dash_k, symbol=pd_dash_k, category_orders={x_axis: x_labels},
                              title='<b>%s%s</b>' % (common_title, lineplot_y_axis), height=786,
                              color_discrete_sequence=px.colors.qualitative.D3)

                fig.update_traces(mode="markers+lines")
                fig.update_layout(
                    title={
                        'y': 0.92,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': dict(size=20)
                    },
                    yaxis_tickfont_size=16,
                    xaxis_tickfont_size=16,
                    yaxis_titlefont_size=22,
                    xaxis_titlefont_size=18,

                )
                st.plotly_chart(fig, use_container_width=True)

        st.success('Done!')

    return new_settings
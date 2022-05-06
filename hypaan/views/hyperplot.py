import copy

import streamlit as st
import hiplot as hip

import parser, settings_manager

import importlib
importlib.reload(parser)
importlib.reload(settings_manager)

def display_hiplot(parser_obj, exp_results, output_csv_file, param_list, existing_group_by):

    metric_list = parser_obj.metric_list

    display_param_list = sorted(list(param_list.keys()))

    query_params = st.experimental_get_query_params()

    settings = settings_manager.InputSettingsManager(param_prefix='hiplot')

    # set default settings as specified in URL query params
    # first val == default value, second val = parser func from/to str, third val = key in state session, fourth val = valid values
    settings.add_definition(groupby_attr=([], settings.LIST_FN, 'hiplot_form_groupby_attr', display_param_list),
                            groupby_metric=([], settings.LIST_FN, 'hiplot_form_groupby_metric', metric_list))

    settings.parse(query_params, st.session_state)

    ###########################################################################################################
    ## HI-plot
    with st.spinner('Displaying hiplot ...'):

        with st.form(key='hiplot_groupby_form'):
            col1, col2 = st.columns(2)
            with col1:
                groupby_attr = st.multiselect("Group by attribute", display_param_list, **settings.as_streamlit_args('groupby_attr'))
            with col2:
                groupby_metric = st.multiselect("Metric:", metric_list, **settings.as_streamlit_args('groupby_metric'))

            submit_button = st.form_submit_button(label='Select attribute group by best metric')

        # save settings for URL query params
        new_settings = dict(groupby_attr=groupby_attr,
                            groupby_metric=groupby_metric)

        # finalize new settings
        new_settings = settings.compile_new_settings(**new_settings)

        if groupby_attr is not None and groupby_metric is not None:
            exp_results = copy.deepcopy(exp_results)

            exp_results = parser_obj.experiment_group_by(exp_results, groupby_attr, groupby_metric)

        if len(exp_results) > 0:
            # We create a large experiment with 1000 rows
            xp = hip.Experiment.from_iterable(exp_results)  # EXPERIMENTAL: Reduces bandwidth at first load

            # save to CSV file - but ignore if grouping by attribute is enabled
            if groupby_attr is None and existing_group_by is None:
                xp.to_csv(output_csv_file)
                xp.to_html(output_csv_file.replace(".csv", ".html"))

            # xp._compress = True
            # ... convert it to streamlit and cache that (`@st.cache` decorator)
            xp.to_streamlit(key="hiplot").display()
        st.success('Done!')

    return new_settings
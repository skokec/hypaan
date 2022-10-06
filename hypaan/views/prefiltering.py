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

def display_prefiltering(results_structures):

    all_variables_str = []
    param_list = []

    for res in results_structures:
        if type(res) == dict:
            for attr_list in res.values():
                all_variables_str.extend(attr_list)
                param_list.extend([a.split("=")[0] for a in attr_list])

    all_variables_str = list(np.unique(all_variables_str))
    param_list = list(np.unique(param_list))

    print(all_variables_str)
    print(len(all_variables_str))

    print(param_list)
    print(len(param_list))

    query_params = st.experimental_get_query_params()

    settings = settings_manager.InputSettingsManager(param_prefix='prefilter')

    # set default settings as specified in URL query params
    # first val == default value, second val = parser func from/to str, third val = key in state session
    settings.add_definition(varin=([], settings.LIST_FN,'prefilter_form_var_include',all_variables_str),
                            varout=([], settings.LIST_FN,'prefilter_var_exclude_exact',all_variables_str),)

    settings.parse(query_params, st.session_state)

    # parse again with additional values for varout
    settings.parse(query_params, st.session_state)

    ###########################################################################################################
    ## Line plot
    with st.form(key='prefilter_form'):
        st.markdown("* Select specific attributes to be required or forbidden to reduce the number paths for scanning.\n * This will avoid loading unnneccessary results and provide significant speed-ups when loading from a large set of experiments.")
        st.markdown("""---""")
        var_include = st.multiselect("RETAIN experiments that include at least one of those attribute values:", all_variables_str, **settings.as_streamlit_args('varin'))

        logging.debug("using var_include: ", var_include)


        var_exclude = st.multiselect("REMOVE experiments that include any of attribute values:" , all_variables_str, **settings.as_streamlit_args('varout'))

        submit_button = st.form_submit_button(label='Continue')

    # save settings for URL query params
    new_settings = dict(varin=var_include,
                        varout=var_exclude)

    # finalize new settings
    new_settings = settings.compile_new_settings(**new_settings)

    if submit_button or len(var_include) > 0 or len(var_include) > 0:
        with st.spinner('Removing experiment paths based on selected attributes ...'):

            num_removed = 0
            if len(var_include) > 0:
                results_structures, num_removed_varin = _experiment_retain_only_vars(results_structures,var_include)
                num_removed += num_removed_varin

            if len(var_exclude) > 0:
                results_structures, num_removed_varout = _experiment_exclude_vars(results_structures,var_exclude)
                num_removed += num_removed_varout

            if num_removed > 0:
                st.success('Done - removed %d paths !' % num_removed)
            else:
                st.success('Done !')

    else:
        results_structures = []
    return results_structures, new_settings


def _experiment_exclude_vars(results_structures, var_excluded):
    excluded_variables = [(var.split("=")[0], "=".join(var.split("=")[1:])) for var in var_excluded]

    for k, v in excluded_variables:
        logging.debug("excluded experiments with key: '%s' with value: '%s'" % (k, v))

    # retain only experiments that do not contain any excluded var
    num_removed = 0
    new_results_structures = []
    for res in results_structures:
        if type(res) == dict:
            new_res = {}
            for path, attr_list in res.items():
                if any([ex_var in attr_list for ex_var in var_excluded]):
                    # do not add
                    num_removed+=1
                else:
                    new_res[path] = attr_list
        else:
            new_res = res
        new_results_structures.append(new_res)
    return new_results_structures, num_removed

def _experiment_retain_only_vars(results_structures, var_included):
    included_variables = [(var.split("=")[0], "=".join(var.split("=")[1:])) for var in var_included]

    for k, v in included_variables:
        logging.debug("requireing exp to have key: '%s' with value: '%s'" % (k, v))

    # retain only experiments that do contain required vars
    new_results_structures = []
    num_removed = 0
    for res in results_structures:
        if type(res) == dict:
            new_res = {}
            for path, attr_list in res.items():
                # check if key is in this list and then retain only the paths that have key-value present
                present_attr_keys = [a.split("=")[0] for a in attr_list]
                if any([in_key in present_attr_keys and in_var not in attr_list for in_var, (in_key, _) in zip(var_included, included_variables)]):
                    # do not add
                    num_removed+=1
                else:
                    new_res[path] = attr_list
        else:
            new_res = res
        new_results_structures.append(new_res)
    return new_results_structures, num_removed

class InputSettingsManager:
    """
    Class for managing settings for input streamlit components (st.selectbox, st.multiselect, st.checkbox, ...) that can
    saved from streamlit query params and/or from streamlit state session. Retreive function is prepared for simplified
    use in streamlit input components. Features:
      - use HTTP query as storage (using st.experimental_get_query_params() and st.experimental_set_query_params
      - simplified use in streamlit input components using as_streamlit_args(..) function
      - all settings can be prefixed to avoid conflicts when used multiple times
      - all input/output parsing is automatically managed by providing type of input (INT, STR, BOOL, LIST, CHOICE)
      - settings are automatically populated based on following priority:
        1. streamlit session state
        2. HTTP param query
        3. default value (default values are not stored in HTTP param query dictionary)

    Each setting is defined using InputSettingsManager.add_definition(key_name=(...)). Then existing settings from
    state session or from query params can be read using:
        InputSettingsManager.parse(st.experimental_get_query_params(), st.session_state)

    Finally, settings can be retrieved using __get_item__ idiom or can be used to define streamlit input component using
    InputSettingsManager.as_streamlit_args(), e.g.:
        var1 = st.multiselect("Your label", input_list, **settings.as_streamlit_args('your_key1'))
        var2 = st.selectbox("Your label", input_list, **settings.as_streamlit_args('your_key2',value_name='index')))

    Any updates to vars can be retrived for store in as query_param:
        new_settings = settings.compile_new_settings(your_key1=var1, your_key2=var2)

    Example usage:
        query_params = st.experimental_get_query_params()

        settings = InputSettingsManager(param_prefix='my_namespace')

        settings.add_definition(your_key1=([], settings.LIST_FN, 'my_streamlit_input_component_name1', input_list),
                                your_key2=([], settings.LIST_FN, 'my_streamlit_input_component_name2', input_list))

        settings.parse(query_params, st.session_state)

        var1 = st.multiselect("Your label", input_list, **settings.as_streamlit_args('your_key1'))
        var2 = st.selectbox("Your label", input_list, **settings.as_streamlit_args('your_key2',value_name='index')))

        ...

        new_settings = settings.compile_new_settings(your_key1=var1, your_key2=var2)

        st.experimental_set_query_params(**{k:v for k,v in new_settings.items() if v is not None and len(v) > 0})

    """
    PARAM_DELIM = ";"

    # from string and to string functions (1: parsing from query string, 2: saving to query string, 3: parsing from sessions state)
    STR_FN = (lambda x, d: d.index(x), lambda x: x, lambda x, d: d.index(x))
    CHOICE_FN = (lambda x,d: d.index(x), lambda x: x, lambda x,d: d.index(x))
    INT_FN = (lambda x,d: int(x), lambda x: str(x), lambda x,d: int(x))
    BOOL_FN = (lambda x,d: x == 'True', lambda x: str(x), lambda x,d: x)
    LIST_FN = (lambda x,d: x.split(InputSettingsManager.PARAM_DELIM), lambda x: InputSettingsManager.PARAM_DELIM.join(x), lambda x,d: x)


    def __init__(self, param_prefix):
        self.param_prefix = param_prefix
        self.settings_def = {}
        self.settings = {}

    def add_definition(self, **kwargs):
        """
        Define  settings for specific key that can be stored. The definition is expected as tuple of 4 values:
            first val == default value,
            second val = parser func from/to str based on predefined SettingsManager.STR/INT/BOOL/LIST/CHOICE_FN functions
            third val = key from state session

        For instance:
            settings.add_definition(groupby_attr=([], settings.LIST_FN, 'hiplot_form_groupby_attr', display_param_list))
        this will add settings with name='groupby_attr' and streamlit state session key='hiplot_form_groupby_attr', and is
        expected as list of values from display_param_list.
        """
        for k,v in kwargs.items():
            self.settings_def[k] = v

    def parse(self, query_params_, session_state_params):
        query_params = {k.replace(self.param_prefix + "_", ""): v for k, v in query_params_.items() if k.startswith(self.param_prefix)}
        self.settings = {}

        # parse all query prams into settings or use default setting if not present or use session state if present
        for key, (default, (fn_query, _, fn_sess), sess_key, valid) in self.settings_def.items():
            if sess_key in session_state_params:
                self.settings[key] = fn_sess(session_state_params[sess_key], valid)
            elif key in query_params:
                self.settings[key] = fn_query(query_params[key][0], valid)
            else:
                self.settings[key] = default

    def compile_new_settings(self, **kwargs):
        # remove any default settings
        new_settings = {k: self.settings_def[k][1][1](v) for k, v in kwargs.items() if self.settings_def[k][0] != v}

        # append prefix and remove unused settings
        return {self.param_prefix + '_' + k: v for k, v in new_settings.items() if v is not None and len(v) > 0}

    def as_streamlit_args(self, param_name, value_name='default', key_name='key'):
        """
        Return settings for param_name as dictionary of keys that can be directly used in streamlit input call.

        For instance as_streamlit_args(param_name='my_key') will return:
          dict(default='value-of-the-key-setting',
               key='default-value-of-key')
        based on
        """
        key = self.settings_def[param_name][2]
        valid_values = self.settings_def[param_name][3]
        value = self.settings[param_name]

        # filter out invalid list values
        if valid_values is not None and isinstance(value,list):
            value = [v for v in value if v in valid_values]

        return {value_name: value,
                key_name: key}

    def __getitem__(self, item):
        return self.settings[item]



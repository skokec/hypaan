import logging
import os
import json
import dill
from collections import OrderedDict

def iter_variants(v_k, f, exp_types):
    v, k = v_k
    new_variants = {}
    path_exists = lambda _p: os.path.exists(_p) or ('{}' in _p and any([os.path.exists(_p.format(t)) for t in exp_types]))
    if type(f) == dict:
        for v_, k_ in f.items():
            p = os.path.join(v, v_)
            if path_exists(p):
                new_variants[p] = k + k_
    else:
        p = os.path.join(v, f)
        if path_exists(p):
            new_variants[p] = k
    return new_variants

def load_json_metrics(res_name):
    with open(res_name) as f:
        metrics = json.load(f)
    return metrics, {}

def parse_exp(res_name, attributes_key_val, metrics, METRICS_SUFFIX, load_metrics_fn=None):
    try:
        attrib = None
        if os.path.exists(res_name):
            attrib = OrderedDict()
            for i, v in enumerate(attributes_key_val):
                v_split = v.split("=")
                if "=" in v:
                    attrib[v_split[0]] = "=".join(v_split[1:])
                else:
                    attrib['H%d' % i] = v

            # use default json version if load_metrics_fn is not provided
            if load_metrics_fn is None:
                load_metrics_fn = load_json_metrics
            else:
                load_metrics_fn = dill.loads(load_metrics_fn)

            # parse metrics file
            res_metrics, res_keys = load_metrics_fn(res_name)

            # update attributes
            if res_keys is not None and len(res_keys) > 0:
                attrib.update(res_keys)

            for m in metrics:
                if m in res_metrics:
                    attrib[METRICS_SUFFIX + m] = res_metrics[m]
    except Exception as e:
        logging.error("Error while parsing file '%s': " % res_name, e)
        raise Exception("Error while parsing file '%s': %s" % (res_name, str(e))) from e
    return attrib

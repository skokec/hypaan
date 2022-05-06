import os
import numpy as np
import logging

import itertools, functools, dill
from stqdm import stqdm as tqdm
#from tqdm import tqdm
import parallel_exp


from metric import Metric

class ResultsParser:
    '''
    Parser class for all results saved by test_polar.py.

    It searches for all experiments existing experiments based on provided datasets_struct.get_dataset_struct() list.
    All possible combinations of experiments are searched for and for each found experiment is parsed to ['key=value',...]
    list for further processing as returned by ResultsParser.get_results_list().
    '''
    METRICS_SUFFIX="."

    def __init__(self, dataset_def):

        self.experiment_types = dataset_def.get_experiment_types()
        self.metrics = Metric.from_definition(dataset_def.get_metrics_defintion())
        self.metrics_parser_fn = None
        if 'get_metrics_file_parser_fn' in dir(dataset_def):
            self.metrics_parser_fn = dataset_def.get_metrics_file_parser_fn()

        self.metric_list = [m.get_name() for m in self.metrics]
        self.metric_list_best_val = {m.get_name(): m.get_best_value() for m in self.metrics}
        self.metric_list_direction = {m.get_name(): m.get_best_direction() for m in self.metrics}

    def _generate_pattern_variants(self, results_structures, exp_types, pool):
        '''
        Generate list of existing filepaths from a list of all possible results (fast parallel version).
        '''

        variants_count = []
        variants_first_empty = None

        exp_types = [exp_type for _, t_mark in exp_types.items()
                        for exp_type in ([t_mark] if type(t_mark) not in [list, tuple] else t_mark)]

        variants = {'': []}
        progress_bar = tqdm(results_structures, desc='Traversing/unwinding %d folder levels' % len(results_structures))
        for ii,f in enumerate(progress_bar):
            new_variants = pool.starmap_async(parallel_exp.iter_variants,
                                              zip(variants.items(), itertools.repeat(f), itertools.repeat(exp_types)))
            new_variants = new_variants.get()

            variants = {k: v for x in [i.items() for i in new_variants] for k, v in x}

            variants_count.append(len(variants))
            if len(variants) == 0 and variants_first_empty is None:
                variants_first_empty = f

            progress_bar.set_postfix(dict(total_found=len(variants)))

        return variants, variants_count, variants_first_empty

    def parse_results_list(self, dataset_name, results_structures, exp_type, pool=None):
        assert pool != None

        with tqdm(total=2,desc='Searching and parsing all experiments') as main_progress_bar:
            main_progress_bar.set_postfix_str("Step 1: traversing/unwinding all folder combinations")

            exp_types = {exp_type: self.experiment_types[exp_type]}

            metric_list = [m.get_name() for m in self.metrics]

            logging.info('#################################################################')
            logging.info('#########################  %s #################################' % dataset_name)
            logging.info('Traversing/unwinding %d folder levels' % len(results_structures))
            print(exp_types)
            all_exp, level_counts, level_first_empty = self._generate_pattern_variants(results_structures, exp_types, pool)

            for i, c in enumerate(level_counts):
                logging.info('%slevel(%d): %d ' % (' -> ' if i > 0 else '',i, c))

            if len(all_exp) == 0:
                logging.warning('WARNING: NO valid folders found from unwinding of %d folder levels' % len(results_structures))
                #logging.warning('First empty layer when unwinding following list: %s', level_first_empty)

            logging.info('done .. (total=%d)' % len(all_exp))

            main_progress_bar.update(1)
            main_progress_bar.set_postfix_str("Step 2: loading/parsing all potential metric files")

            traversing_details = dict(found_level_counts=level_counts,
                                      first_empty_level=level_first_empty)

            metrics_parser_fn_str = dill.dumps(self.metrics_parser_fn) if self.metrics_parser_fn is not None else None

            t_mark = self.experiment_types[exp_type]

            if type(t_mark) not in [list,tuple]:
                t_mark = [t_mark]
            exp_results = []

            logging.info('Parsing %s experiments for %s' % (exp_type, dataset_name))
            if len(all_exp) > 0:
                for exp_type in t_mark:
                    exp_results += [(patt.format(exp_type), exp_keys + ['exp_type=%s' % exp_type], metric_list, self.METRICS_SUFFIX)
                                        for patt,exp_keys in all_exp.items()]
                total_items = len(exp_results)
                exp_results = pool.starmap_async(functools.partial(parallel_exp.parse_exp,load_metrics_fn=metrics_parser_fn_str),
                                                 exp_results)

                exp_results = self.wait_async_progress(exp_results, total_items,
                                                       desc='Reading and parsing %d potential metrics files' % total_items)

            exp_results = [exp for exp in exp_results if exp is not None]
            for exp in exp_results:
                for metric in self.metrics:
                    m = metric.get_name()
                    if self.METRICS_SUFFIX+m in exp:
                        exp[self.METRICS_SUFFIX+m] = metric.parse_value(exp[self.METRICS_SUFFIX+m])

            main_progress_bar.update(1)
            logging.info('done (total=%d)' % len(exp_results))
            logging.info('#################################################################')

        return exp_results, os.path.join(*results_structures[:2],'hiplot_%s.csv' % exp_type.lower()), traversing_details

    @staticmethod
    def wait_async_progress(ret, total_items, **kwargs):
        import time

        max_items = ret._number_left
        if max_items > 0:
            last_item = max_items
            with tqdm(total=total_items, **kwargs) as progress_bar:
                while ret._number_left > 0:
                    current_left = ret._number_left
                    diff = last_item - current_left
                    last_item = current_left
                    progress_bar.update(int(np.round(diff / max_items * total_items)))
                    time.sleep(0.1)

        return ret.get()

    def get_available_params(self, exp_results):
        param_list = {}

        metric_list = [m.get_name() for m in self.metrics]

        for exp in exp_results:
            for k in exp.keys():
                if k[len(self.METRICS_SUFFIX):] in metric_list: continue
                if k in param_list:
                    param_list[k].append(exp[k])
                else:
                    param_list[k] = [exp[k]]

        # remove duplicated values
        param_list = {k: np.unique(v) for k, v in param_list.items()}
        # remove vars with single value only
        return {k:v for k,v in param_list.items() if len(v) > 1}

    def experiment_group_by(self, exp_results, group_by, best_by, best_by_modify_op=None, remove_gruped_attributes=False,
                            exp_results_for_op_metrics=None):
        if group_by is None or len(group_by) <= 0 or best_by is None or len(best_by) <= 0:
            return exp_results

        best_target = [self.metric_list_best_val[m] for m in best_by]
        best_by = [ResultsParser.METRICS_SUFFIX + m for m in best_by]

        assert len(best_target) == len(best_by)

        exp_to_str_fn = lambda exp: ";".join(["%s=%s" % (k, v) for k, v in exp.items() if k.startswith(self.METRICS_SUFFIX) is False and k not in group_by])

        exp_to_str_mertric_fn = lambda exp: ";".join(["%s=%s" % (k, exp[k]) for k in sorted(exp.keys()) if k.startswith(self.METRICS_SUFFIX) is False and k != "exp_type"])

        grouped_exp_for_op_metrics = {}
        missing_grouped_exp_op_metrics = []
        if exp_results_for_op_metrics is not None:
            assert best_by_modify_op == None, "DO NOT USE best_by_modify_op and exp_results_for_op_metrics simultaneously"

            for exp in exp_results_for_op_metrics:
                grouped_exp_for_op_metrics[exp_to_str_mertric_fn(exp)] = exp

        grouped_exp = {}
        for exp in exp_results:
            exp_str = exp_to_str_fn(exp)
            # use metrics from provided exp_results_for_op_metrics if available else use original exp
            exp_for_op_metrics = exp
            if len(grouped_exp_for_op_metrics) > 0:
                exp_metrics_str = exp_to_str_mertric_fn(exp)
                if exp_metrics_str in grouped_exp_for_op_metrics:
                    exp_for_op_metrics = grouped_exp_for_op_metrics[exp_metrics_str]
                else:
                    logging.warning('%s not found in corresponding exp metrics' % exp_metrics_str)
                    missing_grouped_exp_op_metrics.append(exp_metrics_str)
                    continue

            if exp_str not in grouped_exp:
                grouped_exp[exp_str] = dict(all_exps=[], op_values=[])
            grouped_exp[exp_str]['all_exps'].append(exp)

            grouped_exp[exp_str]['op_values'].append([exp_for_op_metrics[k] for k in best_by])

        logging.info("from experiment_group_by: num grouped exp : %d" % len(grouped_exp))
        if len(missing_grouped_exp_op_metrics) > 0:
            logging.warning("BUT MISSING %d op_vals metric from corresponding exp list - those exp were ignored" % len(missing_grouped_exp_op_metrics) )

        retained_exp = []
        for k, exp_group in grouped_exp.items():
            # select exp based on provided op
            op_values = np.array(exp_group['op_values']) - np.array(best_target).reshape(1, -1)
            op_values = (op_values ** 2).sum(axis=1)
            best_id = np.where(op_values == np.min(op_values))[0][0]

            if remove_gruped_attributes:
                retained_exp.append({k: v for k, v in exp_group['all_exps'][best_id].items() if k not in group_by})
            else:
                retained_exp.append(exp_group['all_exps'][best_id])

            if best_by_modify_op is not None:
                for i,k in enumerate(best_by):
                    retained_exp[-1][k] = best_by_modify_op(np.array(exp_group['op_values'])[:,i])

        return retained_exp


    def experiment_exclude_vars(self, exp_results, var_excluded, only_exact_combination=False):
        excluded_variables = [(var.split("=")[0], "=".join(var.split("=")[1:])) for var in var_excluded]

        for k, v in excluded_variables:
            logging.debug("excluded key: '%s' with value: '%s'" % (k, v))

        op = all if only_exact_combination else any

        # retain only experiments that do not contain any excluded var
        return [exp for exp in exp_results if not op([exp[k] == v for k, v in excluded_variables])]

    def experiment_retain_only_vars(self, exp_results, var_included):
        included_variables = [(var.split("=")[0], "=".join(var.split("=")[1:])) for var in var_included]

        for k, v in included_variables:
            logging.debug("required key: '%s' with value: '%s'" % (k, v))

        return [exp for exp in exp_results if all([exp[k] == v for k, v in included_variables])]
class Metric:
    def __init__(self, name, best_val, best_direction, mod_fn=None):
        self.name = name
        self.best_val = best_val
        self.best_direction = best_direction
        self.mod_fn = mod_fn

    def get_name(self):
        return self.name

    def get_best_value(self):
        return self.best_val

    def get_best_direction(self):
        return self.best_direction

    def parse_value(self, input):
        return input if self.mod_fn is None else self.mod_fn(input)

    @staticmethod
    def from_definition(metrics_def):
        return [Metric(name=name, **kwargs) for name,kwargs in metrics_def.items()]
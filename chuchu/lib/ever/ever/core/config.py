import importlib
import os
import pprint
import sys
import warnings
from ast import literal_eval
from collections import OrderedDict

__all__ = ['import_config', 'AttrDict']


# from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
def _import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module


def import_config(config_name_or_path: str, prefix='configs'):
    if config_name_or_path.endswith('.py'):
        m = _import_file('ever.cfg', config_name_or_path)
    else:
        cfg_path_segs = [prefix] + config_name_or_path.split('.')
        cfg_path_segs[-1] = cfg_path_segs[-1] + '.py'
        m = _import_file('ever.cfg', os.path.join(os.path.curdir, *cfg_path_segs))
    return AttrDict.from_dict(m.config)


class AttrDict(OrderedDict):
    def __init__(self, **kwargs):
        super(AttrDict, self).__init__(**kwargs)
        self.update(kwargs)

    @staticmethod
    def from_dict(dict):
        ad = AttrDict()
        ad.update(dict)
        return ad

    def __setitem__(self, key: str, value):
        super(AttrDict, self).__setitem__(key, value)
        super(AttrDict, self).__setattr__(key, value)

    def __setattr__(self, key, value):
        super(AttrDict, self).__setitem__(key, value)
        super(AttrDict, self).__setattr__(key, value)

    def update(self, config: dict):
        for k, v in config.items():
            if k not in self:
                self[k] = AttrDict()
            if isinstance(v, dict):
                self[k].update(v)
            elif isinstance(v, list) and all([isinstance(i, dict) for i in v]):
                self[k] = [AttrDict.from_dict(i) for i in v]
            else:
                self[k] = v

    def update_from_list(self, str_list: list):
        assert len(str_list) % 2 == 0
        for key, value in zip(str_list[0::2], str_list[1::2]):
            key_list = key.split('.')
            item = None
            last_key = key_list.pop()
            for sub_key in key_list:
                item = self[sub_key] if item is None else item[sub_key]
            try:
                item[last_key] = literal_eval(value)
            except:
                item[last_key] = value
                warnings.warn(f'a string {value} is set to {key}')

    def __str__(self):
        return pprint.pformat(self)

    def to_file(self, save_path):
        with open(save_path, 'w') as f:
            f.write(str(self))

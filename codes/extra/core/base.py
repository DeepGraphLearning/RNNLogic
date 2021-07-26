import inspect
from functools import wraps
from collections import defaultdict
from contextlib import contextmanager


class _MetaContainer(object):

    meta_types = set()
    enable_auto_context = False
    # When auto context is enabled, any member name that starts with a meta type is considered as that type.
    # e.g. graph.node_value are regarded as node attributes
    # This may introduce ambiguity and side effects.
    # By default, this is disabled.

    def __init__(self, meta=None, **kwargs):
        if meta is None:
            meta = {}

        self._setattr("_meta_context", None)
        self._setattr("meta", meta)
        for k, v in kwargs.items():
            self._setattr(k, v)

    @contextmanager
    def context(self, type):
        if type is not None and type not in self.meta_types:
            raise ValueError("Expect context type in %s, but got `%s`" % (self.meta_types, type))
        backup = self._meta_context
        self._setattr("_meta_context", type)
        yield
        self._setattr("_meta_context", backup)

    def __setattr__(self, key, value):
        type = self._meta_context
        if type is None and self.enable_auto_context:
            for _type in self.meta_types:
                if key.startswith(_type):
                    type = _type
                    break
        if type:
            self.meta[key] = type
        self._setattr(key, value)

    def __delattr__(self, key):
        if key in self.meta:
            del self.meta[key]
        del self.key

    def _setattr(self, key, value):
        return super(_MetaContainer, self).__setattr__(key, value)

    @property
    def data(self):
        return {k: getattr(self, k) for k in self.meta}


class Tree(defaultdict):

    def __init__(self):
        super(Tree, self).__init__(Tree)

    def flatten(self, prefix=None, result=None):
        if prefix is None:
            prefix = ""
        else:
            prefix = prefix + "."
        if result is None:
            result = {}
        for k, v in self.items():
            if isinstance(v, Tree):
                v.flatten(prefix + k, result)
            else:
                result[prefix + k] = v
        return result


class Registry(object):

    table = Tree()

    @classmethod
    def register(cls, name):

        def wrapper(obj):
            entry = cls.table
            keys = name.split(".")
            for key in keys[:-1]:
                entry = entry[key]
            if keys[-1] in entry:
                raise KeyError("`%s` has already been registered by %s" % keys[-1], entry[keys[-1]])

            entry[keys[-1]] = obj
            obj._registry_key = name

            return obj

        return wrapper

    @classmethod
    def get(cls, name):
        entry = cls.table
        keys = name.split(".")
        for i, key in enumerate(keys):
            if key not in entry:
                raise KeyError("Can't find `%s` in `%s`" % (key, ".".join(keys[:i])))
            entry = entry[key]
        return entry

    @classmethod
    def search(cls, name):
        count = 0
        for k, v in cls.table.flatten().items():
            if name in k:
                count += 1
                result = v
        if count == 0:
            raise KeyError("Can't find any registered key containing `%s`" % name)
        if count > 1:
            raise KeyError("Ambiguous key `%s`" % name)
        return result


class Configurable(type):

    def config_dict(self):
        if hasattr(self, "_registry_key"):
            cls = self._registry_key
        else:
            cls = self.__class__.__name__
        config = {"class": cls}

        for k, v in self.config.items():
            if isinstance(type(v), Configurable):
                v = v.config_dict()
            config[k] = v
        return config

    @classmethod
    def load_config_dict(cls, config):
        if cls == Configurable:
            cls = Registry.search(config["class"])
        elif cls != config["class"]:
            raise ValueError("Expect config class to be `%s`, but found `%s`" % (cls.__name__, config["class"]))

        new_config = {}
        for k, v in config.items():
            if isinstance(v, dict) and "class" in v:
                v = Configurable.load_config_dict(v)
            if k != "class":
                new_config[k] = v
        return cls(**new_config)

    def __new__(typ, *args, **kwargs):

        cls = type.__new__(typ, *args, **kwargs)
        init = cls.__init__

        @wraps(init)
        def wrapper(*args, **kwargs):
            func = inspect.signature(init)
            func = func.bind(*args, **kwargs)
            func.apply_defaults()
            config = func.arguments.copy()
            self = config.pop(next(iter(func.arguments.keys())))
            self.config = dict(config)
            return init(*args, **kwargs)

        cls.__init__ = wrapper
        cls.load_config_dict = Configurable.load_config_dict
        cls.config_dict = Configurable.config_dict
        return cls
import sys

import importlib



def call_function(method_name, module_prefix, *args):
    """Dynamically imports and calls a processing function (core version)."""
    module_name = f'{module_prefix}.{method_name}'
    spec = importlib.util.find_spec(module_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    func = getattr(module, method_name)
    return func(*args)


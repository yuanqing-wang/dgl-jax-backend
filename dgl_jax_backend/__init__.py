import os
import types
from importlib.util import find_spec, module_from_spec

def get_dgl():
    this_path = __file__.replace("__init__.py", "")
    new_dgl_init_path = os.path.join(this_path, "_init/_dgl__init__.py")

    spec = find_spec("dgl")
    spec.origin = new_dgl_init_path
    spec.loader.path = new_dgl_init_path

    dgl = module_from_spec(spec)
    spec.loader.exec_module(dgl)
    return dgl

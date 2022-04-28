from __future__ import absolute_import

import sys
import os
import json
import importlib
import logging

_enabled_apis = set()

logger = logging.getLogger("dgl-jax-backend")

def _gen_missing_api(api, mod_name):
    def _missing_api(*args, **kwargs):
        raise ImportError('API "%s" is not supported by backend "%s".'
                          ' You can switch to other backends by setting'
                          ' the DGLBACKEND environment.' % (api, mod_name))
    return _missing_api

def load_backend(mod_name):
    # Load backend does four things:
    # (1) Import backend framework (PyTorch, MXNet, Tensorflow, etc.)
    # (2) Import DGL C library.  DGL imports it *after* PyTorch/MXNet/Tensorflow.  Otherwise
    #     DGL will crash with errors like `munmap_chunk(): invalid pointer`.
    # (3) Sets up the tensoradapter library path.
    # (4) Import the Python wrappers of the backend framework.  DGL does this last because
    #     it already depends on both the backend framework and the DGL C library.
    if mod_name in ["pytorch", "tensorflow", "mxnet"]:
        from dgl.backend import load_backend
        load_backend(mod_name)
    elif mod_name == "jax":
        import jax
        mod = jax

        from dgl._ffi.base import load_tensor_adapter # imports DGL C library
        version = mod.__version__
        load_tensor_adapter(mod_name, version)
        logger.debug('Using backend: %s' % mod_name)
        # mod = importlib.import_module('.%s' % mod_name, __name__)
        from _backend import jax as _jax
        mod = _jax
        # thismod = sys.modules[__name__]
        import dgl
        thismod = dgl.backend
        from dgl.backend import backend as _backend
        for api in _backend.__dict__.keys():
            if api.startswith('__'):
                # ignore python builtin attributes
                continue
            if api == 'data_type_dict':
                # load data type
                if api not in mod.__dict__:
                    raise ImportError('API "data_type_dict" is required but missing for'
                                      ' backend "%s".' % (mod_name))
                data_type_dict = mod.__dict__[api]()
                for name, dtype in data_type_dict.items():
                    setattr(thismod, name, dtype)

                # override data type dict function
                setattr(thismod, 'data_type_dict', data_type_dict)

                # for data types with aliases, treat the first listed type as
                # the true one
                rev_data_type_dict = {}
                for k, v in data_type_dict.items():
                    if not v in rev_data_type_dict.keys():
                        rev_data_type_dict[v] = k
                setattr(thismod,
                        'reverse_data_type_dict',
                        rev_data_type_dict)
                # log backend name
                setattr(thismod, 'backend_name', mod_name)
            else:
                # load functions
                if api in mod.__dict__:
                    _enabled_apis.add(api)
                    setattr(thismod, api, mod.__dict__[api])
                else:
                    setattr(thismod, api, _gen_missing_api(api, mod_name))

            thismod.to_dgl_nd = mod.zerocopy_to_dgl_ndarray
            thismod.from_dgl_nd = mod.zerocopy_from_dgl_ndarray

            dgl.backend_name = "jax"
            import importlib
            importlib.reload(dgl)
    else:
        raise NotImplementedError('Unsupported backend: %s' % mod_name)

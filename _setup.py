import os
import wget

BASE_URL = "https://raw.githubusercontent.com/yuanqing-wang/dgl/jax-0.0.1/python/"
BACKEND_INIT_URL = BASE_URL + "dgl/backend/__init__.py"
BACKEND_JAX_INIT_URL = BASE_URL + "dgl/backend/jax/__init__.py"
TENSOR_URL = BASE_URL + "dgl/backend/jax/tensor.py"
SPARSE_URL = BASE_URL + "dgl/backend/jax/sparse.py"

NN_INIT_URL = BASE_URL + "dgl/nn/jax/__init__.py"
NN_GLOB_URL = BASE_URL + "dgl/nn/jax/glob.py"
NN_UTILS_URL = BASE_URL + "dgl/nn/jax/utils.py"
NN_CONV_INIT_URL = BASE_URL + "dgl/nn/conv/__init__.py"

NN_MODEL_NAMES = [
    "__init__",
    "graphconv", "tagconv", "relgraphconv", "gatconv", "sageconv", "edgeconv",
]

def force_download(url, path):
    if os.path.exists(path):
        os.remove(path)
    wget.download(url, path)

def force_mkdir(dir):
    if os.path.exists(dir):
        import shutil
        shutil.rmtree(dir)
    os.mkdir(dir)

def make_empty_file(path):
    with open(path, "w") as file_handle:
        file_handle.write("")

def _setup():
    print(
        "Hacking DGL to sneak JAX backend in."
    )

    from importlib.util import find_spec
    dgl_root = find_spec("dgl").origin.replace("__init__.py", "")
    dgl_backend_root = os.path.join(dgl_root, "backend")

    jax_backend_dir = os.path.join(dgl_backend_root, "jax")
    force_mkdir(jax_backend_dir)

    backend_init_path = os.path.join(dgl_backend_root, "__init__.py")
    force_download(BACKEND_INIT_URL, backend_init_path)

    backend_jax_init_path = os.path.join(dgl_backend_root, "jax/__init__.py")
    force_download(BACKEND_JAX_INIT_URL, backend_jax_init_path)

    tensor_path = os.path.join(dgl_backend_root, "jax/tensor.py")
    force_download(TENSOR_URL, tensor_path)

    sparse_path = os.path.join(dgl_backend_root, "jax/sparse.py")
    force_download(SPARSE_URL, sparse_path)

    nn_path = os.path.join(dgl_root, "nn")
    nn_jax_path = os.path.join(nn_path, "jax")
    force_mkdir(nn_jax_path)
    force_download(NN_INIT_URL, os.path.join(nn_jax_path, "__init__.py"))
    force_download(NN_GLOB_URL, os.path.join(nn_jax_path, "glob.py"))
    force_download(NN_UTILS_URL, os.path.join(nn_jax_path, "utils.py"))

    nn_jax_conv_path = os.path.join(nn_jax_path, "conv")
    force_mkdir(nn_jax_conv_path)

    for nn_model_name in NN_MODEL_NAMES:
        nn_model_url = BASE_URL + "/dgl/nn/jax/conv/%s.py" % nn_model_name
        nn_model_path = os.path.join(nn_jax_conv_path, "%s.py" % nn_model_name)
        force_download(nn_model_url, nn_model_path)

    optim_path = os.path.join(dgl_root, "optim")
    optim_jax_path = os.path.join(optim_path, "jax")
    force_mkdir(optim_jax_path)
    make_empty_file(os.path.join(optim_jax_path, "__init__.py"))

    distributed_nn_path = os.path.join(dgl_root, "distributed/nn")
    distributed_nn_jax_path = os.path.join(distributed_nn_path, "jax")
    force_mkdir(distributed_nn_jax_path)
    make_empty_file(os.path.join(distributed_nn_jax_path, "__init__.py"))

    distributed_optim_path = os.path.join(dgl_root, "distributed/optim")
    distributed_optim_jax_path = os.path.join(distributed_optim_path, "jax")
    force_mkdir(distributed_optim_jax_path)
    make_empty_file(os.path.join(distributed_optim_jax_path, "__init__.py"))

    os.environ["DGL_BACKEND"] = "jax"

if __name__ == "__main__":
    _setup()

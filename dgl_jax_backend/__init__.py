import os
this_path = __path__[0]
os.system("rm -rf " + os.path.join(this_path, "_dgl"))
os.mkdir(os.path.join(this_path, "_dgl"))

from importlib.util import find_spec
dgl_origin = find_spec("dgl").origin.replace("__init__.py", "")
for rel_path in os.listdir(dgl_origin):
    if rel_path == "backend": continue
    full_path = os.path.join(dgl_origin, rel_path)
    os.symlink(
        full_path, os.path.join(this_path, "_dgl", rel_path),
        target_is_directory=os.path.isdir(full_path),
    )

dgl_backend_origin = os.path.join(dgl_origin, "backend")
os.mkdir(os.path.join(this_path, "_dgl/backend"))
for rel_path in os.listdir(dgl_backend_origin):
    if rel_path == "__init__.py": continue
    full_path = os.path.join(dgl_origin, rel_path)
    os.symlink(
        full_path, os.path.join(this_path, "_dgl", "backend", rel_path),
        target_is_directory=os.path.isdir(full_path),
    )

os.symlink(
    os.path.join(this_path, "_swap", "_backend__init__.py"),
    os.path.join(this_path, "_dgl/backend/__init__.py"),
)

os.symlink(
    os.path.join(this_path, "_backend/jax"),
    os.path.join(this_path, "_dgl/backend/jax"),
)

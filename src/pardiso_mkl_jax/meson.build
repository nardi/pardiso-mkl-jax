# Locate the XLA FFI headers bundled with the installed jaxlib and the MKL
# headers provided by mkl-devel. Both are ordinary Python packages, so their
# paths are discovered by asking the interpreter directly rather than
# hardcoding install locations.
jax_ffi_include_dir = run_command(python,
  ['-c', 'import jax; print(jax.ffi.include_dir())'],
  check: true,
).stdout().strip()

mkl_include_dir = run_command(python,
  ['-c', 'import sysconfig; print(sysconfig.get_path("data") + "/include")'],
  check: true,
).stdout().strip()

mkl_lib_dir = run_command(python,
  ['-c', 'import sysconfig; print(sysconfig.get_path("data") + "/lib")'],
  check: true,
).stdout().strip()

# The mkl PyPI wheel ships only a versioned SONAME (e.g. libmkl_rt.so.3), not
# an unversioned libmkl_rt.so symlink, so a plain -lmkl_rt does not resolve at
# link time. Find the exact file and link it with the GNU ld ":filename"
# syntax instead.
mkl_rt_finder_script = '''
import glob
import os
import sys

lib_dir = sys.argv[1]
matches = sorted(glob.glob(os.path.join(lib_dir, "libmkl_rt.so*")))
assert matches, f"libmkl_rt not found under {lib_dir}"
print(os.path.basename(matches[0]))
'''

mkl_rt_library_name = run_command(python,
  ['-c', mkl_rt_finder_script, mkl_lib_dir],
  check: true,
).stdout().strip()

# include_directories() rejects absolute paths that happen to fall under the
# source tree, which .venv/include does whenever the virtual environment is
# created inside the project directory (the normal uv layout). Passing the
# include and link paths as raw compiler arguments instead avoids that
# restriction, and works the same regardless of where .venv lives.
#
# At runtime libmkl_rt is loaded explicitly (see _mkl_loader.py) rather than
# through rpath, because the mkl wheel installs its shared libraries directly
# under the environment's lib/ directory, not under site-packages, and that
# directory can differ between an isolated build environment and the final
# install environment.
jax_ffi_dependency = declare_dependency(
  compile_args: ['-I' + jax_ffi_include_dir],
)

mkl_dependency = declare_dependency(
  compile_args: ['-I' + mkl_include_dir],
  link_args: ['-L' + mkl_lib_dir, '-l:' + mkl_rt_library_name],
)

pardiso_ffi_core = static_library('pardiso_ffi_core',
  '_pardiso_ffi.cc',
  dependencies: [jax_ffi_dependency, mkl_dependency],
)

python.extension_module('_ffi',
  '_ffi.pyx',
  link_with: pardiso_ffi_core,
  dependencies: [mkl_dependency],
  override_options: ['cython_language=cpp'],
  install: true,
  subdir: 'pardiso_mkl_jax',
)

python.install_sources(
  [
    '__init__.py',
    '_mkl_loader.py',
    '_ffi.pyi',
    'matrix.py',
    'primitive.py',
    'solver.py',
    'py.typed',
  ],
  subdir: 'pardiso_mkl_jax',
)

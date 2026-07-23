# Locate the XLA FFI headers bundled with the installed jaxlib and the MKL
# headers provided by mkl-devel. Both are ordinary Python packages, so their
# paths are discovered by asking the interpreter directly rather than
# hardcoding install locations.
jax_ffi_include_dir = run_command(python,
  ['-c', 'import jax; print(jax.ffi.include_dir())'],
  check: true,
).stdout().strip()

mkl_data_dir = run_command(python,
  ['-c', 'import sysconfig; print(sysconfig.get_path("data"))'],
  check: true,
).stdout().strip()

host_system = host_machine.system()

# The Linux mkl wheel installs straight into the environment's include/ and
# lib/ directories, but the Windows wheel uses the conda-style "Library"
# layout (Library/include, Library/lib, Library/bin) instead.
if host_system == 'windows'
  mkl_include_dir = mkl_data_dir / 'Library' / 'include'
  mkl_lib_dir = mkl_data_dir / 'Library' / 'lib'
else
  mkl_include_dir = mkl_data_dir / 'include'
  mkl_lib_dir = mkl_data_dir / 'lib'
endif

# include_directories() rejects absolute paths that happen to fall under the
# source tree, which .venv/include does whenever the virtual environment is
# created inside the project directory (the normal uv layout). Passing the
# include path as a raw compiler argument instead avoids that restriction,
# and works the same regardless of where .venv lives.
jax_ffi_dependency = declare_dependency(
  compile_args: ['-I' + jax_ffi_include_dir],
)

if host_system == 'windows'
  # Unlike the Linux .so, the Windows mkl-devel wheel ships an unversioned
  # mkl_rt.lib import library, so it resolves through the ordinary compiler
  # library search rather than needing the versioned-filename workaround
  # below.
  cpp_compiler = meson.get_compiler('cpp')
  mkl_rt_dependency = cpp_compiler.find_library('mkl_rt', dirs: [mkl_lib_dir])
  mkl_dependency = declare_dependency(
    compile_args: ['-I' + mkl_include_dir],
    dependencies: [mkl_rt_dependency],
  )
else
  # The mkl PyPI wheel ships only a versioned SONAME (e.g. libmkl_rt.so.3),
  # not an unversioned libmkl_rt.so symlink, so a plain -lmkl_rt does not
  # resolve at link time. Find the exact file and link it with the GNU ld
  # ":filename" syntax instead.
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

  # At runtime libmkl_rt is loaded explicitly (see _mkl_loader.py) rather
  # than through rpath, because the mkl wheel installs its shared libraries
  # directly under the environment's lib/ directory, not under
  # site-packages, and that directory can differ between an isolated build
  # environment and the final install environment.
  mkl_dependency = declare_dependency(
    compile_args: ['-I' + mkl_include_dir],
    link_args: ['-L' + mkl_lib_dir, '-l:' + mkl_rt_library_name],
  )
endif

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

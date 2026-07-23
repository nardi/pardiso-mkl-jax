"""Loads libmkl_rt before the compiled extension needs it.

The mkl PyPI wheel installs its shared libraries directly under the Python
environment's lib/ directory (or, on Windows, Library/bin/) rather than under
site-packages. A build-time rpath baked into the compiled extension would
point at wherever the package happened to be built, which is not necessarily
the same environment it ends up installed into, so it cannot be relied on.
Loading libmkl_rt globally here instead means the dynamic linker finds it
already resident in the process by the time the _ffi extension module is
imported, regardless of where either one was built or installed.
"""

import ctypes
import os
import pathlib
import sys


def load_libmkl_rt() -> ctypes.CDLL:
    """Load libmkl_rt globally so the compiled extension can resolve MKL symbols."""
    if sys.platform == "win32":
        # mkl_rt.dll depends on sibling DLLs (mkl_core, mkl_intel_thread,
        # etc.) in the same directory. Unlike Linux's dynamic linker, Windows
        # does not search a loaded DLL's own directory for its dependencies
        # unless that directory is explicitly registered first.
        library_directory = pathlib.Path(sys.prefix) / "Library" / "bin"
        candidates = sorted(library_directory.glob("mkl_rt.*.dll"))
        if not candidates:
            raise ImportError(
                f"could not find mkl_rt under {library_directory}, "
                "is the 'mkl' package installed in this environment?"
            )
        os.add_dll_directory(str(library_directory))
        # add_dll_directory only affects extension-module loading and ctypes
        # calls. oneMKL picks its threading-layer backend (e.g.
        # mkl_intel_thread.*.dll) with its own internal LoadLibrary call at
        # runtime, which uses the classic search order and so only finds
        # sibling MKL DLLs if their directory is actually on PATH.
        os.environ["PATH"] = str(library_directory) + os.pathsep + os.environ.get("PATH", "")
        return ctypes.CDLL(str(candidates[0]))

    library_directory = pathlib.Path(sys.prefix) / "lib"
    candidates = sorted(library_directory.glob("libmkl_rt.so*"))
    if not candidates:
        raise ImportError(
            f"could not find libmkl_rt under {library_directory}, "
            "is the 'mkl' package installed in this environment?"
        )
    return ctypes.CDLL(str(candidates[0]), mode=ctypes.RTLD_GLOBAL)


libmkl_rt = load_libmkl_rt()

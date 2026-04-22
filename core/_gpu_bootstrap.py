# _gpu_bootstrap.py — register pip-installed NVIDIA CUDA runtime DLLs.
#
# On Windows, the pip wheels `nvidia-*-cu12` ship the CUDA libraries
# (cudart, cublas, etc.) into <site-packages>/nvidia/<lib>/bin/, but those
# directories are NOT on the DLL search path by default, so cupy fails to
# load with "DLL load failed". This module walks the nvidia/ tree once at
# import time and registers each bin/ directory via os.add_dll_directory.
#
# Import this BEFORE importing cupy. It's a no-op on Linux/macOS or when
# the nvidia/ packages aren't installed.

import os
import sys


def _register_nvidia_dll_dirs() -> None:
    if sys.platform != "win32":
        return
    try:
        import nvidia  # type: ignore
    except ImportError:
        return
    bin_dirs = []
    for nvidia_root in nvidia.__path__:
        if not os.path.isdir(nvidia_root):
            continue
        for sub in os.listdir(nvidia_root):
            bin_dir = os.path.join(nvidia_root, sub, "bin")
            if os.path.isdir(bin_dir):
                bin_dirs.append(bin_dir)

    # add_dll_directory: covers Python-initiated LoadLibrary calls.
    if hasattr(os, "add_dll_directory"):
        for d in bin_dirs:
            try:
                os.add_dll_directory(d)
            except (OSError, FileNotFoundError):
                pass

    # Prepend to PATH: covers DLL-to-DLL loads (nvrtc loading its builtins).
    # add_dll_directory does NOT propagate to indirect loads.
    if bin_dirs:
        existing = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join(bin_dirs) + os.pathsep + existing


_register_nvidia_dll_dirs()

# source code in C (bttlThreshold): [PPSU2023](https://inria.hal.science/hal-04146298)


import ctypes
import os
import platform
import subprocess
import sys

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(MODULE_DIR, "../../../"))
PPSU_DIR = os.path.join(BASE_DIR, "PPSU2023")

LIB_NAME = (
    "libmatching_for_PPSU.dll" if sys.platform == "win32" else "libmatching_for_PPSU.so"
)
LIB_PATH = os.path.join(PPSU_DIR, LIB_NAME)
SETUPC_PATH = os.path.join(PPSU_DIR, "setupc.py")


def _build_env() -> dict[str, str]:
    """
    Build subprocess environment for compiling the PPSU shared library.

    Why this exists
    ---------------
    The PPSU shared object is a machine-specific compiled artifact. On macOS,
    especially Apple Silicon, stale or incorrectly-targeted binaries can be
    present in the working tree. Rebuilds must therefore preserve the caller's
    environment and, when appropriate, force an arm64 build so that ctypes can
    load the resulting library into the current Python process.
    """
    env: dict[str, str] = os.environ.copy()

    if sys.platform == "darwin" and platform.machine() == "arm64":
        arch_flag = "-arch arm64"
        env["ARCHFLAGS"] = arch_flag

        cflags = env.get("CFLAGS", "")
        ldflags = env.get("LDFLAGS", "")

        if arch_flag not in cflags:
            env["CFLAGS"] = f"{cflags} {arch_flag}".strip()
        if arch_flag not in ldflags:
            env["LDFLAGS"] = f"{ldflags} {arch_flag}".strip()

    return env


def build_shared_library() -> None:
    """
    Build the PPSU shared library in-place inside ``PPSU2023``.

    Why this exists
    ---------------
    The repository should not rely on a prebuilt shared object checked into the
    working tree because the binary is platform- and architecture-specific.
    This helper rebuilds the library deterministically from source and ensures
    the build is launched from the PPSU source directory rather than the caller's
    current working directory.
    """
    if not os.path.isfile(SETUPC_PATH):
        raise RuntimeError(f"setupc.py not found at expected path: {SETUPC_PATH}")

    if os.path.exists(LIB_PATH):
        os.remove(LIB_PATH)

    print(f"[INFO] Attempting to build shared library via {SETUPC_PATH}...")
    subprocess.run(
        [sys.executable, "setupc.py", "build_ext", "--inplace"],
        check=True,
        cwd=PPSU_DIR,
        env=_build_env(),
    )
    print("[INFO] Build completed.")


def load_shared_library() -> ctypes.CDLL:
    """
    Load the PPSU shared library, rebuilding it once if needed.

    Why this exists
    ---------------
    Import-time library loading is fragile when a stale binary from another
    architecture is present. This helper centralizes the recovery path so the
    module does not rely on whatever artifact happened to be checked into the
    tree previously.
    """
    try:
        return ctypes.CDLL(LIB_PATH)
    except OSError as exc:
        print(f"[WARNING] Failed to load shared library at {LIB_PATH}: {exc}")
        build_shared_library()
        return ctypes.CDLL(LIB_PATH)


# implementing lazy load to save time
_LIB = None


def _get_lib():
    global _LIB
    if _LIB is None:
        _LIB = load_shared_library()
        _LIB.bttlThreshold.argtypes = [
            ctypes.POINTER(ctypes.c_int),  # col_ptrs
            ctypes.POINTER(ctypes.c_int),  # col_ids
            ctypes.POINTER(ctypes.c_double),  # col_vals
            ctypes.c_int,  # n
            ctypes.c_int,  # m
            ctypes.POINTER(ctypes.c_int),  # match
            ctypes.POINTER(ctypes.c_int),  # row_match
            ctypes.POINTER(ctypes.c_int),  # row_ptrs
            ctypes.POINTER(ctypes.c_int),  # row_ids
            ctypes.POINTER(ctypes.c_double),  # row_vals
            ctypes.POINTER(ctypes.c_int),  # fend_cols
            ctypes.POINTER(ctypes.c_int),  # fend_rows
            ctypes.c_int,  # lbapAlone
            ctypes.POINTER(ctypes.c_double),  # thrshld_g
            ctypes.c_int,  # sprankknown
        ]
        _LIB.bttlThreshold.restype = ctypes.c_int  # number of iterations
    return _LIB


def bttl_threshold(col_ptrs, col_ids, col_vals, n, m, sprankknown=0, lbapAlone=1):
    """
    Python wrapper for the bttlThreshold function in the shared library.

    Args:
        col_ptrs (list[int]): Column pointers (CSR format).
        col_ids (list[int]): Row indices (CSR format).
        col_vals (list[float]): Edge weights in CSR format.
        n (int): Number of columns.
        m (int): Number of rows.
        sprankknown (int): Structural rank of the matrix (default: 0).

    Returns:
        dict: Matching results, including column-to-row, row-to-column mappings, and threshold.
    """
    lib = _get_lib()
    # Convert inputs to ctypes
    col_ptrs = (ctypes.c_int * len(col_ptrs))(*col_ptrs)
    col_ids = (ctypes.c_int * len(col_ids))(*col_ids)
    col_vals = (ctypes.c_double * len(col_vals))(*col_vals)
    match = (ctypes.c_int * n)(-1)  # Initialize match array with -1
    row_match = (ctypes.c_int * m)(-1)  # Initialize row_match array with -1
    row_ptrs = (ctypes.c_int * (m + 1))()  # Placeholder for row pointers
    row_ids = (ctypes.c_int * len(col_ids))()  # Placeholder for row indices
    row_vals = (ctypes.c_double * len(col_vals))()  # Placeholder for row values
    fend_cols = (ctypes.c_int * n)()  # Placeholder for fend_cols
    fend_rows = (ctypes.c_int * m)()  # Placeholder for fend_rows
    thrshld_g = ctypes.c_double()  # Threshold value

    iterations = lib.bttlThreshold(
        col_ptrs,
        col_ids,
        col_vals,
        ctypes.c_int(n),
        ctypes.c_int(m),
        match,
        row_match,
        row_ptrs,
        row_ids,
        row_vals,
        fend_cols,
        fend_rows,
        ctypes.c_int(lbapAlone),
        ctypes.byref(thrshld_g),
        sprankknown,
    )

    return {
        "iterations": iterations,
        "match": list(match),
        "row_match": list(row_match),
        "threshold": thrshld_g.value,
    }

"""pytest configuration: make the repository root importable.

block-DDA_Py is not an installed package, so the test modules import
``bl_dda``, ``bl_krylov``, ``mvp_fft`` and ``shape_model`` from the repository
root. pytest itself puts ``tests/`` on ``sys.path`` (rootdir conftest), which
makes the helper module ``shifted_cases`` importable.
"""
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

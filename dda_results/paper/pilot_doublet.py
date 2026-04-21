"""Pilot HDF5: 2-sphere doublet, n=1.5+0.01i, a_eq=0.05 μm only.
Smoke-test for the doublet pipeline (shape build, run_dda dispatch,
rhs-scaling diagnostic).
Run:  PYTHONPATH=. .venv/bin/python -m dda_results.paper.pilot_doublet
"""
import os
from dda_results.paper._common import create_paper_h5, N_LOW

if __name__ == "__main__":
    here = os.path.dirname(__file__)
    create_paper_h5(
        os.path.join(here, "pilot_doublet.hdf5"),
        m_p        = N_LOW,
        a_eq_list  = [0.05],
        bc_ratio   = 1.0,
        ab_ratio   = 1.0,
        gre_beta   = 0.0,
        shape_kind = "doublet",
        light_source = "(pilot) λ=0.638 μm, n=1.5+0.01i, doublet a_eq=0.05",
    )

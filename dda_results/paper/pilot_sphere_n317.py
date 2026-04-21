"""Pilot HDF5: single sphere, n=3.17+0.16i, a_eq=0.1 μm only.
Used to validate estimate → run_dda → check_h5 small-loop before
launching the full paper sweep.
Run:  PYTHONPATH=. .venv/bin/python -m dda_results.paper.pilot_sphere_n317
"""
import os
from dda_results.paper._common import create_paper_h5, N_HIGH

if __name__ == "__main__":
    here = os.path.dirname(__file__)
    create_paper_h5(
        os.path.join(here, "pilot_sphere_n317.hdf5"),
        m_p       = N_HIGH,
        a_eq_list = [0.1],
        bc_ratio  = 1.0,
        ab_ratio  = 1.0,
        gre_beta  = 0.0,
        light_source = "(pilot) λ=0.638 μm, n=3.17+0.16i, sphere a_eq=0.1",
    )

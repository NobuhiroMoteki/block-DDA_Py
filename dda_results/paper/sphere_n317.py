"""Single sphere, high-index dielectric (n = 3.17 + 0.16i) @ λ = 0.638 μm.
Run:  PYTHONPATH=. .venv/bin/python -m dda_results.paper.sphere_n317
"""
import os
from dda_results.paper._common import create_paper_h5, N_HIGH, A_EQ_FULL

if __name__ == "__main__":
    here = os.path.dirname(__file__)
    create_paper_h5(
        os.path.join(here, "sphere_n317.hdf5"),
        m_p       = N_HIGH,
        a_eq_list = A_EQ_FULL,
        bc_ratio  = 1.0,
        ab_ratio  = 1.0,
        gre_beta  = 0.0,
        light_source = "(paper) λ=0.638 μm, n=3.17+0.16i, sphere",
    )

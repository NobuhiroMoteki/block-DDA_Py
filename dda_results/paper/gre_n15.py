"""GRE (a/b = b/c = 1, β_gre = 0.2), low-index (n = 1.5 + 0.01i) @ λ = 0.638 μm.
Run:  PYTHONPATH=. .venv/bin/python -m dda_results.paper.gre_n15
"""
import os
from dda_results.paper._common import create_paper_h5, N_LOW, A_EQ_FULL

if __name__ == "__main__":
    here = os.path.dirname(__file__)
    create_paper_h5(
        os.path.join(here, "gre_n15.hdf5"),
        m_p       = N_LOW,
        a_eq_list = A_EQ_FULL,
        bc_ratio  = 1.0,
        ab_ratio  = 1.0,
        gre_beta  = 0.2,
        light_source = "(paper) λ=0.638 μm, n=1.5+0.01i, GRE β=0.2",
    )

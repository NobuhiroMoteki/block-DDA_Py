"""Single sphere, Au (J&C 1972: n = 0.17525 + 3.4830i) @ λ = 0.638 μm.
Au sweep is truncated to a_eq ≤ 0.2 μm (CLAUDE.md §2).
Run:  PYTHONPATH=. .venv/bin/python -m dda_results.paper.sphere_Au
"""
import os
from dda_results.paper._common import create_paper_h5, N_AU, A_EQ_AU

if __name__ == "__main__":
    here = os.path.dirname(__file__)
    create_paper_h5(
        os.path.join(here, "sphere_Au.hdf5"),
        m_p       = N_AU,
        a_eq_list = A_EQ_AU,
        bc_ratio  = 1.0,
        ab_ratio  = 1.0,
        gre_beta  = 0.0,
        light_source = "(paper) λ=0.638 μm, Au (J&C 1972), sphere",
    )

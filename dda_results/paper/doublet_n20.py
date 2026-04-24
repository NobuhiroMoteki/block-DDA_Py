"""2-sphere doublet, mid-index (n = 2.0 + 0.0i) @ λ = 0.638 μm.
Touching equal-sphere doublet aligned along particle z; monomer
radius R = a_eq / 2^(1/3); gap = 0.1 R (CLAUDE.md §2.4).
Run:  PYTHONPATH=. .venv/bin/python -m dda_results.paper.doublet_n20
"""
import os
from dda_results.paper._common import create_paper_h5, N_20, A_EQ_FULL

if __name__ == "__main__":
    here = os.path.dirname(__file__)
    create_paper_h5(
        os.path.join(here, "doublet_n20.hdf5"),
        m_p        = N_20,
        a_eq_list  = A_EQ_FULL,
        bc_ratio   = 1.0,
        ab_ratio   = 1.0,
        gre_beta   = 0.0,
        shape_kind = "doublet",
        light_source = "(paper) λ=0.638 μm, n=2.0+0.0i, doublet (gap=0.1R)",
    )

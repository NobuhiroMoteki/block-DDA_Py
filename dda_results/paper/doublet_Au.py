"""2-sphere doublet, Au (J&C 1972) @ λ = 0.638 μm.
Touching equal-sphere doublet aligned along particle z; monomer
radius R = a_eq / 2^(1/3); gap = 0.1 R (CLAUDE.md §2.4).
Au sweep is truncated to a_eq ≤ 0.2 μm.
Run:  PYTHONPATH=. .venv/bin/python -m dda_results.paper.doublet_Au
"""
import os
from dda_results.paper._common import create_paper_h5, N_AU, A_EQ_AU

if __name__ == "__main__":
    here = os.path.dirname(__file__)
    create_paper_h5(
        os.path.join(here, "doublet_Au.hdf5"),
        m_p        = N_AU,
        a_eq_list  = A_EQ_AU,
        bc_ratio   = 1.0,
        ab_ratio   = 1.0,
        gre_beta   = 0.0,
        shape_kind = "doublet",
        light_source = "(paper) λ=0.638 μm, Au (J&C 1972), doublet (gap=0.1R)",
    )

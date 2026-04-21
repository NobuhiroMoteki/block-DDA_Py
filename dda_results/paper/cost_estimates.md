# Paper sweep cost estimates (CLAUDE.md §5)

Pre-run estimates captured by `scripts/estimate_cost.py` before each
production slot launch. Records worst-case shape × (wl × m_p) per file.

Columns: `slot = (i_rv, i_bc, i_ab, i_bt)`, **peak RSS** and **t_total**
are the sweep worst case across shape slots.

| file | shape_kind | spheroid | peak RSS | t_sweep_est | t_actual | converged? | notes |
|------|------------|----------|----------|-------------|----------|------------|-------|

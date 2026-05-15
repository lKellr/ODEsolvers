# TODO
- rewrite in C++

## Extrapolation
- rational extrapolation is not working good enough? Why is ODEX rational so good in convergence case (too good?) but so bad in extrapoaltion case? Note: it is good because of the different step sequence!
- Deuflhard control not working at k=1
- Deuflhard control correctly with information theory, storing which errors fail how often
- PI control
- Shampine SIMP control
- monitor growth of basic:scheme solutions? Early versions of METAN1

- find good default values for implicit_rel_costs (will depend on the equation, maybe even provide a function to find it automatically?)

- recompute Jacobian only if theta is above some tolerance
- step shortening (at end) handled by controller

## Tests
- profile extrapolation code: norm and product in step control is expensive -> NUMBA
- test small table size -> indexing errors

# Notes
- AB convergence rate is not better than starter order (Euler or Midpoint)
- ODEX use order 2*(k+1) for linear problems, 2*k + 1 on nonlinear ones

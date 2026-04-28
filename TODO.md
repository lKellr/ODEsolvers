# TODO
- rewrite in C++

## Extrapolation
- errors are larger than tolerance
- SODEX is not working
- find good default values for implicit_rel_costs (will depend on the equation, maybe even provide a function to find it automatically?)
- convergence: compare actual h!

- recompute Jacobian only if theta is above some tolerance
- step shortening (at end) handled by controller
- controller PI

## Tests
- test implicit extrapolation schemes
- convergence checks: h- and k-first
- profile extrapolation code: norm and product in step control is expensive -> NUMBA
- test small table size -> indexing errors
- test rational extrapolation

# Notes
- AB convergence rate is not better than starter order (Euler or Midpoint)

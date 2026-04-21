# TODO

## Extrapolation
- controller prefers too low order
- ODEX smoothed does not work
- SODEX is not working
- check indices in controller
- find good default values for implicit_rel_costs (will depend on the equation, maybe even provide a function to find it automatically?)
- convergence: compare actual h!
- recompute Jacobian only if theta is above some tolerance
- allow for early checks when base scheme diverges

## Tests
- test implicit extrapolation schemes
- convergence checks: h- and k-first
- profile extrapolation code: nrom and product in step control is expensive -> NUMBA
- test small table size -> indexing errors

# Notes
- AB convergence rate is not better than starter order (Euler or Midpoint)

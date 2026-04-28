# TODO
- rewrite in C++

## Extrapolation
- SODEX is not working
- check indices in controller
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




- h order increase and p order icnrease have different results for stability; k limiting might lead to parameter set that is not expxpected to converge -> set h according to limited k? But for h limiting we do the reverse?, (either limit before finding s), or just exclude the increase options! Can this error even appear? Error should be greater than estiamted in the last step!
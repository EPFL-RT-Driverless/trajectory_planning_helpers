# v2.0.6

added `COLCON_IGNORE` file to ignore this package in colcon builds (for brains repo)

# v2.0.5

Optimized functions `calc_splines`, `interp_track_widths`, `calc_spline_lengths`, `interp_splines`.

# v2.0.4

Fixed small bug in `local_path_matching()` to be able to localize points after the last
point in the track. 

# v2.0.3

fixed a small bug in `opt_min_curv` in the prints

# v2.0.2
 
Fixed a bug in `calc_splines` and only left `quadprog` as dependency for curvature minimization

# v2.0.1

Fixed import of git deps in `setup.py`

# v2.0.0

Updated the dependency system to full pip and updated the repo to match
`python_boilerplate` v2.0.1

# v1.2.0

Changed the interface of some functions

# v1.1.0

added splunif to the stack

# v1.0.0

Upgraded the import system and added an option in `opt_min_curv` to choose the QP
solver.

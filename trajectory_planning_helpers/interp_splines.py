import math

import numpy as np

from .calc_spline_lengths import calc_spline_lengths


def interp_splines(
    coeffs_x: np.ndarray,
    coeffs_y: np.ndarray,
    spline_lengths: np.ndarray = None,
    closed: bool = True,
    stepsize_approx: float = None,
    stepnum_fixed: list = None,
) -> tuple:
    """
    author:
    Alexander Heilmeier & Tim Stahl

    .. description::
    Interpolate points on one or more splines with third order. The last point (i.e. t = 1.0)
    can be included if option is set accordingly (should be prevented for a closed raceline in most cases). The
    algorithm keeps stepsize_approx as good as possible.

    .. inputs::
    :param coeffs_x:        coefficient matrix of the x splines with size (no_splines x 4).
    :type coeffs_x:         np.ndarray
    :param coeffs_y:        coefficient matrix of the y splines with size (no_splines x 4).
    :type coeffs_y:         np.ndarray
    :param spline_lengths:  array containing the lengths of the inserted splines with size (no_splines, ).
    :type spline_lengths:   np.ndarray
    :param closed:          whether the path should be considered as closed or not
    :type closed:           bool
    :param stepsize_approx: desired stepsize of the points after interpolation.                      \\ Provide only one
    :type stepsize_approx:  float
    :param stepnum_fixed:   return a fixed number of coordinates per spline, list of length no_splines. \\ of these two!
    :type stepnum_fixed:    list

    .. outputs::
    :return path_interp:    interpolated path points.
    :rtype path_interp:     np.ndarray
    :return spline_inds:    contains the indices of the splines that hold the interpolated points.
    :rtype spline_inds:     np.ndarray
    :return t_values:       containts the relative spline coordinate values (t) of every point on the splines.
    :rtype t_values:        np.ndarray
    :return dists_interp:   total distance up to every interpolation point.
    :rtype dists_interp:    np.ndarray

    .. notes::
    len(coeffs_x) = len(coeffs_y) = len(spline_lengths)

    len(path_interp) = len(spline_inds) = len(t_values) = len(dists_interp)
    """

    # ------------------------------------------------------------------------------------------------------------------
    # INPUT CHECKS -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # check sizes
    if coeffs_x.shape[0] != coeffs_y.shape[0]:
        raise RuntimeError("Coefficient matrices must have the same length!")

    if spline_lengths is not None and coeffs_x.shape[0] != spline_lengths.size:
        raise RuntimeError("coeffs_x/y and spline_lengths must have the same length!")

    # check if coeffs_x and coeffs_y have exactly two dimensions and raise error otherwise
    if not (coeffs_x.ndim == 2 and coeffs_y.ndim == 2):
        raise RuntimeError("Coefficient matrices do not have two dimensions!")

    # check if step size specification is valid
    if (stepsize_approx is None and stepnum_fixed is None) or (
        stepsize_approx is not None and stepnum_fixed is not None
    ):
        raise RuntimeError(
            "Provide one of 'stepsize_approx' and 'stepnum_fixed' and set the other to 'None'!"
        )

    if stepnum_fixed is not None and len(stepnum_fixed) != coeffs_x.shape[0]:
        raise RuntimeError(
            "The provided list 'stepnum_fixed' must hold an entry for every spline!"
        )

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE NUMBER OF INTERPOLATION POINTS AND ACCORDING DISTANCES -------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if stepsize_approx is not None:
        # get the total distance up to the end of every spline (i.e. cumulated distances)
        if spline_lengths is None:
            spline_lengths = calc_spline_lengths(
                coeffs_x=coeffs_x, coeffs_y=coeffs_y, quickndirty=False
            )

        dists_cum = np.cumsum(spline_lengths)

        # calculate number of interpolation points and distances (+1 because last point is included at first)
        no_interp_points = math.ceil(dists_cum[-1] / stepsize_approx) + 1
        dists_interp = np.linspace(0.0, dists_cum[-1], no_interp_points)

    else:
        # get total number of points to be sampled (subtract overlapping points)
        no_interp_points = sum(stepnum_fixed) - (len(stepnum_fixed) - 1)
        dists_interp = None

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE INTERMEDIATE STEPS -------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # create arrays to save the values
    path_interp = np.zeros((no_interp_points, 2))  # raceline coords (x, y) array
    # path_interp2 = np.zeros((no_interp_points, 2))  # raceline coords (x, y) array
    spline_inds = np.zeros(
        no_interp_points, dtype=int
    )  # save the spline index to which a point belongs
    t_values = np.zeros(no_interp_points)  # save t values
    # t_values2 = np.zeros(no_interp_points)  # save t values

    if stepsize_approx is not None:  # always True in our implementation

        # --------------------------------------------------------------------------------------------------------------
        # APPROX. EQUAL STEP SIZE ALONG PATH OF ADJACENT SPLINES -------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # create steps with stepsize_approx

        # find the spline that hosts the current interpolation point
        j = np.argmax(dists_interp[:, np.newaxis] < dists_cum, axis=1)
        spline_inds = np.copy(j)

        # get spline t value depending on the progress within the current element
        t_values[j > 0] = (
            dists_interp[j > 0] - dists_cum[j - 1][j > 0]
        ) / spline_lengths[j][j > 0]
        t_values[j == 0] = dists_interp[j == 0] / spline_lengths[0]
        t_values[-1] = 0.0

        # calculate coords
        path_interp[:, 0] = coeffs_x[j][:, 0]

        path_interp[:, 0] += coeffs_x[j][:, 1] * t_values
        path_interp[:, 0] += coeffs_x[j][:, 2] * np.power(t_values, 2)
        path_interp[:, 0] += coeffs_x[j][:, 3] * np.power(t_values, 3)

        path_interp[:, 1] = coeffs_y[j][:, 0]

        path_interp[:, 1] += coeffs_y[j][:, 1] * t_values
        path_interp[:, 1] += coeffs_y[j][:, 2] * np.power(t_values, 2)
        path_interp[:, 1] += coeffs_y[j][:, 3] * np.power(t_values, 3)

        path_interp[-1][0] = 0.0
        path_interp[-1][1] = 0.0

    else:

        # --------------------------------------------------------------------------------------------------------------
        # FIXED STEP SIZE FOR EVERY SPLINE SEGMENT ---------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        j = 0

        for i in range(len(stepnum_fixed)):
            # skip last point except for last segment
            if i < len(stepnum_fixed) - 1:
                t_values[j : (j + stepnum_fixed[i] - 1)] = np.linspace(
                    0, 1, stepnum_fixed[i]
                )[:-1]
                spline_inds[j : (j + stepnum_fixed[i] - 1)] = i
                j += stepnum_fixed[i] - 1

            else:
                t_values[j : (j + stepnum_fixed[i])] = np.linspace(
                    0, 1, stepnum_fixed[i]
                )
                spline_inds[j : (j + stepnum_fixed[i])] = i
                j += stepnum_fixed[i]

        t_set = np.column_stack(
            (
                np.ones(no_interp_points),
                t_values,
                np.power(t_values, 2),
                np.power(t_values, 3),
            )
        )

        # remove overlapping samples
        n_samples = np.array(stepnum_fixed)
        n_samples[:-1] -= 1

        path_interp[:, 0] = np.sum(
            np.multiply(np.repeat(coeffs_x, n_samples, axis=0), t_set), axis=1
        )
        path_interp[:, 1] = np.sum(
            np.multiply(np.repeat(coeffs_y, n_samples, axis=0), t_set), axis=1
        )

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE LAST POINT IF REQUIRED (t = 1.0) -----------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if closed:
        path_interp = path_interp[:-1]
        spline_inds = spline_inds[:-1]
        t_values = t_values[:-1]

        if dists_interp is not None:
            dists_interp = dists_interp[:-1]

    else:
        path_interp[-1, 0] = np.sum(coeffs_x[-1])
        path_interp[-1, 1] = np.sum(coeffs_y[-1])
        spline_inds[-1] = coeffs_x.shape[0] - 1
        t_values[-1] = 1.0

    # NOTE: dists_interp is None, when using a fixed step size
    return path_interp, spline_inds, t_values, dists_interp

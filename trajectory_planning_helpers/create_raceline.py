import numpy as np
from .calc_splines import calc_splines
from .calc_spline_lengths import calc_spline_lengths
from .interp_splines import interp_splines
from .calc_normal_vectors_ahead import calc_normal_vectors_ahead

# from time import perf_counter

def create_raceline(
    refline: np.ndarray,
    normvectors: np.ndarray,
    alpha: np.ndarray,
    stepsize_interp: float,
    closed: bool = True,
    psi_s: float = None,
    psi_e: float = None,
) -> tuple:
    """
    author:
    Alexander Heilmeier

    .. description::
    This function includes the algorithm part connected to the interpolation of the raceline after the optimization.

    .. inputs::
    :param refline:         array containing the track reference line [x, y] (unit is meter, must be unclosed!)
    :type refline:          np.ndarray
    :param normvectors:     normalized normal vectors for every point of the reference line [x_component, y_component]
                            (unit is meter, must be unclosed! must be the same length as refline!)
    :type normvectors:      np.ndarray
    :param alpha:           solution vector of the optimization problem containing the lateral shift in m for every point.
    :type alpha:            np.ndarray
    :param stepsize_interp: stepsize in meters which is used for the interpolation after the raceline creation.
    :type stepsize_interp:  float
    :param closed:          whether the track is closed or not

    :param psi_s:           heading at the start of the raceline, must be specified if closed
    :param psi_e:

    .. outputs::
    :return raceline_interp:                interpolated raceline [x, y] in m.
    :rtype raceline_interp:                 np.ndarray
    :return A_raceline:                     linear equation system matrix of the splines on the raceline.
    :rtype A_raceline:                      np.ndarray
    :return coeffs_x_raceline:              spline coefficients of the x-component.
    :rtype coeffs_x_raceline:               np.ndarray
    :return coeffs_y_raceline:              spline coefficients of the y-component.
    :rtype coeffs_y_raceline:               np.ndarray
    :return normvectors_raceline:           normalized normal vectors for every point of the raceline [x_component, y_component]
    :rtype normvectors_raceline:            np.ndarray
    :return spline_inds_raceline_interp:    contains the indices of the splines that hold the interpolated points.
    :rtype spline_inds_raceline_interp:     np.ndarray
    :return t_values_raceline_interp:       containts the relative spline coordinate values (t) of every point on the
                                            splines.
    :rtype t_values_raceline_interp:        np.ndarray
    :return s_raceline_interp:              total distance in m (i.e. s coordinate) up to every interpolation point.
    :rtype s_raceline_interp:               np.ndarray
    :return spline_lengths_raceline:        lengths of the splines on the raceline in m.
    :rtype spline_lengths_raceline:         np.ndarray
    :return el_lengths_raceline_interp_cl:  distance between every two points on interpolated raceline in m (closed!).
    :rtype el_lengths_raceline_interp_cl:   np.ndarray
    """

    # calculate raceline on the basis of the optimized alpha values
    raceline = refline + np.expand_dims(alpha, 1) * normvectors

    # t1 = perf_counter()
    (
        coeffs_x_raceline,
        coeffs_y_raceline,
        A_raceline,
        normvectors_raceline,
    ) = calc_splines(
        path=raceline,
        use_dist_scaling=True,
        closed=closed,
        psi_s=psi_s,
        psi_e=psi_e,
    )
    # print("    calc_splines {} ms".format((perf_counter() - t1)*1000))
    if not closed:
        normvectors_raceline = np.vstack(
            (normvectors_raceline, calc_normal_vectors_ahead(psi_e))
        )

    # calculate new spline lengths
    # t1 = perf_counter()
    spline_lengths_raceline = calc_spline_lengths(
        coeffs_x=coeffs_x_raceline, coeffs_y=coeffs_y_raceline
    )
    # print("    calc_spline_lengths {} ms".format((perf_counter() - t1) * 1000))
    # interpolate splines for evenly spaced raceline points
    # t1 = perf_counter()
    (
        raceline_interp,
        spline_inds_raceline_interp,
        t_values_raceline_interp,
        s_raceline_interp,
    ) = interp_splines(
        coeffs_x=coeffs_x_raceline,
        coeffs_y=coeffs_y_raceline,
        spline_lengths=spline_lengths_raceline,
        closed=closed,
        stepsize_approx=stepsize_interp,
    )
    # print("    interp_splines {} ms".format((perf_counter() - t1) * 1000))
    # calculate element lengths
    if not closed:
        s_tot_raceline = s_raceline_interp[-1]
    else:
        s_tot_raceline = float(np.sum(spline_lengths_raceline))

    el_lengths_raceline_interp = np.diff(s_raceline_interp)
    if closed:
        el_lengths_raceline_interp = np.append(
            el_lengths_raceline_interp, s_tot_raceline - s_raceline_interp[-1]
        )

    return (
        raceline_interp,
        A_raceline,
        coeffs_x_raceline,
        coeffs_y_raceline,
        normvectors_raceline,
        spline_inds_raceline_interp,
        t_values_raceline_interp,
        s_raceline_interp,
        spline_lengths_raceline,
        el_lengths_raceline_interp,
    )

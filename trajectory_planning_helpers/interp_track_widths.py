import numpy as np
from time import perf_counter

def interp_track_widths(
    w_track: np.ndarray,
    spline_inds: np.ndarray,
    t_values: np.ndarray,
    incl_last_point: bool = False,
) -> np.ndarray:
    """
    author:
    Alexander Heilmeier

    .. description::
    The function (linearly) interpolates the track widths in the same steps as the splines were interpolated before.

    Keep attention that the (multiple) interpolation of track widths can lead to unwanted effects, e.g. that peaks
    in the track widths can disappear if the stepsize is too large (kind of an aliasing effect).

    .. inputs::
    :param w_track:         array containing the track widths in meters [w_track_right, w_track_left] to interpolate,
                            optionally with banking angle in rad: [w_track_right, w_track_left, banking]
    :type w_track:          np.ndarray
    :param spline_inds:     indices that show which spline (and here w_track element) shall be interpolated.
    :type spline_inds:      np.ndarray
    :param t_values:        relative spline coordinate values (t) of every point on the splines specified by spline_inds
    :type t_values:         np.ndarray
    :param incl_last_point: bool flag to show if last point should be included or not.
    :type incl_last_point:  bool

    .. outputs::
    :return w_track_interp: array with interpolated track widths (and optionally banking angle).
    :rtype w_track_interp:  np.ndarray

    .. notes::
    All inputs are unclosed.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE INTERMEDIATE STEPS -------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    w_track_cl = np.vstack((w_track, w_track[0]))
    no_interp_points = t_values.size  # unclosed

    if incl_last_point:
        w_track_interp = np.zeros((no_interp_points + 1, w_track.shape[1]))
        w_track_interp[-1] = w_track_cl[-1]
    else:
        w_track_interp = np.zeros((no_interp_points, w_track.shape[1]))

    # loop through every interpolation point
    # t1 = perf_counter()
    # for i in range(no_interp_points):
    #     # find the spline that hosts the current interpolation point
    #     ind_spl = spline_inds[i]
    #
    #     # calculate track widths (linear approximation assumed along one spline)
    #     w_track_interp[i, 0] = np.interp(
    #         t_values[i], (0.0, 1.0), w_track_cl[ind_spl : ind_spl + 2, 0]
    #     )
    #     w_track_interp[i, 1] = np.interp(
    #         t_values[i], (0.0, 1.0), w_track_cl[ind_spl : ind_spl + 2, 1]
    #     )
    #
    #     if w_track.shape[1] == 3:
    #         w_track_interp[i, 2] = np.interp(
    #             t_values[i], (0.0, 1.0), w_track_cl[ind_spl : ind_spl + 2, 2]
    #         )
    # print("    for loop : {} ms".format((perf_counter() - t1) * 1000))

    #print(w_track_cl[spline_inds : spline_inds + 2].shape)
    # t1 = perf_counter()
    w_track_cl_m = np.array([w_track_cl[spline_inds], w_track_cl[spline_inds + 1]]).transpose(1, 0, 2)

    def multiInterp(x, fp):
        i = np.arange(x.size)
        j = 0
        d = x
        return (1 - d) * fp[i, j] + fp[i, j + 1] * d

    #xp = np.array([np.zeros(no_interp_points), np.ones(no_interp_points)]).T
    w_track_interp[:, 0] = multiInterp(t_values, w_track_cl_m[:,0])
    w_track_interp[:, 1] = multiInterp(t_values, w_track_cl_m[:, 1])
    if (w_track.shape[1] == 3):
        w_track_interp[:, 2] = multiInterp(t_values, w_track_cl_m[:, 2])
    #print("    numpy power : {} ms".format((perf_counter() - t1) * 1000))
    #print("    diff : {}".format(np.linalg.norm(w_track_interp2 - w_track_interp)))
    return w_track_interp

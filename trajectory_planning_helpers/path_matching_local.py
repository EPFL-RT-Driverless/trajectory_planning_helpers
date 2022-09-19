import numpy as np
from .angle3pt import angle3pt
from typing import Union


def path_matching_local(
    path: np.ndarray,
    ego_position: np.ndarray,
    consider_as_closed: bool = False,
    s_tot: Union[float, None] = None,
) -> tuple:
    """
    author:
    Alexander Heilmeier

    .. description::
    Get the corresponding s coordinate and the displacement of the own vehicle in relation to a local path.

    .. inputs::
    :param path:                Unclosed path used to match ego position ([s, x, y]).
    :type path:                 np.ndarray
    :param ego_position:        Ego position of the vehicle ([x, y]).
    :type ego_position:         np.ndarray
    :param consider_as_closed:  If the path is closed in reality we can interpolate between last and first point. This
                                can be enforced by setting consider_as_closed = True.
    :type consider_as_closed:   bool
    :param s_tot:               Total length of path in m.
    :type s_tot:                Union[float, None]

    .. outputs::
    :return s_interp:           Interpolated s position of the vehicle in m.
    :rtype s_interp:            np.ndarray
    :return d_displ:            Estimated displacement from the trajectory in m.
    :rtype d_displ:             np.ndarray
    """

    # ------------------------------------------------------------------------------------------------------------------
    # CHECK INPUT ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if path.shape[1] != 3:
        raise RuntimeError("Inserted path must have 3 columns [s, x, y]!")

    if consider_as_closed and s_tot is None:
        print(
            "WARNING: s_tot is not handed into path_matching_local function! Estimating s_tot on the basis of equal"
            "stepsizes"
        )
        s_tot = path[-1, 0] + path[1, 0] - path[0, 0]  # assume equal stepsize

    # ------------------------------------------------------------------------------------------------------------------
    # SELF LOCALIZATION ON RACELINE ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # get the nearest path point to ego position
    dists_to_cg = np.hypot(path[:, 1] - ego_position[0], path[:, 2] - ego_position[1])
    ind_min = np.argpartition(dists_to_cg, 1)[0]

    # get previous and following point on path
    if consider_as_closed:
        if ind_min == 0:
            ind_prev = dists_to_cg.shape[0] - 1
            ind_follow = 1

        elif ind_min == dists_to_cg.shape[0] - 1:
            ind_prev = ind_min - 1
            ind_follow = 0

        else:
            ind_prev = ind_min - 1
            ind_follow = ind_min + 1

    else:
        ind_prev = max(ind_min - 1, 0)
        ind_follow = min(ind_min + 1, dists_to_cg.shape[0] - 1)

    # get angle between selected point and neighbours
    ang_prev = np.abs(angle3pt(path[ind_min, 1:], ego_position, path[ind_prev, 1:]))

    ang_follow = np.abs(angle3pt(path[ind_min, 1:], ego_position, path[ind_follow, 1:]))

    # extract neighboring points -> closest point and the point resulting in the larger angle
    if ang_prev > ang_follow:
        a_pos = path[ind_prev, 1:]
        b_pos = path[ind_min, 1:]
        s_curs = np.append(path[ind_prev, 0], path[ind_min, 0])
    else:
        a_pos = path[ind_min, 1:]
        b_pos = path[ind_follow, 1:]
        s_curs = np.append(path[ind_min, 0], path[ind_follow, 0])

    # adjust s if closed path shell be considered and we have the case of interpolation between last and first point
    if consider_as_closed:
        if ind_min == 0 and ang_prev > ang_follow:
            s_curs[1] = s_tot
        elif ind_min == dists_to_cg.shape[0] - 1 and ang_prev <= ang_follow:
            s_curs[1] = s_tot

    # project the ego position onto the line between the two points
    dx = b_pos[0] - a_pos[0]
    dy = b_pos[1] - a_pos[1]
    lam = (dx * (ego_position[0] - a_pos[0]) + dy * (ego_position[1] - a_pos[1])) / (
        dx**2 + dy**2
    )

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE REQUIRED INFORMATION -----------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # calculate current path length
    s_interp = s_curs[0] + lam * (s_curs[1] - s_curs[0])

    # get displacement between ego position and path (needed for lookahead distance)
    x_proj = a_pos[0] + lam * dx
    y_proj = a_pos[1] + lam * dy

    # try:
    #     d_displ = np.sqrt(
    #         np.linalg.norm(ego_position - a_pos) ** 2
    #         - lam**2 * np.linalg.norm(b_pos - a_pos) ** 2
    #     )
    # except RuntimeWarning:
    #     d_displ = 0.0
    #     print(
    #         np.linalg.norm(ego_position - a_pos) ** 2,
    #         lam**2 * np.linalg.norm(b_pos - a_pos) ** 2,
    #     )
    d_displ = np.sqrt((ego_position - x_proj) ** 2 + (ego_position - y_proj) ** 2)

    return s_interp, d_displ

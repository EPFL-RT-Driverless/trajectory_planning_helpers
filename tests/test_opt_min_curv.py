import numpy as np
import os
import matplotlib.pyplot as plt

from trajectory_planning_helpers import calc_splines, opt_min_curv

if __name__ == "__main__":

    # --- PARAMETERS ---
    CLOSED = False

    # --- IMPORT TRACK ---
    # load data from csv file
    csv_data_temp = np.loadtxt(
        os.path.join(os.path.dirname(__file__), "example_files/berlin_2018.csv"),
        comments="#",
        delimiter=",",
    )

    # get coords and track widths out of array
    reftrack = csv_data_temp[:, 0:4]
    psi_s = 0.0
    psi_e = 2.0

    # --- CALCULATE MIN CURV ---
    if CLOSED:
        coeffs_x, coeffs_y, M, normvec_norm = calc_splines(
            path=np.vstack((reftrack[:, 0:2], reftrack[0, 0:2]))
        )
    else:
        reftrack = reftrack[200:600, :]
        coeffs_x, coeffs_y, M, normvec_norm = calc_splines(
            path=reftrack[:, 0:2], psi_s=psi_s, psi_e=psi_e
        )

        # extend norm-vec to same size of ref track (quick fix for testing only)
        normvec_norm = np.vstack((normvec_norm[0, :], normvec_norm))

    alpha_mincurv, curv_error_max = opt_min_curv(
        reftrack=reftrack,
        normvectors=normvec_norm,
        A=M,
        kappa_bound=0.4,
        w_veh=2.0,
        closed=CLOSED,
        psi_s=psi_s,
        psi_e=psi_e,
        print_debug=True,
        method="quadprog",
    )

    # --- PLOT RESULTS ---
    path_result = reftrack[:, 0:2] + normvec_norm * np.expand_dims(
        alpha_mincurv, axis=1
    )
    bound1 = reftrack[:, 0:2] - normvec_norm * np.expand_dims(reftrack[:, 2], axis=1)
    bound2 = reftrack[:, 0:2] + normvec_norm * np.expand_dims(reftrack[:, 3], axis=1)

    plt.plot(reftrack[:, 0], reftrack[:, 1], ":")
    plt.plot(path_result[:, 0], path_result[:, 1])
    plt.plot(bound1[:, 0], bound1[:, 1], "k")
    plt.plot(bound2[:, 0], bound2[:, 1], "k")
    plt.axis("equal")
    plt.show()

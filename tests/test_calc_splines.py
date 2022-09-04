import matplotlib.pyplot as plt
import numpy as np

from trajectory_planning_helpers import calc_splines, interp_splines

if __name__ == "__main__":

    path_coords = np.array([[50.0, 10.0], [10.0, 4.0], [0.0, 0.0]])
    psi_s_ = np.pi / 2.0
    psi_e_ = np.pi / 1.3
    coeffs_x_, coeffs_y_ = calc_splines(path=path_coords, psi_s=psi_s_, psi_e=psi_e_)[
        0:2
    ]

    path_interp = interp_splines(
        coeffs_x=coeffs_x_,
        coeffs_y=coeffs_y_,
        incl_last_point=True,
        stepsize_approx=0.5,
    )[0]

    plt.plot(path_interp[:, 0], path_interp[:, 1])
    plt.axis("equal")
    plt.show()

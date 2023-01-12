import os
from pathlib import Path
import numpy as np
import pysocialforce as psf
from datetime import datetime as dt

from pysocialforce.utils.config import DefaultConfig


if __name__ == "__main__":
    # initial states, each entry is the position, velocity and goal of a pedestrian in the form of (px, py, vx, vy, gx, gy)
    #  x,  y - position
    # vx, vy - speed vector
    # gx, gy - goal position (used for DesiredForce - disabled, and stopping fully in scene.step() - commented out)
    initial_state = np.array(
        [
        #   [   x     y    vx    vy   gx   gy]
            # [ 0.0,  0.0,  2.0,  0.0, 0.0, 0.0],
            # [ 0.0,  4.0,  1.0,  0.0, 0.0, 0.0],
            # [ 0.0,  0.5,  2.0,  0.0, 0.0, 0.0],
            # [ 0.0, -0.5,  2.0,  0.0, 0.0, 0.0],
            # [ 0.0,  1.0,  2.0,  0.0, 0.0, 0.0],
            # [ 0.0, -1.0,  2.0,  0.0, 0.0, 0.0],

            [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            # [ 0.0, 3.0, 0.0, 1.0, 0.0, 0.0],
            [ -40.0, 3.0, -3.0, -1.0, 0.0, 0.0],

          # [ 0.5,  0.0,  0.5,  0.0, 0.0, 0.0],
          # [ 0.0,  0.5,  0.0,  0.5, 0.0, 0.0],
          # [-0.5,  0.0, -0.5,  0.0, 0.0, 0.0],
          # [ 0.0, -0.5,  0.0, -0.5, 0.0, 0.0],
          # [ 0.5,  0.5,  0.5,  0.5, 0.0, 0.0],
          # [-0.5, -0.5, -0.5, -0.5, 0.0, 0.0],
          # [ 0.5, -0.5,  0.5, -0.5, 0.0, 0.0],
          # [-0.5,  0.5, -0.5,  0.5, 0.0, 0.0],
        ]
    )
    # social groups informoation is represented as lists of indices of the state array
    # groups = [[1, 0], [2]]
    ### groups = [[0], [1], [2], [3], [4], [5], [6], [7]]
    groups = [[0]]
    # list of linear obstacles given in the form of (x_min, x_max, y_min, y_max)
    # obs = [[0, 8, 8, 0]]
    # obs = [[10, 10, -2, 2]]
    # obs = [[3, 4, 0, 0]]
    # obs = [[-1, -1, -1, 11], [3, 3, -1, 11]]
    # obs = [[-10, -10, 1, 4]]
    obs = None
    obs = [[-400,  20, 35, 35], [-400,  20, -35, -35], 
           [-40, -41, -15, -15], [-40, -41, -16, -16], [-40, -40, -15, -16], [-41, -41, -15, -16], [-40, -41, -15, -16], [-40, -41, -16, -15],
           [-80, -81, -5, -5], [-80, -81, -6, -6], [-80, -80, -5, -6], [-81, -81, -5, -6], [-80, -81, -5, -6], [-80, -81, -6, -5],
           [-120, -121, -15, -15], [-120, -121, -16, -16], [-120, -120, -15, -16], [-121, -121, -15, -16], [-120, -121, -15, -16], [-120, -121, -16, -15], ]
    # initiate the simulator,
    s = psf.Simulator(
        initial_state,
        groups=groups,
        obstacles=obs,
        config_file=Path(__file__).resolve().parent.joinpath("example.toml"),
    )
    # update 80 steps
    s.step(200)



    timestamp = dt.now().strftime("%H%M%S")

    f = open("images/exmaple" + timestamp + ".txt", "w")
    f.write(DefaultConfig.CONFIG)
    f.close()

    with psf.plot.SceneVisualizer(s, "images/exmaple" + timestamp) as sv:
        sv.animate()
        # sv.plot()
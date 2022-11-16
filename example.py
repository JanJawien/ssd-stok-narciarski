from pathlib import Path
import numpy as np
import pysocialforce as psf
from datetime import datetime as dt


if __name__ == "__main__":
    # initial states, each entry is the position, velocity and goal of a pedestrian in the form of (px, py, vx, vy, gx, gy)
    #  x,  y - position
    # vx, vy - speed vector
    # gx, gy - goal position (used for DesiredForce - disabled, and stopping fully in scene.step() - commented out)
    initial_state = np.array(
        [
        #   [   x     y    vx    vy   gx   gy]
            [ 0.5,  0.0,  0.5,  0.0, 0.0, 0.0],
            [ 0.0,  0.5,  0.0,  0.5, 0.0, 0.0],
            [-0.5,  0.0, -0.5,  0.0, 0.0, 0.0],
            [ 0.0, -0.5,  0.0, -0.5, 0.0, 0.0],
            [ 0.5,  0.5,  0.5,  0.5, 0.0, 0.0],
            [-0.5, -0.5, -0.5, -0.5, 0.0, 0.0],
            [ 0.5, -0.5,  0.5, -0.5, 0.0, 0.0],
            [-0.5,  0.5, -0.5,  0.5, 0.0, 0.0],
            # [1.0, 0.0, 0.0, 0.5, 2.0, 10.0],
            # [2.0, 0.0, 0.0, 0.5, 3.0, 10.0],
            # [3.0, 0.0, 0.0, 0.5, 4.0, 10.0],
        ]
    )
    # social groups informoation is represented as lists of indices of the state array
    # groups = [[1, 0], [2]]
    groups = [[0], [1], [2], [3], [4], [5], [6], [7]]
    # list of linear obstacles given in the form of (x_min, x_max, y_min, y_max)
    # obs = [[-1, -1, -1, 11], [3, 3, -1, 11]]
    # obs = [[1, 2, 7, 8]]
    obs = None
    # initiate the simulator,
    s = psf.Simulator(
        initial_state,
        groups=groups,
        obstacles=obs,
        config_file=Path(__file__).resolve().parent.joinpath("example.toml"),
    )
    # update 80 steps
    s.step(50)

    with psf.plot.SceneVisualizer(s, "images/exmaple" + dt.now().strftime("%H%M%S")) as sv:
        sv.animate()
        # sv.plot()
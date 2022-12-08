"""Utility functions to process state."""
from typing import Tuple

import numpy as np
from numba import njit
import matplotlib.pyplot as plt


# @jit
# def normalize(array_in):
#     """nx2 or mxnx2"""
#     if len(array_in.shape) == 2:
#         vec, fac = normalize_array(array_in)
#         return vec, fac
#     factors = []
#     vectors = []
#     for m in array_in:
#         vec, fac = normalize_array(m)
#         vectors.append(vec)
#         factors.append(fac)

#     return np.array(vectors), np.array(factors)


@njit
def vector_angles(vecs: np.ndarray) -> np.ndarray:
    """Calculate angles for an array of vectors
    :param vecs: nx2 ndarray
    :return: nx1 ndarray
    """
    ang = np.arctan2(vecs[:, 1], vecs[:, 0])  # atan2(y, x)
    return ang


@njit
def left_normal(vecs: np.ndarray) -> np.ndarray:
    vecs = np.fliplr(vecs) * np.array([-1.0, 1.0])
    return vecs


@njit
def right_normal(vecs: np.ndarray) -> np.ndarray:
    vecs = np.fliplr(vecs) * np.array([1.0, -1.0])
    return vecs


@njit
def normalize(vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize nx2 array along the second axis
    input: [n,2] ndarray
    output: (normalized vectors, norm factors)
    """
    norm_factors = []
    for line in vecs:
        norm_factors.append(np.linalg.norm(line))
    norm_factors = np.array(norm_factors)
    normalized = vecs / np.expand_dims(norm_factors, -1)
    # get rid of nans
    for i in range(norm_factors.shape[0]):
        if norm_factors[i] == 0:
            normalized[i] = np.zeros(vecs.shape[1])
    return normalized, norm_factors


@njit
def desired_directions(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given the current state and destination, compute desired direction."""
    destination_vectors = state[:, 4:6] - state[:, 0:2]
    directions, dist = normalize(destination_vectors)
    return directions, dist


@njit
def vec_diff(vecs: np.ndarray) -> np.ndarray:
    """r_ab
    r_ab := r_a − r_b.
    """
    diff = np.expand_dims(vecs, 1) - np.expand_dims(vecs, 0)
    return diff


def each_diff(vecs: np.ndarray, keepdims=False) -> np.ndarray:
    """
    :param vecs: nx2 array
    :return: diff with diagonal elements removed
    """
    diff = vec_diff(vecs)
    # diff = diff[np.any(diff, axis=-1), :]  # get rid of zero vectors
    diff = diff[
        ~np.eye(diff.shape[0], dtype=bool), :
    ]  # get rif of diagonal elements in the diff matrix
    if keepdims:
        diff = diff.reshape(vecs.shape[0], -1, vecs.shape[1])

    return diff


@njit
def speeds(state: np.ndarray) -> np.ndarray:
    """Return the speeds corresponding to a given state."""
    #     return np.linalg.norm(state[:, 2:4], axis=-1)
    speed_vecs = state[:, 2:4]
    speeds_array = np.array([np.linalg.norm(s) for s in speed_vecs])
    return speeds_array


@njit
def center_of_mass(vecs: np.ndarray) -> np.ndarray:
    """Center-of-mass of a given group"""
    return np.sum(vecs, axis=0) / vecs.shape[0]


@njit
def minmax(vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_min = np.min(vecs[:, 0])
    y_min = np.min(vecs[:, 1])
    x_max = np.max(vecs[:, 0])
    y_max = np.max(vecs[:, 1])
    return (x_min, y_min, x_max, y_max)


def foc_to_per(a: float, b: float, axis: np.ndarray, direction: np.ndarray):
    # https://math.etsu.edu/multicalc/prealpha/chap3/chap3-2/part4.htm
    axis = axis.copy()
    direction = direction.copy()
    axis /= np.linalg.norm(axis)
    direction /= np.linalg.norm(direction)
    cos = np.dot(axis, direction)
    c = np.sqrt(a**2 - b**2)
    return (b**2 / a) / (1 - c/a * cos)


def ellipse_factor(speed_vec: np.ndarray):
    """Function responsible for stretching circle into ellipse based on agent speed, and other related calculations.
    For speed vector of norm 0 it should always return 1"""
    return np.linalg.norm(speed_vec)/10 + 1 


def ellipse_obstacle_force(speed: np.ndarray, obstacle: np.ndarray, radius: float) -> np.ndarray:
    b = radius
    a = b * ellipse_factor(speed)
    r = foc_to_per(a, b, speed, obstacle)
    d = np.linalg.norm(obstacle)

    if d > r:
        return [0, 0]

    c = np.sqrt(a**2 - b**2)
    min = a - c
    max = 2*a - min
    norm_value = 1 - ((r-min) / (max-min)) # <0, 1>, 0: edge, 1: focal point

    return obstacle / d * -np.exp(-4 * norm_value)


def ped_ellipse_center(ped: np.ndarray, radius: float):
    speed = ped[2:4].copy()
    b = radius
    a = b * ellipse_factor(speed)
    c = np.sqrt(a**2 - b**2)

    speed /= np.linalg.norm(speed) * c
    return ped[0:2] + speed


def ellipse_social_force(ped: np.ndarray, other_ped: np.ndarray, radius: float):
    # op_future_pos = ped_ellipse_center(other_ped, radius)
    ped_pos_delta = other_ped[0:2] - ped[0:2]

    force = ellipse_obstacle_force(ped[2:4].copy(), ped_pos_delta, radius)
    force = np.array(force) * np.linalg.norm(ped[2:4])
    return force


def slowingValue(speed: float):
	return 0 if speed<1.0 else (1 if speed>1.5 else (1 - np.sqrt(1-((2*speed - 2)**2))))
	# return 0 if speed<1.0 else (1 if speed>1.5 else (2*speed - 2))


def speedingValue(speed: float):
	return 1 if speed<0.5 else (0 if speed>1.0 else (1 - np.sqrt(1-((2*speed - 2)**2))))
	# return 1 if speed<0.5 else (0 if speed>1.0 else (-2*speed + 2))


def applyDesiredSpeedForce(direction: np.ndarray, turns_right: bool, is_too_fast: bool, desired_speed: float):
	speed_val = np.linalg.norm(direction)
	direction /= speed_val
	force_val = 0
	if is_too_fast:
		force_val = slowingValue(speed_val/desired_speed)
	else:
		force_val = speedingValue(speed_val/desired_speed)

	if force_val == 0:
		return [0, 0]

	sin = np.sin(np.arcsin(force_val/2/speed_val)*2)
	if turns_right:
		sin *= -1
	return rotateVector(direction*force_val, sin)


def rotateVector(vector: np.ndarray, sin: float):
	if sin == 0:
		return vector
	
	cos = np.sqrt(1 - (sin**2))
	rot = [[cos, -sin], [sin, cos]]
	return np.dot(rot, vector)
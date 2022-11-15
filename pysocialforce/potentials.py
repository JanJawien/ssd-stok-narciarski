"""Interaction potentials."""

import numpy as np

from pysocialforce.utils import stateutils


class PedPedPotential(object):
    """Ped-ped interaction potential.

    v0 is in m^2 / s^2.
    sigma is in m.
    """


class PedSpacePotential(object):
    """Pedestrian-obstacles interaction potential.

    obstacles is a list of numpy arrays containing points of boundaries.

    u0 is in m^2 / s^2.
    r is in m
    """

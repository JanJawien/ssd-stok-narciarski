# coding=utf-8

"""Synthetic pedestrian behavior with social groups simulation according to the Extended Social Force model.

See Helbing and Molnár 1998 and Moussaïd et al. 2010
"""
from pysocialforce.utils import DefaultConfig
from pysocialforce.scene import PedState, EnvState
from pysocialforce import forces

import numpy as np


class Simulator:
    """Simulate social force model.

    ...

    Attributes
    ----------
    state : np.ndarray [n, 6] or [n, 7]
       Each entry represents a pedestrian state, (x, y, v_x, v_y, d_x, d_y, [tau])
    obstacles : np.ndarray
        Environmental obstacles
    groups : List of Lists
        Group members are denoted by their indices in the state
    config : Dict
        Loaded from a toml config file
    max_speeds : np.ndarray
        Maximum speed of pedestrians
    forces : List
        Forces to factor in during navigation

    Methods
    ---------
    capped_velocity(desired_velcity)
        Scale down a desired velocity to its capped speed
    step()
        Make one step
    """

    def __init__(self, state, groups=None, obstacles=None, config_file=None):
        self.config = DefaultConfig()
#         if config_file:
#             self.config.load_config(config_file)
        # TODO: load obstacles from config
        self.scene_config = self.config.sub_config("scene")
        # initiate obstacles
        self.env = EnvState(obstacles, self.config("resolution", 10.0))

        # initiate agents
        self.peds = PedState(state, groups, self.config)

        # construct forces
        self.forces = self.make_forces(self.config)

        # construct friction forces
        self.friction_forces = self.make_friction_forces(self.config)

    def make_friction_forces(self, force_configs):
        """Construct friction forces; DO NOT CHANGE THE ORDER OF THESE FORCES"""
        force_list = [
			# forces.AirResistanceForce(),        # TODO opory powietrza  
			forces.StaticFrictionForce(),       # tarcie w bezruchu 
			forces.KinematicFrictionForce()     # tarcie w jeździe
        ]

        # initiate forces
        for force in force_list:
            force.init(self, force_configs)

        return force_list

    def make_forces(self, force_configs):
        """Construct forces; DO NOT CHANGE THE ORDER OF THESE FORCES"""
        force_list = [
            # forces.DesiredForce(),        # oryginalny, jazda do celu
            # forces.SocialForce(),         # originalny
            # forces.ObstacleForce(),       # oryginalny
			forces.ParallelDownhillForce(),     # zjazd wzdłuż kierunku nart
            forces.EllipticalObstacleForce(),   # zamieniony na elipsę
            forces.EllipticalSocialForce(),     # zamieniony na elipsę
			forces.TowardsDownhillForce(),      # skręcanie pod względem prędkości
        ]
        group_forces = [
            # forces.GroupCoherenceForceAlt(),
            # forces.GroupRepulsiveForce(),
            # forces.GroupGazeForceAlt(),
        ]
        if self.scene_config("enable_group"):
            force_list += group_forces

        # initiate forces
        for force in force_list:
            force.init(self, force_configs)

        return force_list

    def compute_forces(self):
        """compute forces"""
        # calculate all primary forces acting on each skier
        primary_forces = np.array([f._get_force() for f in self.forces])
        primary_sum = np.sum(primary_forces, 0)
        # print("Forces:")
        # print(primary_forces)
        # print("Sum:")
        # print(primary_sum)

        # calculate friction forces based on primary forces
        friction_forces = np.array([f._get_force(primary_forces) for f in self.friction_forces])
        friction_sum = np.sum(friction_forces, 0)
        # print("Friction forces:")
        # print(friction_forces)
        # print("Friction sum:")
        # print(friction_sum)

        output = (primary_sum + friction_sum)
        print(primary_sum)
        print(friction_sum)
        print(output)
        return output

    def get_states(self):
        """Expose whole state"""
        return self.peds.get_states()

    def get_length(self):
        """Get simulation length"""
        return len(self.get_states()[0])

    def get_obstacles(self):
        return self.env.obstacles

    def step_once(self):
        """step once"""
        self.peds.step(self.compute_forces())

    def step(self, n=1):
        """Step n time"""
        for _ in range(n):
            self.step_once()
        return self

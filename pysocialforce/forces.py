"""Calculate forces for individuals and groups"""
import math
import re
from abc import ABC, abstractmethod

import numpy as np
from pysocialforce.utils import Config, stateutils, logger, height


def camel_to_snake(camel_case_string):
    """Convert CamelCase to snake_case"""

    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_case_string).lower()


class Force(ABC):
    """Force base class"""

    def __init__(self):
        super().__init__()
        self.scene = None
        self.peds = None
        self.factor = 1.0
        self.config = Config()

    def init(self, scene, config):
        """Load config and scene"""
        # load the sub field corresponding to the force name from global confgi file
        self.config = config.sub_config(camel_to_snake(type(self).__name__))
        if self.config:
            self.factor = self.config("factor", 1.0)

        self.scene = scene
        self.peds = self.scene.peds

        # print(type(self))
        # print(type(self.config))
        # print(self.factor)
        # print()


class DesiredForce(Force):
    """Calculates the force between this agent and the next assigned waypoint.
    If the waypoint has been reached, the next waypoint in the list will be
    selected.
    :return: the calculated force
    """

    def _get_force(self):
        relexation_time = self.config("relaxation_time", 0.5)
        goal_threshold = self.config("goal_threshold", 0.1)
        pos = self.peds.pos()
        vel = self.peds.vel()
        goal = self.peds.goal()
        direction, dist = stateutils.normalize(goal - pos)
        force = np.zeros((self.peds.size(), 2))
        force[dist > goal_threshold] = (
            direction * self.peds.max_speeds.reshape((-1, 1)) - vel.reshape((-1, 2))
        )[dist > goal_threshold, :]
        force[dist <= goal_threshold] = -1.0 * vel[dist <= goal_threshold]
        force /= relexation_time
        return force * self.factor


class SocialForce(Force):
    """Calculates the social force between this agent and all the other agents
    belonging to the same scene.
    It iterates over all agents inside the scene, has therefore the complexity
    O(N^2). A better
    agent storing structure in Tscene would fix this. But for small (less than
    10000 agents) scenarios, this is just
    fine.
    :return:  nx2 ndarray the calculated force
    """

    def _get_force(self):
        lambda_importance = self.config("lambda_importance", 2.0)
        gamma = self.config("gamma", 0.35)
        n = self.config("n", 2)
        n_prime = self.config("n_prime", 3)

        pos_diff = stateutils.each_diff(self.peds.pos())  # n*(n-1)x2 other - self
        diff_direction, diff_length = stateutils.normalize(pos_diff)
        vel_diff = -1.0 * stateutils.each_diff(self.peds.vel())  # n*(n-1)x2 self - other

        # compute interaction direction t_ij
        interaction_vec = lambda_importance * vel_diff + diff_direction
        interaction_direction, interaction_length = stateutils.normalize(interaction_vec)

        # compute angle theta (between interaction and position difference vector)
        theta = stateutils.vector_angles(interaction_direction) - stateutils.vector_angles(
            diff_direction
        )
        # compute model parameter B = gamma * ||D||
        B = gamma * interaction_length

        force_velocity_amount = np.exp(-1.0 * diff_length / B - np.square(n_prime * B * theta))
        force_angle_amount = -np.sign(theta) * np.exp(
            -1.0 * diff_length / B - np.square(n * B * theta)
        )
        force_velocity = force_velocity_amount.reshape(-1, 1) * interaction_direction
        force_angle = force_angle_amount.reshape(-1, 1) * stateutils.left_normal(
            interaction_direction
        )

        force = force_velocity + force_angle  # n*(n-1) x 2
        force = np.sum(force.reshape((self.peds.size(), -1, 2)), axis=1)
        return force * self.factor

class ObstacleForce(Force):
    """Calculates the force between this agent and the nearest obstacle in this
    scene.
    :return:  the calculated force
    """

    def _get_force(self):
        sigma = self.config("sigma", 0.2)
        threshold = self.config("threshold", 0.2) + self.peds.agent_radius
        force = np.zeros((self.peds.size(), 2))
        if len(self.scene.get_obstacles()) == 0:
            return force
        obstacles = np.vstack(self.scene.get_obstacles())
        pos = self.peds.pos()

        for i, p in enumerate(pos):
            diff = p - obstacles
            directions, dist = stateutils.normalize(diff)
            dist = dist - self.peds.agent_radius
            if np.all(dist >= threshold):
                continue
            dist_mask = dist < threshold
            directions[dist_mask] *= np.exp(-dist[dist_mask].reshape(-1, 1) / sigma)
            force[i] = np.sum(directions[dist_mask], axis=0)
            # print()

        return force * self.factor



class EllipticalObstacleForce(Force):
    """Calculates the force between this agent and all obstacle points within their social ellipse.
    :return:  the calculated force
    """

    def _get_force(self):
        threshold = self.config("threshold", 0.2)
        force = np.zeros((self.peds.size(), 2))
        if len(self.scene.get_obstacles()) == 0:
            return force
        obstacles = np.vstack(self.scene.get_obstacles()) 
        peds = self.scene.peds.state

        for i, ped in enumerate(peds):
            ped_to_obs = obstacles - ped[:2]
            speed = ped[2:4].copy()
            forces = np.array([stateutils.ellipse_obstacle_force(speed, v, threshold) for v in ped_to_obs])
            force[i] = np.sum(forces, axis=0)
            # print()

        force *= self.factor
        return force



class EllipticalSocialForce(Force):
    """Calculates the force between agents within their social ellipses.
    :return:  the calculated force
    """

    def _get_force(self):
        threshold = self.config("threshold", 0.2)
        force = np.zeros((self.peds.size(), 2))
        peds = self.scene.peds.state

        for i, ped in enumerate(peds):
            forces = []
            for oi, oped in enumerate(peds):
                if oi == i:
                    continue
                forces.append(stateutils.ellipse_social_force(ped, oped, threshold))
            force[i] = np.sum(forces, axis=0)

        force *= self.factor
        return force


class TowardsDownhillForce(Force):
    """Applies force turning a skier towards the downhill gradient if their speed is lower than desired
    """

    def _get_force(self):
        dead_angle = self.config("dead_angle", 0.15)
        max_angle = self.config("max_angle", 0.35)
        desired_speed = self.config("desired_speed", 5)
        factor_offset = self.config("factor_offset", 0.0)
        turn_overshoot = self.config("turn_overshoot", 0.01)
        towards_downhill_turn_bias = self.config("towards_downhill_turn_bias", 0.3)
        force = np.zeros((self.peds.size(), 2))
        peds = self.scene.peds.state

        for i, ped in enumerate(peds):
            speed = ped[2:4].copy()                                 # speed vector
            left = [-speed[1], speed[0]]                            # leftward vector
            right = [speed[1], -speed[0]]                           # rightward vector
            downhill = height.calculate_grad(ped[0], ped[1])        # downhill direction vector
            cos_left = stateutils.vectorCos(left, downhill)         # cos between leftward and downhill
            cos_right = stateutils.vectorCos(right, downhill)       # cos between rightward and downhill
            slowing_down = np.linalg.norm(speed) > desired_speed    # is going too fast?

            # angle between current direction and desired direction (downhill if wants to got faster, uphill if wants to go slower)
            desired_direction_angle = np.arccos(stateutils.vectorCos((speed*-1 if slowing_down else speed), downhill)) 
            # if angle within dead angle (downhill direction margin), dont turn
            ped_dead_angle = dead_angle
            if not slowing_down:
                ped_dead_angle *= towards_downhill_turn_bias
            if desired_direction_angle < ped_dead_angle:
                force[i] = [0, 0]
                continue
            # perform weaker turn when closing to downhill axis to avoid oscillating around it
            ped_max_angle = max_angle
            if not slowing_down:
                ped_max_angle *= towards_downhill_turn_bias
            if desired_direction_angle < ped_max_angle:
                ped_max_angle = desired_direction_angle + turn_overshoot

            # get direction of desired turn
            direction = left
            turns_right = ((slowing_down and cos_left > cos_right) or
                           (not slowing_down and cos_right > cos_left))
            if turns_right:
                direction = right
            
            force[i] = stateutils.applyDesiredSpeedForce(direction, turns_right, slowing_down, desired_speed, self.factor, ped_max_angle, factor_offset)
        return force


class ParallelDownhillForce(Force):
    '''
    Calculates the gravity force acting on agent.
    :return:  the calculated force
    '''
    def _get_force(self):
        m = self.config("m")
        g = self.config("g")
        force = np.zeros((self.peds.size(), 2))
        peds = self.scene.peds.state

        # f = open("out.txt", "a")

        for i, ped in enumerate(peds):
            # f.write("v: " + str(ped[2:4]) + "\n")
            x = ped[0]
            y = ped[1]
            vx = ped[2]
            vy = ped[3]
            sin_alpha = height.sin_alpha(x, y)
            # sin_beta = math.sqrt(1 - height.cos_beta(x, y, vx, vy) ** 2)
            sin_beta = height.sin_beta(x, y, vx, vy)

            f_val = m * g * sin_alpha * sin_beta 
            v_norm = [vx, vy] / np.linalg.norm([vx, vy])
            force[i] = v_norm*f_val
            # print()

        # f.write("\n")
        # f.close()

        return force * self.factor


class AirResistanceForce(Force):
    def _get_force(self, primary_forces):
        Sd = self.config("Sd")
        force = np.ones((self.peds.size(), 2))
        peds = self.scene.peds.state

        for i, ped in enumerate(peds):
            speed = ped[2:4].copy()
            speed_val = np.linalg.norm(speed)
            direction = speed / speed_val

            force[i] = direction * speed_val**2 * Sd * -1/2

        return force * self.factor


class KinematicFrictionForce(Force):
    def _get_force(self, primary_forces):
        m = self.config("m")
        g = self.config("g")
        friction_coef = self.config("friction_coef")
        min_speed = self.config("min_speed", 0.05)
        force = np.ones((self.peds.size(), 2))
        peds = self.scene.peds.state

        for i, ped in enumerate(peds):
            if np.linalg.norm(ped[2:4]) < min_speed:
                force[i] = [0, 0]
                continue

            # print("Kinematic | " + str(i) + ":")
            # ped_forces = primary_forces[1:, i, :]
            # print(ped_forces)

            sin_alpha = height.sin_alpha(ped[0], ped[1])
            cos_alpha = np.sqrt(1 - (sin_alpha**2) )
            cos_beta = height.cos_beta(ped[0], ped[1], ped[2], ped[3])
            
            Fc = np.linalg.norm( np.sum(primary_forces[1:, i, :]) ) # wartość wszystich sił poza ParallelDownhillForce (czyli tych, które powodują skręt)
            Flat = m * g * sin_alpha * cos_beta
            Ftl = Fc + Flat
            Fn = m * g * cos_alpha
            Feff = np.sqrt(Ftl**2 + Fn**2)

            direction = (ped[2:4].copy()) / np.linalg.norm(ped[2:4])

            force[i] = direction * ( friction_coef * (Fn + Feff) ) * -1

        return force * self.factor


class StaticFrictionForce(Force):
    def _get_force(self, primary_forces):
        m = self.config("m")
        g = self.config("g")
        static_friction_coef = self.config("static_friction_coef")
        max_speed = self.config("max_speed", 0.05)
        force = np.ones((self.peds.size(), 2))
        peds = self.scene.peds.state

        for i, ped in enumerate(peds):
            if np.linalg.norm(ped[2:4]) > max_speed:
                force[i] = [0, 0]
                continue

            # print("Kinematic | " + str(i) + ":")
            # ped_forces = primary_forces[:, i, :]
            # print(ped_forces)

            sin_alpha = height.sin_alpha(ped[0], ped[1])
            cos_alpha = np.sqrt(1 - (sin_alpha**2) )
            
            Fnet = np.sum(primary_forces[:, i, :], 0) # wartość wszystich sił działających na narciarza
            Fn = m * g * cos_alpha

            value = np.linalg.norm(Fnet) if np.linalg.norm(Fnet) < Fn*static_friction_coef else Fn*static_friction_coef

            direction = (Fnet.copy()) / np.linalg.norm(Fnet)

            force[i] = direction * value * -1

        return force * self.factor








class GroupCoherenceForceAlt(Force):
    """ Alternative group coherence force as specified in pedsim_ros"""

    def _get_force(self):
        forces = np.zeros((self.peds.size(), 2))
        if self.peds.has_group():
            for group in self.peds.groups:
                threshold = (len(group) - 1) / 2
                member_pos = self.peds.pos()[group, :]
                com = stateutils.center_of_mass(member_pos)
                force_vec = com - member_pos
                norms = stateutils.speeds(force_vec)
                softened_factor = (np.tanh(norms - threshold) + 1) / 2
                forces[group, :] += (force_vec.T * softened_factor).T
        return forces * self.factor


class GroupRepulsiveForce(Force):
    """Group repulsive force"""

    def _get_force(self):
        threshold = self.config("threshold", 0.5)
        forces = np.zeros((self.peds.size(), 2))
        if self.peds.has_group():
            for group in self.peds.groups:
                size = len(group)
                member_pos = self.peds.pos()[group, :]
                diff = stateutils.each_diff(member_pos)  # others - self
                _, norms = stateutils.normalize(diff)
                diff[norms > threshold, :] = 0
                # forces[group, :] += np.sum(diff, axis=0)
                forces[group, :] += np.sum(diff.reshape((size, -1, 2)), axis=1)

        return forces * self.factor


class GroupGazeForceAlt(Force):
    """Group gaze force"""

    def _get_force(self):
        forces = np.zeros((self.peds.size(), 2))
        directions, dist = stateutils.desired_directions(self.peds.state)
        if self.peds.has_group():
            for group in self.peds.groups:
                group_size = len(group)
                # 1-agent groups don't need to compute this
                if group_size <= 1:
                    continue
                member_pos = self.peds.pos()[group, :]
                member_directions = directions[group, :]
                member_dist = dist[group]
                # use center of mass without the current agent
                relative_com = np.array(
                    [
                        stateutils.center_of_mass(member_pos[np.arange(group_size) != i, :2])
                        - member_pos[i, :]
                        for i in range(group_size)
                    ]
                )

                com_directions, com_dist = stateutils.normalize(relative_com)
                # angle between walking direction and center of mass
                element_prod = np.array(
                    [np.dot(d, c) for d, c in zip(member_directions, com_directions)]
                )
                force = (
                    com_dist.reshape(-1, 1)
                    * element_prod.reshape(-1, 1)
                    / member_dist.reshape(-1, 1)
                    * member_directions
                )
                forces[group, :] += force

        return forces * self.factor

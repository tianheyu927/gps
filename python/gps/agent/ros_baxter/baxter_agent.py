""" This file defines an agent for the Baxter ROS environment. """
import copy
import time
import numpy as np

import rospy

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
# TODO: ask Michael about this
from gps.agent.config import BAXTER_AGENT_ROS
from gps.agent.ros_baxter.baxter_utils import GPSBaxterInterface
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, RGB_IMAGE_SIZE, \
        CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE, IMAGE_FEAT, \
        END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, \
        TRIAL_ARM, AUXILIARY_ARM
from gps.sample.sample import Sample
try:
    from gps.algorithm.policy.tf_policy import TfPolicy
except ImportError:  # user does not have tf installed.
    TfPolicy = None


class BaxterAgentROS(Agent):
    """
    Communication between the algorithms and ROS is done through
    this class for a Baxter robot.
    """
    def __init__(self, hyperparams, init_node=True):
        """
        Initialize agent.
        Args:
            hyperparams: Dictionary of hyperparameters.
            init_node: Whether or not to initialize a new ROS node.
        """
        config = copy.deepcopy(BAXTER_AGENT_ROS)
        config.update(hyperparams)
        Agent.__init__(self, config)
        if init_node:
            rospy.init_node('gps_agent_ros_node')

        self.baxter = GPSBaxterInterface()

        self._seq_id = 0  # Used for setting seq in ROS commands.

        conditions = self._hyperparams['conditions']

        self._setup_conditions()
        self._setup_world()
        # self.x0 = self._hyperparams['x0']

        r = rospy.Rate(1)
        r.sleep()

    def _setup_conditions(self):
        """
        Helper method for setting some hyperparameters that may vary by
        condition.
        """
        conds = self._hyperparams['conditions']
        for field in ('x0', 'x0var', 'reset_conditions'):
            self._hyperparams[field] = setup(self._hyperparams[field], conds)

    def _setup_world(self):
        """
        Helper method for handling setup.
        """
        self._joint_idx = list(range(7))
        self._vel_idx = [i + 7 for i in self._joint_idx]

        # Initialize x0
        self.x0 = []
        for i in range(self._hyperparams['conditions']):
            if END_EFFECTOR_POINTS in self.x_data_types:
                # TODO: this assumes END_EFFECTOR_VELOCITIES is also in datapoints right?
                eepts = np.array(self.baxter.get_ee_pose('left'))
                self.x0.append(
                    np.concatenate([self._hyperparams['x0'][i], eepts, np.zeros_like(eepts)])
                )
            elif END_EFFECTOR_POINTS_NO_TARGET in self.x_data_types:
                eepts = np.array(self.baxter.get_ee_pose('left'))
                eepts_notgt = np.delete(eepts, self._hyperparams['target_idx'])
                self.x0.append(
                    np.concatenate([self._hyperparams['x0'][i], eepts_notgt, np.zeros_like(eepts_notgt)])
                )
            else:
                self.x0.append(self._hyperparams['x0'][i])

    def get_data(self, arm='left'):
        """
        Request for the most recent value for data/sensor readings.
        Returns entire sample report (all available data) in sample.
        Args:
            arm: TRIAL_ARM or AUXILIARY_ARM.
        """
        sample = Sample(self)
        sample.set(JOINT_ANGLES, np.array(self.baxter.get_joint_angles('left')), t=0)
        sample.set(JOINT_VELOCITIES, np.array(self.baxter.get_joint_velocities('left')), t=0)
        eepts = np.array(self.baxter.get_ee_pose('left'))
        sample.set(END_EFFECTOR_POINTS, eepts, t=0)
        eevel = np.array(self.baxter.get_ee_vel('left'))
        sample.set(END_EFFECTOR_POINT_VELOCITIES, np.zeros_like(eepts), t=0)

        if (END_EFFECTOR_POINTS_NO_TARGET in self._hyperparams['obs_include']):
            sample.set(END_EFFECTOR_POINTS_NO_TARGET, np.delete(eepts, self._hyperparams['target_idx']), t=0)
            sample.set(END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, np.delete(np.zeros_like(eepts), self._hyperparams['target_idx']), t=0)
        
        sample.set(END_EFFECTOR_POINT_JACOBIANS, np.array(self.baxter.get_ee_jac('left')), t=0)

        return sample

    def relax_arm(self, arm):
        """
        Relax one of the arms of the robot.
        Args:
            arm: Either TRIAL_ARM or AUXILIARY_ARM.
        """
        values = [0 for _ in range(7)]
        self.baxter.set_baxter_joint_efforts(arm, values)

    def reset_arm(self, arm, mode, data):
        """
        Issues a position command to an arm.
        Args:
            arm: Either TRIAL_ARM or AUXILIARY_ARM.
            mode: An integer code (defined in gps_pb2).
            data: An array of floats.
        """
        self.baxter.set_joint_angles(arm, data)

    def reset(self, condition):
        """
        Reset the agent for a particular experiment condition.
        Args:
            condition: An index into hyperparams['reset_conditions'].
        """
        condition_data = self._hyperparams['reset_conditions'][condition]
        self.reset_arm('left', condition_data['left']['mode'],
            condition_data['left']['data'])
        # self.reset_arm(TRIAL_ARM, condition_data[TRIAL_ARM]['mode'],
        #                condition_data[TRIAL_ARM]['data'])
        # self.reset_arm(AUXILIARY_ARM, condition_data[AUXILIARY_ARM]['mode'],
        #                condition_data[AUXILIARY_ARM]['data'])
        time.sleep(2.0)  # useful for the real robot, so it stops completely

    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        """
        Reset and execute a policy and collect a sample.
        Args:
            policy: A Policy object.
            condition: Which condition setup to run.
            verbose: Unused for this agent.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
        Returns:
            sample: A Sample object.
        """
        # Create new sample, populate first time step.
        self.reset(condition)
        new_sample = self._init_sample()
        X = self._hyperparams['x0'][condition]
        U = np.zeros([self.T, self.dU])
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        if np.any(self._hyperparams['x0var'][condition] > 0):
            x0n = self._hyperparams['x0var'] * \
                    np.random.randn(self._hyperparams['x0var'].shape)
            X += x0n

        # Take the sample.
        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            baxter_U = policy.act(X_t, obs_t, t, noise[t, :])
            U[t, :] = baxter_U

            if (t + 1) < self.T:
                for _ in range(self._hyperparams['substeps']):
                    self.baxter.set_joint_efforts('left', baxter_U)
                    # self.baxter.set_joint_velocities('right', right_map)
                self._set_sample(new_sample, t)

        new_sample.set(ACTION, U)
        if save:
            self._samples[condition].append(new_sample)
        return new_sample

    def _init_sample(self):
        """
        Construct a new sample and fill in the first time step.
        """
        sample = Sample(self)

        # Initialize sample
        sample.set(JOINT_ANGLES, np.array(self.baxter.get_joint_angles('left')), t=0)
        sample.set(JOINT_VELOCITIES, np.array(self.baxter.get_joint_velocities('left')), t=0)
        eepts = np.array(self.baxter.get_ee_pose('left'))
        sample.set(END_EFFECTOR_POINTS, eepts, t=0)
        sample.set(END_EFFECTOR_POINT_VELOCITIES, np.zeros_like(eepts), t=0)

        if (END_EFFECTOR_POINTS_NO_TARGET in self._hyperparams['obs_include']):
            sample.set(END_EFFECTOR_POINTS_NO_TARGET, np.delete(eepts, self._hyperparams['target_idx']), t=0)
            sample.set(END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, np.delete(np.zeros_like(eepts), self._hyperparams['target_idx']), t=0)
        
        sample.set(END_EFFECTOR_POINT_JACOBIANS, np.array(self.baxter.get_ee_jac('left')), t=0)

        return sample

    def _set_sample(self, sample, t):
        """
        Set the data for a sample for one time step.
        Args:
            sample: Sample object to set data for.
            t: Time step to set for sample.
        """
        sample.set(JOINT_ANGLES, np.array(self.baxter.get_joint_angles('left')), t=t+1)
        sample.set(JOINT_VELOCITIES, np.array(self.baxter.get_joint_velocities('left')), t=t+1)

        cur_eepts = np.array(self.baxter.get_ee_pose('left'))
        sample.set(END_EFFECTOR_POINTS, cur_eepts, t=t+1)
        prev_eepts = sample.get(END_EFFECTOR_POINTS, t=t)
        eept_vels = (cur_eepts - prev_eepts) / self._hyperparams['dt']
        sample.set(END_EFFECTOR_POINT_VELOCITIES, eept_vels, t=t+1)

        if (END_EFFECTOR_POINTS_NO_TARGET in self._hyperparams['obs_include']):
            sample.set(END_EFFECTOR_POINTS_NO_TARGET, np.delete(cur_eepts, self._hyperparams['target_idx']), t=t+1)
            sample.set(END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, np.delete(eept_vels, self._hyperparams['target_idx']), t=t+1)

        sample.set(END_EFFECTOR_POINT_JACOBIANS, np.array(self.baxter.get_ee_jac('left')), t=t+1)
        
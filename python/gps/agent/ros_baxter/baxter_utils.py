""" This file defines utilities for the ROS agents. """
from baxter_interface import limb

from baxter_pykdl import baxter_kinematics
baxter_joint_names = ['_s0','_s1','_e0','_e1','_w0','_w1','_w2']

class GPSBaxterInterface(object):
    def __init__(self):
        self.right_arm = limb.Limb('right')
        self.left_arm = limb.Limb('left')
        self.limbs = {'right': self.right_arm, 'left': self.left_arm}

    def set_joint_angles(self, limb, joint_angles):
        """
        limb: String 'left' or 'right'
        joint_angles: list of joint angles
        """
        assert limb in ['left', 'right']
        angle_map = self.map_values_to_joints(limb, joint_angles)
        self.limbs[limb].move_to_joint_positions(angle_map)

    def set_joint_velocities(self, limb, joint_velocities):
        """
        limb: String 'left' or 'right'
        joint_efforts: list of joint velocities
        """
        assert limb in ['left', 'right']
        vel_map = self.map_values_to_joints(limb, joint_velocities)
        self.limbs[limb].set_joint_velocities(vel_map)

    def set_joint_efforts(self, limb, joint_efforts):
        """
        limb: String 'left' or 'right'
        joint_efforts: list of joint efforts
        """
        assert limb in ['left', 'right']
        effort_map = self.map_values_to_joints(limb, joint_efforts)
        self.limbs[limb].set_joint_torques(effort_map)

    def get_joint_angles(self, limb):
        """
        limb: String 'left' or 'right'
        -----
        Returns list of joint angles
        """
        return self.limbs[limb].joint_angles().values()

    def get_joint_velocities(self, limb):
        """
        limb: String 'left' or 'right'
        -----
        Returns list of joint velocities
        """
        return self.limbs[limb].joint_velocities().values()

    def get_joint_efforts(self, limb):
        """
        limb: String 'left' or 'right'
        -----
        Returns list of joint efforts
        """
        return self.limbs[limb].joint_efforts().values()

    def get_ee_pose(self, limb):
        """
        limb: String 'left' or 'right'
        -----
        Returns the cartesian endpoint pose (x, y, z)
        """
        pose = self.limbs[limb].endpoint_pose()
        return pose['position']

    def get_ee_quaternion(self, limb):
        """
        limb: String 'left' or 'right'
        ------
        Returns the cartesian endpoint quaternion (x, y, z, w)
        """
        return self.limbs[limb].endpoint_pose()['orientation']

    def get_ee_vel(self, limb):
        """
        limb: String 'left' or 'right'
        -----
        Returns the endpoint linear velocity
        """
        vel = self.endpoint_velocity()
        return vel['linear']

    def get_ee_ang_vel(self, limb):
        """
        limb: String 'left' or 'right'
        -----
        Returns the endpoint angular velocity
        """
        vel = self.endpoint_velocity()
        return vel['angular']

    def get_ee_jac(self, limb='left'):
        """
        limb: String 'left' or 'right'
        -----
        Returns the endpoint jacobian
        """
        return baxter_kinematics(limb).jacobian()

    def map_values_to_joints(self, limb, values):
        """
        limb: String 'left' or 'right'
        values: List of numerical values to map to joints using above ordering
        -----
        Returns a dictionary with joint_name:value
        """
        assert len(values) == 7
        mapping = {}
        for i in range(len(baxter_joint_names)):
            mapping[limb+baxter_joint_names[i]] = values[i]

        return mapping

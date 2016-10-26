from __future__ import division

from datetime import datetime
import os.path
import numpy as np
import operator

from gps import __file__ as gps_filepath
from gps.agent.mjc import obstacle_pointmass
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.cost.cost_fk import CostFK
#from gps.algorithm.cost.cost_fk_blocktouch import CostFKBlock
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_sum import CostSum
#from gps.algorithm.cost.cost_gym import CostGym
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd
from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy_opt.tf_model_example import example_tf_network
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY, RAMP_QUADRATIC, evall1l2term
from gps.utility.data_logger import DataLogger
from gps.algorithm.policy_opt.tf_model_example import multi_modal_network, multi_modal_network_fp

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE, RGB_IMAGE_SIZE, ACTION, \
        END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, IMAGE_FEAT
from gps.gui.config import generate_experiment_info

IMAGE_WIDTH = 80
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

SENSOR_DIMS = {
    JOINT_ANGLES: 2,
    JOINT_VELOCITIES: 2,
    END_EFFECTOR_POINTS: 6,
    END_EFFECTOR_POINT_VELOCITIES: 6,
    END_EFFECTOR_POINTS_NO_TARGET: 3,
    END_EFFECTOR_POINT_VELOCITIES_NO_TARGET: 3,
    ACTION: 2,
    RGB_IMAGE: IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS,
    RGB_IMAGE_SIZE: 3,
    IMAGE_FEAT: 10,  # affected by num_filters set below.
}

PR2_GAINS = np.array([1.0, 1.0])

BASE_DIR = '/'.join(str.split(__file__, '/')[:-2])
EXP_DIR = '/'.join(str.split(__file__, '/')[:-1]) + '/'

CONDITIONS = 7
np.random.seed(14)
target_pos = np.array([1.3, 0.0, 0.])

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': CONDITIONS,
    'train_conditions': range(CONDITIONS),
    'test_conditions': range(CONDITIONS),
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    'models': [obstacle_pointmass(target_pos, wall_center=0.0, hole_height=0.3, control_limit=10),
               obstacle_pointmass(target_pos, wall_center=0.1, hole_height=0.3, control_limit=10),
               obstacle_pointmass(target_pos, wall_center=-0.1, hole_height=0.3, control_limit=10),
               obstacle_pointmass(target_pos, wall_center=0.2, hole_height=0.3, control_limit=10),
               obstacle_pointmass(target_pos, wall_center=-0.2, hole_height=0.3, control_limit=10),
               obstacle_pointmass(target_pos, wall_center=0.3, hole_height=0.3, control_limit=10),
               obstacle_pointmass(target_pos, wall_center=-0.3, hole_height=0.3, control_limit=10)],
    'x0': np.array([-1., 0., 0., 0.]),
    'dt': 0.05,
    'substeps': 1,
    'conditions': common['conditions'],
    'T': 200,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET,
                      END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, IMAGE_FEAT],  # TODO - may want to include fp velocities.
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET, RGB_IMAGE],
    'target_idx': np.array(list(range(3,6))),
    'meta_include': [RGB_IMAGE_SIZE],
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'sensor_dims': SENSOR_DIMS,
    'record_reward': True,
    'camera_pos': np.array([1., 0., 8., 0., 0., 0.]),
    'point_linear': True,
}


algorithm = {
    'type': AlgorithmMDGPS,
    'max_ent_traj': 0.001,
    'conditions': common['conditions'],
    'iterations': 13,
    'kl_step': 1.0, # TODO was 1.0
    'min_step_mult': 0.1, # TODO was 0.5, maybe try 0.1
    'max_step_mult': 3.0, # TODO was 3.0, maybe try 2.0
    'policy_sample_mode': 'replace',
    'sample_on_policy': True,
    'plot_dir': EXP_DIR,
    'step_rule': 'laplace',
    'target_end_effector': target_pos,
}


algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'network_params': {
        'num_filters': [15, 15, 5],
        'obs_include': agent['obs_include'],
        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS_NO_TARGET, END_EFFECTOR_POINT_VELOCITIES_NO_TARGET],
        'obs_image_data': [RGB_IMAGE],
        'image_width': IMAGE_WIDTH,
        'image_height': IMAGE_HEIGHT,
        'image_channels': IMAGE_CHANNELS,
        'sensor_dims': SENSOR_DIMS,
    },
    'network_model': multi_modal_network_fp,
    'fc_only_iterations': 5000,
    'init_iterations': 1000,
    'iterations': 1000,  # was 100
    'weights_file_prefix': EXP_DIR + 'policy',
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  100.0 / PR2_GAINS,
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 1.0,
    'dt': agent['dt'],
    'T': agent['T'],
}

state_cost = {
    'type': CostState,
    'l2': 10.,
    'l1': 0.,
    'alpha': 1e-5,
    'data_types' : {
        JOINT_ANGLES: {
            'wp': np.ones(SENSOR_DIMS[ACTION]),
            'target_state': target_pos[0:2]
        },
    },
}

action_cost = {
    'type': CostAction,
    'wu': np.array([1., 1.])*1e-2
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [state_cost, action_cost],
    'weights': [0.1, 0.1], # used 10,1 for T=3
}


algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 5,
        'min_samples_per_cluster': 40,
        'max_samples': 10,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}


algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 5,
}

NUM_SAMPLES = 5
config = {
    'iterations': algorithm['iterations'],
    'num_samples': NUM_SAMPLES,
    'verbose_trials': 0,  # only show the first image.
    'verbose_policy_trials': 0,
    'common': common,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'conditions': common['conditions'],
}

common['info'] = generate_experiment_info(config)

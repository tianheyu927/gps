""" Hyperparameters for MJC 2D navigation policy optimization. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.mjc import AgentMuJoCo, pointmass, weighted_pointmass
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy.lin_gauss_init import init_pd
from gps.algorithm.policy_opt.tf_model_example import example_tf_network
from gps.algorithm.policy.policy_prior import PolicyPrior
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION
from gps.gui.config import generate_experiment_info


SENSOR_DIMS = {
    JOINT_ANGLES: 2,
    JOINT_VELOCITIES: 2,
    END_EFFECTOR_POINTS: 3,
    END_EFFECTOR_POINT_VELOCITIES: 3,
    ACTION: 2,
}

EXP_DIR = os.path.dirname(__file__) + '/'
CONDITIONS = 10

# target_pos = np.array([1.3, 0.0, 0.])
np.random.seed(47)
target_pos = []
for _ in xrange(CONDITIONS):
    pos = np.random.rand(3)
    pos = pos - 0.5
    pos[2] = 0.0
    target_pos.append(pos)
# wall_1_center = np.array([0.5, -0.8, 0.])
# wall_2_center = np.array([0.5, 0.8, 0.])
# wall_height = 2.8


common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    # 'data_files_dir': EXP_DIR + 'data_files_oracle/',
    'data_files_dir': EXP_DIR + 'data_files_10/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'train_conditions': range(CONDITIONS),
    'test_conditions': range(CONDITIONS),
    'conditions': CONDITIONS,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    'models': [pointmass(target_position=pos) for pos in target_pos],
    'x0': np.array([0., 0., 0., 0.]),
    'dt': 0.05,
    'substeps': 1,
    'conditions': common['conditions'],
    'T': 100,
    'point_linear': True,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'smooth_noise': False,
    'camera_pos': np.array([1., 0., 6., 0., 0., 0.]),
}


# algorithm = {
#     'type': AlgorithmMDGPS,
#     'conditions': common['conditions'],
#     'sample_on_policy': True,
#     'iterations': 30,
#     'kl_step': 1.0,
#     'min_step_mult': 0.01, #0.1
#     'max_step_mult': 4.0,
#     'max_ent_traj': 1.0,
#     'step_rule': 'laplace',
#     'target_end_effector': target_pos,
# }

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'iterations': 15,
    'kl_step': 1.0,
    'min_step_mult': 0.1,
    'max_step_mult': 4.0,
    'max_ent_traj': 1.0,
    'step_rule': 'laplace',
    'target_end_effector': target_pos,
}


algorithm['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 10.0,
    'pos_gains': 10.0,
    'vel_gains_mult': 0.0,
    'dQ': SENSOR_DIMS[ACTION],
    'dt': agent['dt'],
    'T': agent['T'],
}

state_cost = [{
    'type': CostState,
    'l2': 10,
    'l1': 0.0,
    'alpha': 1e-4, #1e-4
    'data_types' : {
        JOINT_ANGLES: {
            'wp': np.ones(SENSOR_DIMS[ACTION]),
            'target_state': target_pos[i][0:2],
        },
        # JOINT_VELOCITIES: {
        #     'wp': 0*np.ones(SENSOR_DIMS[ACTION]),
        # },
    },
} for i in xrange(CONDITIONS)]

action_cost = {
    'type': CostAction,
    'wu': np.array([1.0, 1.0])*0.5, #*0.5, 
}

algorithm['cost'] = [{
    'type': CostSum,
    'costs': [state_cost[i], action_cost],
    'weights': [1.0, 1.0], # used 10,1 for T=3
} for i in xrange(CONDITIONS)]

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 5,
        'min_samples_per_cluster': 20,
        'max_samples': 20,
    }
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}


algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'network_params': {
        'obs_include': agent['obs_include'],
        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
        'sensor_dims': SENSOR_DIMS,
        'n_layers': 3,
        'dim_hidden': 20,
    },
    'network_model': example_tf_network,
    'lr': 1e-3,
    'use_gpu': 1,
    'uses_vision': False,
    # 'fc_only_iterations': 5000,
    # 'init_iterations': 1000,
    'iterations': 1000,
    'weights_file_prefix': common['data_files_dir'] + 'policy',
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 20,
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 10,
    'verbose_trials': 1,
    'verbose_policy_trials': 1,
    'common': common,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)

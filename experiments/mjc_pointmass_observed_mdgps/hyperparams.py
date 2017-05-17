""" Hyperparameters for MJC 2D navigation policy optimization. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.mjc import obstacle_pointmass
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.tf_model_example import multi_modal_network_fp, example_tf_network
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.policy.lin_gauss_init import init_pd
from gps.algorithm.policy.policy_prior import PolicyPrior
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, CONDITION_DATA
from gps.gui.config import generate_experiment_info


SENSOR_DIMS = {
    JOINT_ANGLES: 2,
    JOINT_VELOCITIES: 2,
    END_EFFECTOR_POINTS: 3,
    END_EFFECTOR_POINT_VELOCITIES: 3,
    ACTION: 2,
    CONDITION_DATA: 1,
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = os.path.dirname(__file__)+'/'
target_pos = np.array([1.3, 0.0, 0.])


common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 5,
    # 'conditions': 1,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    'models': [obstacle_pointmass(target_pos, wall_center=0.0, hole_height=0.3, control_limit=20),
               obstacle_pointmass(target_pos, wall_center=0.15, hole_height=0.3, control_limit=20),
               obstacle_pointmass(target_pos, wall_center=-0.15, hole_height=0.3, control_limit=20),
               obstacle_pointmass(target_pos, wall_center=0.25, hole_height=0.3, control_limit=20),
               obstacle_pointmass(target_pos, wall_center=-0.25, hole_height=0.3, control_limit=20),
               ],
    'condition_data': np.array([[0], [0.15], [-0.15], [0.25], [-0.25]]),
    #'filename': './mjc_models/particle2d.xml',
    'x0': np.array([-1., 0., 0., 0.]),
    # 'x0': [np.array([-1., 1., 0., 0.])],
    'dt': 0.05,
    'substeps': 1,
    'conditions': common['conditions'],
    'T': 200,
    'point_linear': True,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, CONDITION_DATA, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, CONDITION_DATA, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'smooth_noise': False,
    'camera_pos': np.array([1., 0., 8., 0., 0., 0.]),
}

#"""
algorithm = {
    'type': AlgorithmMDGPS,
    'conditions': common['conditions'],
    'iterations': 15,
    'kl_step': 1.0,
    'min_step_mult': 0.01,
    'max_step_mult': 4.0,
    'max_ent_traj': 1.0,
    'step_rule': 'laplace',
    'sample_on_policy': True,
    'plot_dir': EXP_DIR,
    'target_end_effector': target_pos,
}
#"""

"""
algorithm = {
    'type': AlgorithmBADMM,
    'conditions': common['conditions'],
    'plot_dir': EXP_DIR,
    'max_ent_traj': 1.0,
    'target_end_effector': target_pos,
    'iterations': 15,
    'lg_step_schedule': np.array([1e-4, 1e-3, 1e-2, 1e-1]),
    'policy_dual_rate': 0.1,
    'ent_reg_schedule': np.array([1e-3, 1e-3, 1e-2, 1e-1]),
    'fixed_lg_step': 3,
    'kl_step': 1.0,
    'init_pol_wt': 0.01,
    'min_step_mult': 0.1,
    'max_step_mult': 3.0,
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1,
    'exp_step_increase': 2.0,
    'exp_step_decrease': 0.5,
    'exp_step_upper': 0.5,
    'exp_step_lower': 1.0,
    'max_policy_samples': 6,
    'policy_sample_mode': 'add',
}
"""

algorithm['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 10.0,
    'pos_gains': 10.0,
    'vel_gains_mult': 0.0,
    'dQ': SENSOR_DIMS[ACTION],
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
    'weights': [1., 1.], # used 10,1 for T=3
}

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

# algorithm['policy_opt'] = {
#     'type': PolicyOptCaffe,
#     'weights_file_prefix': EXP_DIR + 'policy',
#     'iterations': 10000,
#     'network_arch_params': {
#         'n_layers': 2,
#         'dim_hidden': [20],
#     },
# }

algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'network_params': {
        'obs_include': agent['obs_include'],
        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, CONDITION_DATA, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
        'sensor_dims': SENSOR_DIMS,
    },
    'network_model': example_tf_network,
    'fc_only_iterations': 5000,
    'init_iterations': 1000,
    'iterations': 1000,  # was 100
    'weights_file_prefix': EXP_DIR + 'policy',
}

algorithm['policy_prior'] = {
    'type': PolicyPrior,
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 10,
    'verbose_trials': 1,
    #'verbose_policy_trials': 1,
    'common': common,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)

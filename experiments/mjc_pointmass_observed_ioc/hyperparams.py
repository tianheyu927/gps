""" Hyperparameters for MJC 2D navigation policy optimization. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.mjc import obstacle_pointmass
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_ioc_tf import CostIOCTF
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.policy.lin_gauss_init import init_pd
from gps.algorithm.policy.policy_prior import PolicyPrior
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
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
DEMO_DIR = BASE_DIR + '/../experiments/mjc_pointmass_observed/'
target_pos = np.array([1.3, 0.0, 0.])


common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'demo_exp_dir': DEMO_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'demo_controller_file': DEMO_DIR + 'data_files/algorithm_itr_14.pkl',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 4,
    'demo_conditions': 4,
    # 'conditions': 1,

    'LG_demo_file': os.path.join(EXP_DIR, 'data_files', 'demos_LG.pkl'),
    'NN_demo_file': os.path.join(EXP_DIR, 'data_files', 'demos_NN.pkl'),
    'nn_demo': False,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    'models': [
               obstacle_pointmass(target_pos, wall_center=0.4, hole_height=0.3, control_limit=50),
               obstacle_pointmass(target_pos, wall_center=-0.4, hole_height=0.3, control_limit=50),
               obstacle_pointmass(target_pos, wall_center=0.5, hole_height=0.3, control_limit=50),
               obstacle_pointmass(target_pos, wall_center=-0.5, hole_height=0.3, control_limit=50),
               ],
    'condition_data': np.array([[0.4], [-0.4], [0.5], [-0.5]]),
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

demo_agent = {
    'type': AgentMuJoCo,
    'models': [obstacle_pointmass(target_pos, wall_center=0.0, hole_height=0.3, control_limit=50),
               obstacle_pointmass(target_pos, wall_center=0.2, hole_height=0.3, control_limit=50),
               obstacle_pointmass(target_pos, wall_center=-0.2, hole_height=0.3, control_limit=50),
               obstacle_pointmass(target_pos, wall_center=0.3, hole_height=0.3, control_limit=50),
               obstacle_pointmass(target_pos, wall_center=-0.3, hole_height=0.3, control_limit=50),
               ],
    'filename': '',
    'condition_data': np.array([[0], [0.2], [-0.2], [0.3], [-0.3]]),
    'x0': np.array([-1., 0., 0., 0.]),
    # 'x0': [np.array([-1., 1., 0., 0.])],
    'dt': 0.05,
    'substeps': 1,
    'conditions': common['demo_conditions'],
    'T': agent['T'],
    'point_linear': True,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, CONDITION_DATA, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, CONDITION_DATA, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'smooth_noise': False,
    'camera_pos': np.array([1., 0., 8., 0., 0., 0.]),
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    #'sample_on_policy': True,
    'iterations': 15,
    'kl_step': 1.0,
    'min_step_mult': 0.1,
    'max_step_mult': 4.0,
    'max_ent_traj': 1.0,
    'step_rule': 'laplace',
    'target_end_effector': target_pos,
    'ioc' : 'ICML',
    'num_demos': 10,
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

algorithm['gt_cost'] = {
    'type': CostSum,
    'costs': [state_cost, action_cost],
    'weights': [0.1, 0.1], # used 10,1 for T=3
}

algorithm['cost'] = {
    #'type': CostIOCQuadratic,
    'type': CostIOCTF,
    'wu': np.array([1e-5, 1e-5]),
    'dO': np.sum([SENSOR_DIMS[dtype] for dtype in agent['obs_include']]),
    'T': agent['T'],
    'iterations': 1000,
    'demo_batch_size': 10,
    'sample_batch_size': 10,
    'ioc_loss': algorithm['ioc'],
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


algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 20,
}


algorithm['policy_opt'] = {
     'type': PolicyOptCaffe,
     'weights_file_prefix': EXP_DIR + 'policy',
     'iterations': 10000,
     'network_arch_params': {
         'n_layers': 2,
         'dim_hidden': [20],
     },
 }

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 15,
    'verbose_trials': 1,
    'verbose_policy_trials': 1,
    'demo_agent': demo_agent,
    'common': common,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)

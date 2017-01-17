from __future__ import division

from datetime import datetime
import os.path
import numpy as np
import operator

from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.agent.mjc.mjc_models import half_cheetah_hop
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_ioc_supervised_tf import CostIOCSupervised
from gps.algorithm.cost.cost_ioc_tf import CostIOCTF
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd, load_from_file
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY, RAMP_QUADRATIC, evall1l2term
from gps.utility.data_logger import DataLogger

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, RGB_IMAGE, RGB_IMAGE_SIZE, ACTION
from gps.gui.config import generate_experiment_info

SENSOR_DIMS = {
    JOINT_ANGLES: 9,
    JOINT_VELOCITIES: 9,
    ACTION: 6,
}

BASE_DIR = '/'.join(str.split(__file__, '/')[:-2])
EXP_DIR = '/'.join(str.split(__file__, '/')[:-1]) + '/'
DEMO_DIR = BASE_DIR + '/../experiments/cheetah_hop/'
x
CONDITIONS = 1
TARGET_X = 5.0
TARGET_Z = 0.2+0.1

np.random.seed(47)
x0 = []
for _ in range(CONDITIONS):
    x0.append(np.concatenate((0.2*np.random.rand(9)-0.1, 0.1*np.random.randn(9))))

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': CONDITIONS,
    'demo_conditions': 1,
    'demo_exp_dir': DEMO_DIR,
    'demo_controller_file': DEMO_DIR + 'data_files/demo_algorithm.pkl',
    'LG_demo_file': os.path.join(EXP_DIR, 'data_files', 'demos_LG.pkl'),
    'NN_demo_file': os.path.join(EXP_DIR, 'data_files', 'demos_NN.pkl'),
    'nn_demo': False,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    'models': [
        half_cheetah_hop(wall_height=0.5, wall_pos=1.8, gravity=1.0), #wall_sample_range=[0.6, 1.0])
    ],
    'x0': x0[:CONDITIONS],
    'dt': 0.05,
    'substeps': 5,
    'conditions': common['conditions'],
    'T': 200,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES],
    'meta_include': [],
    'camera_pos': np.array([0., -12., 7., 0., 0., 0.]),
    'record_reward': False,
}

demo_agent = {
    'type': AgentMuJoCo,
    'models': [
        half_cheetah_hop(wall_height=0.2, wall_pos=1.8, gravity=1.0)
    ],
    'x0': x0,
    'dt': 0.05,
    'substeps': 5,
    'conditions': common['demo_conditions'],
    'T': 200,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES],
    'meta_include': [],
    'camera_pos': np.array([0., -12., 7., 0., 0., 0.]),
    'record_reward': False,

    'filter_demos': {
        'type': 'min',
        'state_idx': range(0, 1),
        'target': np.array([TARGET_X]),
        'success_upper_bound': 1.0,
    },
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'iterations': 60,
    'kl_step': 0.1,
    'min_step_mult': 1.0,
    'max_step_mult': 10.0,
    'max_ent_traj': 1.0,
    'ioc' : 'ICML',
    'num_demos': 20,
    'init_samples': 20,

    'compute_distances': {
        'type': 'min',
        'targets': [np.array([TARGET_X]) for i in xrange(common['demo_conditions'])],
        'state_idx': list(range(0, 1)),
    }
}

algorithm['policy_opt'] = {
}

algorithm['init_traj_distr'] = {
    'type': load_from_file,
    'init_gains':  np.zeros(SENSOR_DIMS[ACTION]),
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 1e-2,
    'dt': agent['dt'],
    'T': agent['T'],
    'algorithm_file': common['demo_controller_file']
}

torque_cost_1 = {
    'type': CostAction,
    'wu': np.array([1.0 ,1.0, 1.0, 1.0, 1.0, 1.0])*1e-4,
}

state_cost = {
    'type': CostState,
    'l1': 0.0,
    'l2': 10.0,
    'alpha': 1e-5,
    'evalnorm': evall1l2term,
    'data_types': {
        JOINT_ANGLES: {
            'target_state': np.array([TARGET_X, TARGET_Z]+[0.0]*7),
            'wp': np.array([1.0, 0.5] + [0.0]*7)
        },
        JOINT_VELOCITIES: {
            'target_state': np.array([1., 1.]+[0.0]*7),
            'wp': np.array([1.0, 1.0] + [0.0]*7)
        },
    },

}

algorithm['gt_cost'] = {
    'type': CostSum,
    'costs': [torque_cost_1, state_cost],
    'weights': [1.0, 1.0],
}

algorithm['cost'] = {
    'type': CostIOCSupervised,
    'wu': np.ones(6)*1e-4,
    'dO': np.sum([SENSOR_DIMS[dtype] for dtype in agent['obs_include']]),
    'T': agent['T'],
    'iterations': 2000,
    'init_iterations': 10000,
    'demo_batch_size': 10,
    'sample_batch_size': 10,
    'ioc_loss': algorithm['ioc'],
    'agent': demo_agent,
    'demo_file': common['LG_demo_file'],
    'gt_cost': algorithm['gt_cost'],
    'traj_samples': [DEMO_DIR + 'data_files/traj_sample_itr_05.pkl']
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
        'max_iters': 20,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}


algorithm['policy_prior'] = {
}


config = {
    'iterations': algorithm['iterations'],
    'num_samples': 20,
    'verbose_trials': 0,
    'verbose_policy_trials': 0,
    'common': common,
    'agent': agent,
    'demo_agent': demo_agent,
    'gui_on': True,
    'algorithm': algorithm,
    'conditions': common['conditions'],
}
seed = 4

common['info'] = generate_experiment_info(config)

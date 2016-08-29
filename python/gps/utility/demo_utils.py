""" This file generates more demonstration positions for MJC peg insertion experiment. """
import numpy as np
import copy
import numpy.matlib
import random

from gps.utility.data_logger import DataLogger
from gps.sample.sample import Sample


def generate_pos_body_offset(conditions):
	""" Generate position body offset for all conditions. """
	pos_body_offset = []
	for i in xrange(conditions):
		# Make the uniform distribution to be [-0.12, 0.12] for learning from prior. For peg ioc, this should be [-0.1, -0.1].
		pos_body_offset.append(np.hstack((0.24 * (np.random.rand(1, 2) - 0.5), np.zeros((1, 1)))).reshape((3, )))
	return pos_body_offset

def generate_x0(x0, conditions):
	""" Generate initial states for all conditions. """
	x0_lst = [x0.copy() for i in xrange(conditions)]
	for i in xrange(conditions):
		min_pos = np.array(([-2.0], [-0.5], [0.0]))
		max_pos = np.array(([0.2], [0.5], [0.5]))
		direct = np.random.rand(3, 1) * (max_pos - min_pos) + min_pos
		J = np.array(([-0.4233, -0.0164, 0.0], [-0.1610, -0.1183, -0.1373], [0.0191, 0.1850, 0.3279], \
    					[0.1240, 0.2397, -0.2643], [0.0819, 0.2126, -0.0393], [-0.1510, 0.0177, -0.1714], \
						[0.0734, 0.1308, 0.0003]))
		x0_lst[i][range(7)] += J.dot(direct).reshape((7, ))
	return x0_lst

def generate_pos_idx(conditions):
	""" Generate the indices of states. """
	return [np.array([1]) for i in xrange(conditions)]


def eval_demos(agent, demo_file, costfn, n=10):
	demos = DataLogger.unpickle(demo_file)
	demoX = demos['demoX']
	demoU = demos['demoU']
	return eval_demos_xu(agent, demoX, demoU, costfn, n=n)

def eval_demos_xu(agent, demoX, demoU, costfn, n=10):
	num_demos = demoX.shape[0]
	losses = []
	for demo_idx in range(num_demos):
		sample = Sample(agent)
		sample.set_XU(demoX[demo_idx], demoU[demo_idx])
		l, _, _, _, _, _ = costfn.eval(sample)
		losses.append(l)
	return random.sample(losses, n)

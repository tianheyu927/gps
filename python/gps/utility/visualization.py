from gps.proto.gps_pb2 import END_EFFECTOR_POINTS, RGB_IMAGE
from gps.utility.general_utils import flatten_lists
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import logging
import imp
import os
import os.path
import sys
import copy
import argparse
import threading
import time
import scipy.io

TRIALS = 20
THRESH = {'reacher': 0.05, 'pointmass': 0.1}

def compare_samples(gps, N, agent_config, three_dim=False, weight_varying=False, experiment='reacher'):
    """
    Compare samples between IOC and demo policies and visualize them in a plot.
    Args:
        gps: GPS object.
        N: number of samples taken from the policy for comparison
        config: Configuration of the agent to sample.
        three_dim: whether the plot is 3D or 2D.
        weight_varying: whether the experiment is weight-varying or not.
        experiment: whether the experiment is peg, reacher or pointmass.
    """
    assert N == 1
    pol_iter = gps._hyperparams['algorithm']['iterations'] - 1
    pol_iter = 14
    # pol_iter = 13
    algorithm = gps.data_logger.unpickle(gps._data_files_dir + 'algorithm_itr_%02d' % pol_iter + '.pkl')
    if not weight_varying:
        pos_body_offset = gps._hyperparams['agent']['pos_body_offset']
    M = agent_config['conditions']
    # if experiment == 'reacher' and not weight_varying: #reset body offsets
    #     np.random.seed(101)
    #     for m in range(M):
    #         self.agent.reset_initial_body_offset(m)

    pol = algorithm.policy_opt.policy
    # pol_bc = algorithm_bc.policy_opt.policy
    policies = [pol]
    samples = {i: [] for i in xrange(len(policies))}
    agent = agent_config['type'](agent_config)
    if not weight_varying:
        ioc_conditions = [agent_config['target_end_effector'][i][:3] for i in xrange(M)]
    else:
        ioc_conditions = [np.array([np.log10(agent_config['density_range'][i]), 0.]) \
                            for i in xrange(M)]
        print M
        print len(agent_config['density_range'])
    
    from gps.utility.general_utils import mkdir_p

    if 'record_gif' in gps._hyperparams:
        gif_config = gps._hyperparams['record_gif']
        gif_fps = gif_config.get('fps', None)
        gif_dir = gif_config.get('test_gif_dir', gps._hyperparams['common']['data_files_dir'])
        mkdir_p(gif_dir)
    for i in xrange(M):
        # Gather demos.
        for j in xrange(N):
            for k in xrange(len(samples)):
                gif_name = os.path.join(gif_dir, 'pol%d_cond%d.gif' % (k, i))
                sample = agent.sample(
                    policies[k], i,
                    verbose=(i < gps._hyperparams['verbose_trials']), noisy=True,
                    record_gif=gif_name, record_gif_fps=gif_fps
                    )
                samples[k].append(sample)

    dists_to_target = [np.zeros((M*N)) for i in xrange(len(samples))]
    dists_diff = []
    # all_success_conditions, only_ioc_conditions, only_demo_conditions, all_failed_conditions, \
    #     percentages = [], [], [], [], []
    success_conditions, failed_conditions, percentages = [], [], []
    for i in xrange(len(samples[0])):
        if experiment == 'reacher':
            target_position = agent_config['target_end_effector'][i][:3]
        for j in xrange(len(samples)):
            sample_end_effector = samples[j][i].get(END_EFFECTOR_POINTS)
            dists_to_target[j][i] = np.nanmin(np.sqrt(np.sum((sample_end_effector[:, :3] - target_position.reshape(1, -1))**2, axis = 1)), axis = 0)
            # Just choose the last time step since it may become unstable after achieving the minimum point.
            # import pdb; pdb.set_trace()
            # dists_to_target[j][i] = np.sqrt(np.sum((sample_end_effector[:, :3] - target_position.reshape(1, -1))**2, axis = 1))[-1]
        if dists_to_target[0][i] <= THRESH['reacher']:
            success_conditions.append(ioc_conditions[i])
        else:
            failed_conditions.append(ioc_conditions[i])
        # dists_diff.append(np.around(dists_to_target[0][i] - dists_to_target[1][i], decimals=2))
    percentages.append(round(float(len(success_conditions))/len(ioc_conditions), 2))
    percentages.append(round(float(len(failed_conditions))/len(ioc_conditions), 2))
    from matplotlib.patches import Rectangle

    plt.close('all')
    fig = plt.figure(figsize=(8, 4))
    ioc_conditions_zip = zip(*ioc_conditions)
    success_zip = zip(*success_conditions)
    failed_zip = zip(*failed_conditions)

    if three_dim:
        ax = Axes3D(fig)
        ax.scatter(all_success_zip[0], all_success_zip[1], all_success_zip[2], c='y', marker='o')
        ax.scatter(all_failed_zip[0], all_failed_zip[1], all_failed_zip[2], c='r', marker='x')
        ax.scatter(only_ioc_zip[0], only_ioc_zip[1], only_ioc_zip[2], c='g', marker='^')
        ax.scatter(only_demo_zip[0], only_demo_zip[1], only_demo_zip[2], c='r', marker='v')
        training_positions = zip(*pos_body_offset)
        ax.scatter(training_positions[0], training_positions[1], training_positions[2], s=40, c='b', marker='*')
        box = ax.get_position()
    else:
        subplt = plt.subplot()
        subplt.scatter(success_zip[0], success_zip[1], c='g', marker='o', s=50, lw=0)
        if len(failed_conditions) > 0:
            subplt.scatter(failed_zip[0], failed_zip[1], c='r', marker='x', s=50)
        for i in xrange(M*N):
            subplt.annotate(repr(round(dists_to_target[0][i],2)), (ioc_conditions_zip[0][i], ioc_conditions_zip[1][i]))
        # for i, txt in enumerate(dists_to_target[0]):
        #     subplt.annotate(repr(round(txt,2)), (ioc_conditions_zip[0][i], 0.5))
        # for i, txt in enumerate(dists_to_target[1]):
        #     subplt.annotate(repr(round(txt,2)), (ioc_conditions_zip[0][i], 0.0))
        # for i, txt in enumerate(dists_to_target[2]):
        #     subplt.annotate(repr(round(txt,2)), (ioc_conditions_zip[0][i], -0.5))
        ax = plt.gca()
        ax.legend(['success: ' + repr(percentages[0]), 'failed: ' + repr(percentages[1])], \
                    loc='upper center', bbox_to_anchor=(0.4, -0.05), shadow=True, ncol=3)
        if experiment == 'peg':
            ax.add_patch(Rectangle((-0.1, -0.1), 0.2, 0.2, fill = False, edgecolor = 'blue')) # peg
        if experiment == 'reacher':
            ax.add_patch(Rectangle((-0.1, -0.1), 0.2, 0.2, fill = False, edgecolor = 'blue'))
        # for i in xrange(len(policies)):
        #     subplt.annotate(pol_names[i], (ax.get_xticks()[0], yrange[i]), horizontalalignment='left')
        # for i in xrange(len(policies)):
            # subplt.annotate(repr(percentages[2*i]*100) + "%", (ax.get_xticks()[-1], yrange[i]), color='green')
        # ax.xaxis.set_ticks_position('bottom')
        # ax.yaxis.set_ticks_position('left')
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # ax.tick_params(axis='y', which='both',length=0)
    # ax.legend(['all_success: ' + repr(percentages[0]), 'all_failed: ' + repr(percentages[1]), 'only_ioc: ' + repr(percentages[2]), \
    #                 'only_demo: ' + repr(percentages[3])], loc='upper center', bbox_to_anchor=(0.5, -0.05), \
    #                 shadow=True, ncol=2)
    # subplt.plot(all_success_zip[0], [x - 0.5 for x in all_success_zip[1]], c='y', marker='o')
    # if len(all_failed_zip) > 0:
    #     subplt.plot(all_failed_zip[0], [x - 0.5 for x in all_failed_zip[1]], c='r', marker='x')
    # else:
    #     subplt.plot([], [], c='r', marker='x')
    # plt.title("Distribution of samples drawn from various policies of 2-link arm task")
    plt.title("Reacher with multiple conditions")
    plt.savefig(gps._data_files_dir + 'reacher_multiple.png')
    plt.close('all')


def compare_samples_curve(gps, N, agent_config, three_dim=True, weight_varying=False, experiment='peg'):
    """
    Compare samples between IOC and demo policies and visualize them in a plot.
    Args:
        gps: GPS object.
        N: number of samples taken from the policy for comparison
        config: Configuration of the agent to sample.
        three_dim: whether the plot is 3D or 2D.
        weight_varying: whether the experiment is weight-varying or not.
        experiment: whether the experiment is peg, reacher or pointmass.
    """
    pol_iter = gps._hyperparams['algorithm']['iterations'] - 1
    algorithm_ioc = gps.data_logger.unpickle(gps._data_files_dir + 'algorithm_itr_%02d' % pol_iter + '.pkl')
    algorithm_sup = gps.data_logger.unpickle(gps._hyperparams['common']['supervised_exp_dir'] + 'data_files/algorithm_itr_%02d' % pol_iter + '.pkl')
    algorithm_demo = gps.data_logger.unpickle(gps._hyperparams['common']['demo_exp_dir'] + 'data_files/algorithm_itr_09.pkl') # Assuming not using 4 policies
    algorithm_oracle = gps.data_logger.unpickle(gps._hyperparams['common']['demo_exp_dir'] + 'data_files_oracle/algorithm_itr_09.pkl')
    if not weight_varying:
        pos_body_offset = gps._hyperparams['agent']['pos_body_offset']
    M = agent_config['conditions']
    if experiment == 'reacher' and not weight_varying: #reset body offsets
        np.random.seed(101)
        for m in range(M):
            self.agent.reset_initial_body_offset(m)

    pol_ioc = algorithm_ioc.policy_opt.policy
    pol_sup = algorithm_sup.policy_opt.policy
    pol_demo = algorithm_demo.policy_opt.policy
    pol_oracle = algorithm_oracle.policy_opt.policy
    policies = [pol_ioc, pol_sup, pol_demo, pol_oracle]
    successes = {i: np.zeros((TRIALS, M)) for i in xrange(len(policies))}
    agent = agent_config['type'](agent_config)
    if not weight_varying:
        ioc_conditions = agent_config['pos_body_offset']
    else:
        ioc_conditions = [np.log10(agent_config['density_range'][i])-4.0 \
                            for i in xrange(M)]
    pos_body_offset = gps.agent._hyperparams['pos_body_offset'][i]
    target_position = np.array([.1,-.1,.01])+pos_body_offset

    import random

    for seed in xrange(TRIALS):
        random.seed(seed)
        np.random.seed(seed)
        for i in xrange(M):
            # Gather demos.
            for j in xrange(N):
                for k in xrange(len(policies)):
                    sample = agent.sample(
                        policies[k], i,
                        verbose=(i < gps._hyperparams['verbose_trials']), noisy=True
                        )
                    sample_end_effector = sample.get(END_EFFECTOR_POINTS)
                    dists_to_target = np.nanmin(np.sqrt(np.sum((sample_end_effector[:, :3] - target_position.reshape(1, -1))**2, axis = 1)), axis = 0)
                    if dists_to_target <= THRESH[experiment]:
                        successes[k][seed, i] = 1.0
    
    success_rate_ioc = np.mean(successes[0], axis=0)
    success_rate_sup = np.mean(successes[1], axis=0)
    success_rate_demo = np.mean(successes[2], axis=0)
    success_rate_oracle = np.mean(successes[3], axis=0)
    print "ioc mean: " + repr(success_rate_ioc.mean())
    print "sup mean: " + repr(success_rate_sup.mean())
    print "demo mean: " + repr(success_rate_demo.mean())
    print "oracle mean: " + repr(success_rate_oracle.mean())
    print "ioc: " + repr(success_rate_ioc)
    print "sup: " + repr(success_rate_sup)
    print "demo: " + repr(success_rate_demo)
    print "oracle: " + repr(success_rate_oracle)


    from matplotlib.patches import Rectangle

    plt.close('all')
    fig = plt.figure(figsize=(8, 5))


    if three_dim:
        ax = Axes3D(fig)
        ax.scatter(all_success_zip[0], all_success_zip[1], all_success_zip[2], c='y', marker='o')
        ax.scatter(all_failed_zip[0], all_failed_zip[1], all_failed_zip[2], c='r', marker='x')
        ax.scatter(only_ioc_zip[0], only_ioc_zip[1], only_ioc_zip[2], c='g', marker='^')
        ax.scatter(only_demo_zip[0], only_demo_zip[1], only_demo_zip[2], c='r', marker='v')
        training_positions = zip(*pos_body_offset)
        ax.scatter(training_positions[0], training_positions[1], training_positions[2], s=40, c='b', marker='*')
        box = ax.get_position()
    else:
        subplt = plt.subplot()
        subplt.plot(ioc_conditions, success_rate_ioc, '-rx')
        subplt.plot(ioc_conditions, success_rate_sup, '-gx')
        subplt.plot(ioc_conditions, success_rate_demo, '-bx')
        subplt.plot(ioc_conditions, success_rate_oracle, '-yx')
        ax = plt.gca()
        if experiment == 'peg':
            ax.add_patch(Rectangle((-0.1, -0.1), 0.2, 0.2, fill = False, edgecolor = 'blue')) # peg
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    ax.legend(['S3G', 'cost regr', 'RL policy', 'oracle'], loc='lower left')
    plt.ylabel('Success Rate', fontsize=22)
    plt.xlabel('Log Mass', fontsize=22, labelpad=-4)
    plt.title("Generalization for 2-link reacher", fontsize=25)
    plt.savefig(gps._data_files_dir + 'distribution_of_sample_conditions_average_curve.png')
    plt.close('all')

def visualize_samples(gps, N, agent_config, experiment='reacher'):
    """
    Compare samples between IOC and demo policies and visualize them in a plot.
    Args:
        gps: GPS object.
        N: number of samples taken from the policy for comparison
        config: Configuration of the agent to sample.
        experiment: whether the experiment is peg, reacher or pointmass.
    """
    pol_iter = gps._hyperparams['algorithm']['iterations'] - 1
    algorithm_ioc = gps.data_logger.unpickle(gps._data_files_dir + 'algorithm_itr_%02d' % pol_iter + '.pkl')
    algorithm_sup = gps.data_logger.unpickle(gps._hyperparams['common']['supervised_exp_dir'] + 'data_files/algorithm_itr_%02d' % pol_iter + '.pkl')
    algorithm_demo = gps.data_logger.unpickle(gps._hyperparams['common']['demo_exp_dir'] + 'data_files/algorithm_itr_09.pkl') # Assuming not using 4 policies
    algorithm_oracle = gps.data_logger.unpickle(gps._hyperparams['common']['demo_exp_dir'] + 'data_files_oracle/algorithm_itr_09.pkl')
    M = agent_config['conditions']
    pol_ioc = algorithm_ioc.policy_opt.policy
    pol_sup = algorithm_sup.policy_opt.policy
    pol_demo = algorithm_demo.policy_opt.policy
    pol_oracle = algorithm_oracle.policy_opt.policy
    policies = [pol_ioc, pol_demo, pol_oracle, pol_sup]
    samples = {i: [] for i in xrange(len(policies))}
    agent = agent_config['type'](agent_config)
    ioc_conditions = [np.array([np.log10(agent_config['density_range'][i]), 0.]) \
                        for i in xrange(M)]
    print "Number of testing conditions: %d" % M

    from gps.utility.general_utils import mkdir_p

    if 'record_gif' in gps._hyperparams:
        gif_config = gps._hyperparams['record_gif']
        gif_fps = gif_config.get('fps', None)
        gif_dir = gif_config.get('gif_dir', gps._hyperparams['common']['data_files_dir'])
        mkdir_p(gif_dir)
    for i in xrange(M):
        # Gather demos.
        for j in xrange(N):
            for k in xrange(len(samples)):
                gif_name = os.path.join(gif_dir, 'pol%d_cond%d.gif' % (k, i))
                sample = agent.sample(
                    policies[k], i,
                    verbose=(i < gps._hyperparams['verbose_trials']), noisy=True, record_image=True, \
                    record_gif=gif_name, record_gif_fps=gif_fps
                    )
                samples[k].append(sample)

def get_comparison_hyperparams(hyperparam_file, itr):
    """ 
    Make the iteration number the same as the experiment data directory index.
    Args:
        hyperparam_file: the hyperparam file to be changed for two different experiments for comparison.
    """
    hyperparams = imp.load_source('hyperparams', hyperparams_file)
    hyperparams.config['plot']['itr'] = itr
    return hyperparams

def compare_experiments(mean_dists_1_dict, mean_dists_2_dict, success_rates_1_dict, \
                                success_rates_2_dict, pol_iter, exp_dir, hyperparams_compare, \
                                hyperparams):
    """ 
    Compare the performance of two experiments and plot their mean distance to target effector and success rate.
    Args:
        mean_dists_1_dict: mean distance dictionary for one of two experiments to be compared.
        mean_dists_2_dict: mean distance dictionary for one of two experiments to be compared.
        success_rates_1_dict: success rates dictionary for one of the two experiments to be compared.
        success_rates_2_dict: success rates dictionary for one of the two experiments to be compared.
        pol_iter: number of iterations of the algorithm.
        exp_dir: directory of the experiment.
        hyperparams_compare: the hyperparams of the control group.
        hyperparams: the hyperparams of the experimental group.
    """

    plt.close('all')
    avg_dists_1 = [float(sum(mean_dists_1_dict[i][j] for i in seeds))/3 for j in xrange(pol_iter)]
    avg_succ_rate_1 = [float(sum(success_rates_1_dict[i][j] for i in seeds))/3 for j in xrange(pol_iter)]
    avg_dists_2 = [float(sum(mean_dists_2_dict[i][j] for i in seeds))/3 for j in xrange(pol_iter)]
    avg_succ_rate_2 = [float(sum(success_rates_2_dict[i][j] for i in seeds))/3 for j in xrange(pol_iter)]
    plt.plot(range(pol_iter), avg_dists_1, '-x', color='red')
    plt.plot(range(pol_iter), avg_dists_2, '-x', color='green')
    for i in seeds:
        plt.plot(range(pol_iter), mean_dists_1_dict[i], 'ko')
        plt.plot(range(pol_iter), mean_dists_2_dict[i], 'co')
    avg_legend, legend, avg_compare_legend, compare_legend = hyperparams_compare['plot']['avg_legend'], \
            hyperparams_compare['plot']['legend'], hyperparams_compare['plot']['avg_legend_compare'], \
            hyperparams_compare['plot']['legend_compare']
    plt.legend([avg_legend, legend, avg_compare_legend, compare_legend], loc='upper right', ncol=2)
    plt.title(hyperparams_compare['plot']['mean_dist_title'])
    plt.xlabel(hyperparams_compare['plot']['xlabel'])
    plt.ylabel(hyperparams_compare['plot']['ylabel'])
    plt.savefig(exp_dir + hyperparams_compare['plot']['mean_dist_plot_name'])
    plt.close()
    plt.plot(range(pol_iter), avg_succ_rate_1, '-x', color='red')
    plt.plot(range(pol_iter), avg_succ_rate_2, '-x', color='green')
    for i in seeds:
        plt.plot(range(pol_iter), success_rates_1_dict[i], 'ko')
        plt.plot(range(pol_iter), success_rates_2_dict[i], 'co')
    plt.legend([avg_legend, legend, avg_compare_legend, compare_legend], loc='upper right', ncol=2)
    plt.xlabel(hyperparams_compare['plot']['xlabel'])
    plt.ylabel(hyperparams_compare['plot']['ylabel'])
    plt.title(hyperparams_compare['success_title'])
    plt.savefig(exp_dir + hyperparams_compare['plot']['success_plot_name'])

    plt.close()


def plot_cost_3d(gps, sample_lists, costfns):
    """
    Plot the visualization of costs of samples versus its two end-effector coordinates.
    For reacher specifically. 
    Args:
        sample_lists: a list of SampleList
        costfns: a list of cost functions
    """
    # samples = flatten_lists([sample_lists[key].get_samples() for key in sample_lists])
    samples = sample_lists[max(sample_lists.keys())].get_samples()
    T = samples[0].get_X().shape[0]
    costs = np.zeros((len(costfns), len(samples)*T))
    eepts_x = np.hstack([sample.get(END_EFFECTOR_POINTS)[:, 0] for sample in samples])
    eepts_y = np.hstack([sample.get(END_EFFECTOR_POINTS)[:, 1] for sample in samples])
    target_eepts = samples[0].get(END_EFFECTOR_POINTS)[0, 3:5] # Assuming one condition
    # fk cost only
    for i in xrange(len(costfns)-1):
        costs[i, :] = np.hstack([costfns[i].eval(sample, wu=False)[0] for sample in samples])
    costs[-1, :] = np.hstack([costfns[-1].eval(sample)[0] for sample in samples])
    vmax = np.amax([np.sort(costs[i, :])[-1001] for i in xrange(len(costfns)-1)])
    vmin = np.amin([np.sort(costs[i, :])[0] for i in xrange(len(costfns)-1)])
    # import pdb; pdb.set_trace()
    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # colors = ['g', 'b', 'k']
    # shape = ['^', 'o', 'D']
    print "num of cost functions: %d" % len(costfns)
    # eepts_x, eepts_y = np.meshgrid(eepts_x, eepts_y)
    # for i in xrange(len(costfns)):
    #     ax.plot_surface(eepts_x, eepts_y, costs[i, :], color=colors[i])
    # ax.scatter(target_eepts[0], target_eepts[1], 0, color='r')
    # plt.title('costs versus end effectors')
    # plt.savefig(gps._data_files_dir + 'costs_diff.png')
    # ax = fig.add_subplot(111, projection='3d')    # plt.pcolormesh(eepts_x, eepts_y, costs[0, :], cmap='RdBu')
    mask = costs[0, :].argsort()[:-1000]
    print mask
    plt.scatter(eepts_x[mask], eepts_y[mask], c=costs[0, :][mask], cmap=cm.coolwarm, vmin=vmin, vmax=vmax,
                       linewidth=0)
    print target_eepts
    # plt.scatter(target_eepts[0], target_eepts[1], 'ko')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.title('learned cost 0 versus end effectors')
    plt.savefig(gps._data_files_dir + 'cost_0.png')
    
    plt.figure()
    mask = costs[1, :].argsort()[:-1000]
    print mask
    plt.scatter(eepts_x[mask], eepts_y[mask], c=costs[1, :][mask], cmap=cm.coolwarm, vmin=vmin, vmax=vmax,
                       linewidth=0)
    print target_eepts
    # plt.scatter(target_eepts[0], target_eepts[1], 'ko')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.title('learned cost 1 versus end effectors')
    plt.savefig(gps._data_files_dir + 'cost_1.png')

    plt.figure()
    mask = costs[2, :].argsort()[:-1000]
    print mask
    plt.scatter(eepts_x[mask], eepts_y[mask], c=costs[2, :][mask], cmap=cm.coolwarm, vmin=vmin, vmax=vmax,
                       linewidth=0)
    print target_eepts
    # plt.scatter(target_eepts[0], target_eepts[1], 'ko')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.title('learned cost 2 versus end effectors')
    plt.savefig(gps._data_files_dir + 'cost_2.png')

    fig0 = plt.figure()
    # ax0 = fig0.gca(projection='3d')
    # surf0 = ax0.plot_surface(eepts_x, eepts_y, costs[3, :])
    # plt.pcolormesh(eepts_x, eepts_y, costs[3, :], cmap='RdBu')
    gt_mask = costs[3, :].argsort()[:-1000]
    print mask
    plt.scatter(eepts_x[gt_mask], eepts_y[gt_mask], c=costs[3, :][gt_mask], cmap=cm.coolwarm,
                       linewidth=0)
    print target_eepts
    # plt.scatter(target_eepts[0], target_eepts[1], 'ko')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.title('gt costs versus end effectors')
    plt.savefig(gps._data_files_dir + 'gt_cost.png')
    
    fig1 = plt.figure()
    # ax1 = fig1.gca(projection='3d')
    # surf1 = ax1.plot_surface(eepts_x, eepts_y, costs[0, :] - costs[1, :])    # plt.pcolormesh(eepts_x, eepts_y, costs[0, :] - costs[1, :], cmap='RdBu')
    plt.scatter(eepts_x, eepts_y, c=costs[0, :] - costs[1, :], cmap=cm.coolwarm,
                       linewidth=0)
    print target_eepts
    # plt.scatter(target_eepts[0], target_eepts[1], 'ko')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.title('cost diff between 0 and 1 versus end effectors')
    plt.savefig(gps._data_files_dir + 'costs_diff_01.png')
    
    fig2 = plt.figure()    # plt.pcolormesh(eepts_x, eepts_y, costs[0, :] - costs[2, :], cmap='RdBu')
    plt.scatter(eepts_x, eepts_y, c=costs[0, :] - costs[2, :], cmap=cm.coolwarm,
                       linewidth=0)
    print target_eepts
    # plt.scatter(target_eepts[0], target_eepts[1], 'ko')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.title('cost diff between 0 and 2 versus end effectors')
    plt.savefig(gps._data_files_dir + 'costs_diff_02.png')
    
    fig3 = plt.figure()
    # ax3 = fig3.add_subplot(111, projection='3d')    # plt.pcolormesh(eepts_x, eepts_y, costs[1, :] - costs[2, :], cmap='RdBu')
    plt.scatter(eepts_x, eepts_y, c=costs[1, :] - costs[2, :], cmap=cm.coolwarm,
                       linewidth=0)
    print target_eepts
    # plt.scatter(target_eepts[0], target_eepts[1], 'ko')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.title('cost diff between 1 and 2 versus end effectors')
    plt.savefig(gps._data_files_dir + 'costs_diff_12.png')
    # plt.savefig(gps._data_files_dir + 'costs_2d.png')
    save_dict = {'costs': costs, 'eepts_x': eepts_x, 'eepts_y': eepts_y, 'target_eepts': target_eepts}
    gps.data_logger.pickle(gps._data_files_dir + 'cost_eepts.pkl', save_dict)


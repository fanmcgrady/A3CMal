# coding=UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import *  # NOQA

from future import standard_library

standard_library.install_aliases()

import logging
import multiprocessing as mp
import os
import statistics
import time
import datetime
import gym

from multiprocessing import Process

import numpy as np
import copy

"""Columns that describe information about an experiment.

steps: number of time steps taken (= number of actions taken)
episodes: number of episodes finished
elapsed: time elapsed so far (seconds)
mean: mean of returns of evaluation runs
median: median of returns of evaluation runs
stdev: stdev of returns of evaluation runs
max: maximum value of returns of evaluation runs
min: minimum value of returns of evaluation runs
"""
_basic_columns = ('steps', 'episodes', 'elapsed', 'mean',
                  'median', 'stdev', 'max', 'min')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
test_process = locals()
process_path = "process_log.txt"
history_path = "history_log.txt"
TEST_NAME = 'malware-test-v0'


def delete(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        os.remove(c_path)

def test(id, env, obs, agent, scores, steps, max_episode_len=None, explorer=None):
    done = False
    test_r = 0
    t = 0
    while not (done or t == max_episode_len):
        def greedy_action_func():
            return agent.act(obs)

        if explorer is not None:
            a = explorer.select_action(t, greedy_action_func)
        else:
            a = greedy_action_func()
        obs, r, done, info = env.step(a)
        test_r += r
        t += 1
    agent.stop_episode()
    score = 0
    if test_r > 0:
        score = 10
    with open(process_path, 'a+') as f:
        f.write("Test {}  reward = {}  score = {} \n".format(id, test_r, score))

    with open(history_path, 'a+') as f:
        k = []
        v = []
        for temp_k, temp_v in env.history.items():
            k = temp_k
            v = temp_v

        if v['evaded']:
            f.write("{}:{}->success\n".format(id, k))
            f.write("actions are: {}\n\n".format(v['actions']))
        else:
            f.write("{}:{}->fail\n".format(id, k))
            f.write("actions are: {}\n\n".format(v['actions']))

    scores[id] = float(score)
    steps[id] = t


def run_evaluation_episodes(env, agent, n_runs, count, max_episode_len=None,
                            explorer=None, logger=None, test_hooks=[]):
    """Run multiple evaluation episodes and return returns.

    Args:
        env (Environment): Environment used for evaluation
        agent (Agent): Agent to evaluate.
        n_runs (int): Number of evaluation runs.
        max_episode_len (int or None): If specified, episodes longer than this
            value will be truncated.
        explorer (Explorer): If specified, the given Explorer will be used for
            selecting actions.
        logger (Logger or None): If specified, the given Logger object will be
            used for logging results. If not specified, the default logger of
            this module will be used.
    Returns:
        List of returns of evaluation runs.
    """
    # logger = logger or logging.getLogger(__name__)
    scores = mp.Array('f', np.zeros(n_runs))
    steps = mp.Array('i', np.zeros((n_runs,), dtype=np.int))

    delete("Sample/original")
    delete("Sample/modification")
    if os.path.exists("history_log.txt"):
        os.remove("history_log.txt")
    if os.path.exists("process_log.txt"):
        os.remove("process_log.txt")

    env = gym.make(TEST_NAME)

    start = datetime.datetime.now()
    with open(process_path, 'a+') as f:
        f.write("start test {}: start time is {} \n".format(count, start))

    # with open(process_path, 'a+') as f:
    #     f.write("wait all porcess end \n")

    for i in range(n_runs):
        obs = env.reset()
        env_temp = copy.copy(env)
        env = copy.copy(env_temp)
        test_process['Process' + str(i)] = Process(target=test,
                                                   args=(i, env_temp, obs, agent, scores, steps, max_episode_len, explorer))
        test_process.get('Process' + str(i)).start()

    with open(process_path, 'a+') as f:
        f.write('Wait all processed end.\n')

    for i in range(n_runs):
        test_process.get('Process' + str(i)).join()
        test_hooks[1](env, agent, count * n_runs + i, steps[i])

    end = datetime.datetime.now()
    with open(process_path, 'a+') as f:
        f.write("end test {}: end time is {} \n".format(count, end))
        f.write("total time is {} \n".format(end - start))
        f.write("scores is {}\n".format(statistics.mean(scores)))

    test_hooks[0](env, agent, count, statistics.mean(scores) / 10)
    return scores


def eval_performance(env, agent, n_runs, count, max_episode_len=None,
                     explorer=None, logger=None, test_hooks=[]):
    """Run multiple evaluation episodes and return statistics.

    Args:
        env (Environment): Environment used for evaluation
        agent (Agent): Agent to evaluate.
        n_runs (int): Number of evaluation runs.
        max_episode_len (int or None): If specified, episodes longer than this
            value will be truncated.
        explorer (Explorer): If specified, the given Explorer will be used for
            selecting actions.
        logger (Logger or None): If specified, the given Logger object will be
            used for logging results. If not specified, the default logger of
            this module will be used.
    Returns:
        Dict of statistics.
    """
    scores = run_evaluation_episodes(
        env, agent, n_runs, count,
        max_episode_len=max_episode_len,
        explorer=explorer,
        logger=logger, test_hooks=test_hooks)
    stats = dict(
        mean=statistics.mean(scores),
        median=statistics.median(scores),
        stdev=statistics.stdev(scores) if n_runs >= 2 else 0.0,
        max=np.max(scores),
        min=np.min(scores))
    return stats


def record_stats(outdir, values):
    with open(os.path.join(outdir, 'scores.txt'), 'a+') as f:
        print('\t'.join(str(x) for x in values), file=f)


def save_agent(agent, t, outdir, logger, suffix=''):
    dirname = os.path.join(outdir, '{}{}'.format(t, suffix))
    agent.save(dirname)
    logger.info('Saved the agent to %s', dirname)


def update_best_model(agent, outdir, t, old_max_score, new_max_score, logger):
    # Save the best model so far
    logger.info('The best score is updated %s -> %s',
                old_max_score, new_max_score)
    save_agent(agent, t, outdir, logger)


class Evaluator(object):

    def __init__(self, agent, env, n_runs, eval_interval,
                 outdir, max_episode_len=None, explorer=None,
                 step_offset=0, test_hooks=[], logger=None):
        self.agent = agent
        self.env = env
        self.max_score = np.finfo(np.float32).min
        self.start_time = time.time()
        self.n_runs = n_runs
        self.eval_interval = eval_interval
        self.outdir = outdir
        self.max_episode_len = max_episode_len
        self.explorer = explorer
        self.step_offset = step_offset
        self.prev_eval_t = (self.step_offset -
                            self.step_offset % self.eval_interval)
        self.logger = logger or logging.getLogger(__name__)
        self.test_hooks = test_hooks

        self.count = 0

        # Write a header line first
        with open(os.path.join(self.outdir, 'scores.txt'), 'w') as f:
            custom_columns = tuple(t[0] for t in self.agent.get_statistics())
            column_names = _basic_columns + custom_columns
            print('\t'.join(column_names), file=f)

    def evaluate_and_update_max_score(self, t, episodes):
        eval_stats = eval_performance(
            self.env, self.agent, self.n_runs,
            max_episode_len=self.max_episode_len, count=self.count, explorer=self.explorer,
            logger=self.logger, test_hooks=self.test_hooks)
        self.count += 1
        elapsed = time.time() - self.start_time
        custom_values = tuple(tup[1] for tup in self.agent.get_statistics())
        mean = eval_stats['mean']
        values = (t,
                  episodes,
                  elapsed,
                  mean,
                  eval_stats['median'],
                  eval_stats['stdev'],
                  eval_stats['max'],
                  eval_stats['min']) + custom_values
        record_stats(self.outdir, values)
        if mean > self.max_score:
            update_best_model(self.agent, self.outdir, t, self.max_score, mean,
                              logger=self.logger)
            self.max_score = mean
        return mean

    def evaluate_if_necessary(self, t, episodes):
        if t >= self.prev_eval_t + self.eval_interval:
            score = self.evaluate_and_update_max_score(t, episodes)
            self.prev_eval_t = t - t % self.eval_interval
            return score
        return None


class AsyncEvaluator(object):

    def __init__(self, n_runs, eval_interval,
                 outdir, max_episode_len=None, explorer=None,
                 step_offset=0, logger=None):

        self.start_time = time.time()
        self.n_runs = n_runs
        self.eval_interval = eval_interval
        self.outdir = outdir
        self.max_episode_len = max_episode_len
        self.explorer = explorer
        self.step_offset = step_offset
        self.logger = logger or logging.getLogger(__name__)

        # Values below are shared among processes
        self.prev_eval_t = mp.Value(
            'l', self.step_offset - self.step_offset % self.eval_interval)
        self._max_score = mp.Value('f', np.finfo(np.float32).min)
        self.wrote_header = mp.Value('b', False)

        # Create scores.txt
        with open(os.path.join(self.outdir, 'scores.txt'), 'a'):
            pass

    @property
    def max_score(self):
        with self._max_score.get_lock():
            v = self._max_score.value
        return v

    def evaluate_and_update_max_score(self, t, episodes, env, agent):
        eval_stats = eval_performance(
            env, agent, self.n_runs,
            max_episode_len=self.max_episode_len, explorer=self.explorer,
            logger=self.logger)
        elapsed = time.time() - self.start_time
        custom_values = tuple(tup[1] for tup in agent.get_statistics())
        mean = eval_stats['mean']
        values = (t,
                  episodes,
                  elapsed,
                  mean,
                  eval_stats['median'],
                  eval_stats['stdev'],
                  eval_stats['max'],
                  eval_stats['min']) + custom_values
        record_stats(self.outdir, values)
        with self._max_score.get_lock():
            if mean > self._max_score.value:
                update_best_model(
                    agent, self.outdir, t, self._max_score.value, mean,
                    logger=self.logger)
                self._max_score.value = mean
        return mean

    def write_header(self, agent):
        with open(os.path.join(self.outdir, 'scores.txt'), 'w') as f:
            custom_columns = tuple(t[0] for t in agent.get_statistics())
            column_names = _basic_columns + custom_columns
            print('\t'.join(column_names), file=f)

    def evaluate_if_necessary(self, t, episodes, env, agent):
        necessary = False
        with self.prev_eval_t.get_lock():
            if t >= self.prev_eval_t.value + self.eval_interval:
                necessary = True
                self.prev_eval_t.value += self.eval_interval
        if necessary:
            with self.wrote_header.get_lock():
                if not self.wrote_header.value:
                    self.write_header(agent)
                    self.wrote_header.value = True
            return self.evaluate_and_update_max_score(t, episodes, env, agent)
        return None

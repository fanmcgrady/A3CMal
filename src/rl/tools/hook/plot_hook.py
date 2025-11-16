from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import visdom
from chainerrl.experiments import StepHook

# visdom是图形库
class PlotHook(StepHook):
    """Hook that will plot the agent statistics.

    You can use this hook to plot
    as follows:

    .. code-block:: python

        hook = VisdomPlotHook('plot title')

    """

    def __init__(self, title, plot_index=0, xlabel='Training epochs', ylabel='values',
                 figsize=(600, 400), margin=(40, 40, 30, 30), fillarea=False):
        self.win = None
        self.opts = dict(
            fillarea=fillarea,
            showlegend=True,
            width=figsize[0],
            height=figsize[1],
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            # marginleft=margin[0],
            # marginright=margin[1],
            # margintop=margin[2],
            # marginbottom=margin[3]
        )

        self.plot_index = plot_index
        self.episode_step = 0
        self.vis = None  # visdom.Visdom(port=8080)
        # assert self.vis.check_connection(), "Fail to connect to Visdom backend!"

    def plot(self, step: int, sig: dict):
        '''Iterative plot
        example: ss.plot(i, dict(a=i*i, b=51-3*i))
        '''
        return  # 禁用Visdom绘图
        X = np.array([[step] * len(sig.keys())])
        Y = np.array([[sig[k] for k in sig.keys()]])
        self.opts['legend'] = list(sig.keys())
        if self.win is None:
            self.win = self.vis.line(Y=Y, X=X, opts=self.opts)
        else:
            self.vis.line(Y=Y, X=X, win=self.win, update='append', opts=self.opts)

    def __call__(self, env, agent, step, value=0):
        if self.plot_index == 0 or self.plot_index == 1:
            if step % 10 == 0:
                stat = agent.get_statistics()
                d = {stat[self.plot_index][0]: stat[self.plot_index][1]}
                self.plot(step, d)
        elif self.plot_index == 2 or self.plot_index == 3:
            stat = agent.get_statistics()
            d = {stat[self.plot_index - 2][0]: stat[self.plot_index - 2][1]}
            self.plot(step, d)
        elif self.plot_index == 4:
            d = {'Steps to finish (train)': value}
            self.plot(step, d)
        elif self.plot_index == 5:
            d = {'Steps to finish (test)': value}
            self.plot(step, d)
        elif self.plot_index == 6:
            d = {'success rate': value}
            self.plot(step, d)
        else:
            if step % 10 == 0:
                stat = agent.get_statistics()
                d = {stat[self.plot_index][0]: stat[self.plot_index][1]}
                self.plot(step, d)

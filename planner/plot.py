

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# import , Grid, Map, 
from .env import Env, Grid, Map
from .node import Node

class Plot:
    def __init__(self, start, goal, env):
        self.start = Node(start, start, 0, 0)
        self.goal = Node(goal, goal, 0, 0)
        self.env = env
        self.fig = plt.figure("planning")
        self.ax = self.fig.add_subplot()

    def plotEnv(self, name: str) -> None:
        '''
        Plot environment with static obstacles.

        Parameters
        ----------
        name: Algorithm name or some other information
        '''
        plt.plot(self.start.x, self.start.y, marker="s", color="#ff0000")
        plt.plot(self.goal.x, self.goal.y, marker="s", color="#1155cc")

        if isinstance(self.env, Grid):
            obs_x = [x[0] for x in self.env.obstacles]
            obs_y = [x[1] for x in self.env.obstacles]
            plt.plot(obs_x, obs_y, "sk")

        if isinstance(self.env, Map):
            ax = self.fig.add_subplot()
            # boundary
            for (ox, oy, w, h) in self.env.boundary:
                ax.add_patch(patches.Rectangle(
                        (ox, oy), w, h,
                        edgecolor='black',
                        facecolor='black',
                        fill=True
                    )
                )
            # rectangle obstacles
            for (ox, oy, w, h) in self.env.obs_rect:
                ax.add_patch(patches.Rectangle(
                        (ox, oy), w, h,
                        edgecolor='black',
                        facecolor='gray',
                        fill=True
                    )
                )
            # circle obstacles
            for (ox, oy, r) in self.env.obs_circ:
                ax.add_patch(patches.Circle(
                        (ox, oy), r,
                        edgecolor='black',
                        facecolor='gray',
                        fill=True
                    )
                )

        plt.title(name)
        plt.axis("equal")
        # plt.show()

    
    def plot_traj(self, ref, x_traj) -> None:
        '''
        Plot environment with static obstacles.

        Parameters
        ----------
        name: Algorithm name or some other information
        '''
        plt.plot(self.start.x, self.start.y, marker="s", color="#ff0000")
        plt.plot(self.goal.x, self.goal.y, marker="s", color="#1155cc")

        if isinstance(self.env, Grid):
            obs_x = [x[0] for x in self.env.obstacles]
            obs_y = [x[1] for x in self.env.obstacles]
            plt.plot(obs_x, obs_y, "sk")

        if isinstance(self.env, Map):
            ax = self.fig.add_subplot()
            # boundary
            for (ox, oy, w, h) in self.env.boundary:
                ax.add_patch(patches.Rectangle(
                        (ox, oy), w, h,
                        edgecolor='black',
                        facecolor='black',
                        fill=True
                    )
                )
            # rectangle obstacles
            for (ox, oy, w, h) in self.env.obs_rect:
                ax.add_patch(patches.Rectangle(
                        (ox, oy), w, h,
                        edgecolor='black',
                        facecolor='gray',
                        fill=True
                    )
                )
            # circle obstacles
            for (ox, oy, r) in self.env.obs_circ:
                ax.add_patch(patches.Circle(
                        (ox, oy), r,
                        edgecolor='black',
                        facecolor='gray',
                        fill=True
                    )
                )


        plt.plot(x_traj[:,0], x_traj[:, 1], label='Actual Trajectory (x1)')
        plt.plot(ref[:,0], ref[:, 1], 'r--', label='Desired Trajectory (x1)')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Trajectory')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.savefig('traj.png')
        plt.show()

        # plt.title(name)
        plt.axis("equal")
        # plt.show()


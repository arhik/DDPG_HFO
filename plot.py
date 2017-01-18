import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class DDPG_Plot:
    def __init__(self, ax):
        self.ax = ax
        self.ax.set_ylim(-15, 15)
        self.ax.set_xlim(0,10)
        self.line, = ax.plot([],[],'ko', alpha=0.2)
        self.avg_line, = ax.plot([],[], 'r-', alpha=.6)
        self.avg_reward = 0
        self.x = []
        self.y = []
        self.avg_y = []
        plt.draw()

    def init(self):
        self.avg_reward = 0
        self.line.set_data([],[])
        self.avg_line.set_data([],[])
    
    def update(self,i):
        with open('rewards.csv', 'r') as f:
            rows = f.readlines()
            self.x = []
            self.y = []
            self.avg_y = []
            self.avg_reward = 0
            for eachline in rows:
                if len(eachline) > 1:
                    line = eachline.strip('\n')
                    x,y = line.split(',')
                    self.x.append(int(x))
                    self.y.append(float(y))
                    self.avg_reward = sum(self.y[-20:-1])/20 if len(self.y) > 20 else sum(self.y)/len(self.y)
                    self.avg_y.append(self.avg_reward)
                    self.line.set_data(self.x, self.y)
                    self.avg_line.set_data(self.x, self.avg_y)
                    self.ax.set_xlim(0,(1.05)*len(self.x))
                    self.ax.set_ylim(min(self.y)-.5,max(self.y)+.5)

fig, ax = plt.subplots()
ddpg = DDPG_Plot(ax)
ani = animation.FuncAnimation(fig, ddpg.update, interval=1000)
plt.show()


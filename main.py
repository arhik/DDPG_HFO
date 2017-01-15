import hfo
import itertools
from hfo import *
import random
import time
import math

def low_level_reward_function(state_1, state_0, already_close_to_ball, status):
    if not already_close_to_ball:
        I_kick = max(0, (state_1[12] - state_0[12])/2)
        if state_1[12]==1:
            already_close_to_ball = True
    else:
        I_kick = 0
    if status==hfo.GOAL:
        I_goal = 1
    else:
        I_goal = 0

    def ball_goal_dist(s): #distance not proximity
        theta1 = (-1 if s[13] < 0 else 1)*math.acos(s[14])
        theta2 = (-1 if s[51] < 0 else 1)*math.acos(s[52])
        return math.sqrt((1-s[53])**2 + (1-s[15])**2 + 2*(1 - s[53])*(1-s[15])*math.cos(abs(theta1 - theta2)))/math.sqrt((1-s[53])**2 + (1-s[15])**2) # leaving it unnormalized
    
    print("TowardsError: {}\n Kickable: {}\n ball_goal_delta: {}\n bool GOAL:{}".format(
        (state_1[53] - state_0[53]), I_kick, 3*(ball_goal_dist(state_0) - ball_goal_dist(state_1)), 5*I_goal
    ))
    return((state_1[53] - state_0[53]) + I_kick + 3*(ball_goal_dist(state_0) - ball_goal_dist(state_1)) + 5*I_goal)


def main():
    env = hfo.HFOEnvironment()
    env.connectToServer(LOW_LEVEL_FEATURE_SET, config_dir='./', server_port=1234)

    for episode in itertools.count():
        status = env.step()
        s1 = env.getState()
        already_close_to_ball = False
        while status == IN_GAME:
            time.sleep(3)
            env.act(GO_TO_BALL)
            status = env.step()
            s2 = env.getState()
            low_level_reward_function(s2,s1, already_close_to_ball, status)
            s1 = s2
        print("Episode %d ended with")
        if status == SERVER_DOWN:
            env.act(QUIT)
            break

if __name__ == '__main__':
    main()



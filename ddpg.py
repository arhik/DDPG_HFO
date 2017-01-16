import hfo
import numpy as np
import random
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json
import math
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
from collections import namedtuple
import time

OU = OU()       #Ornstein-Uhlenbeck Process

import threading

class Locker(object):
    def __init__(self):
        # self.lock = threading.Lock()
        self._visited = False
        self.count = 0
    
    @property
    def visited(self):
        return self._visited
    
    @visited.setter
    def visited(self, value):
        self._visited = value | self._visited
    
    def step(self):
        self.count +=1
        if self.count > 5:
            self._visited = False
            self.count = 0

    # def release():
    #     self.lock.release()

def dist(x1,y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


    
def high_level_reward_func(state_1, state_0, ball, status):
    agent_x0 = state_0[0]
    agent_y0 = state_0[1]

    agent_x1 = state_1[0]
    agent_y1 = state_1[1]

    ball_x0 = state_0[3]
    ball_y0 = state_0[4]

    ball_x1 = state_1[3]
    ball_y1 = state_1[4]
    if not ball.visited:
        I_kick = state_1[5]
        if state_1[5] == 1:
            ball.visited = True
    else:
        I_kick = 0
    
    if status==hfo.GOAL:
        I_goal = 1
    else:
        I_goal = 0

    return (- dist(agent_x1, agent_y1, ball_x1, ball_y1) + dist(agent_x0, agent_y0, ball_x0, ball_y0))  + \
                1*I_kick + 3*(-dist(ball_x1, ball_y1, 53.00, 0.0) + dist(ball_x0, ball_y0, 53.0, 0.0)) + 5*I_goal


def low_level_reward_function(state_1, state_0, ball, status):
    if not ball.visited:
        I_kick = max(0, (state_1[12] - state_0[12])/2)
        if state_1[12]==1:
            ball.visited = True
    else:
        I_kick = 0
    if status==hfo.GOAL:
        I_goal = 1
    else:
        I_goal = 0

    def ball_goal_dist(s): # distance not proximity
        theta1 = (-1 if s[13] < 0 else 1)*math.acos(s[14])
        theta2 = (-1 if s[51] < 0 else 1)*math.acos(s[52])
        return math.sqrt((1-s[53])**2 + (1-s[15])**2 - 2*(1 - s[53])*(1-s[15])*math.cos(abs(theta1 - theta2)))/math.sqrt((1-s[53])**2 + (1-s[15])**2) # leaving it unnormalized
    
    print("| TowardsError: {}\n| Kickable: {}\n| ball_goal_delta: {}{}\n| bool GOAL:{}".format(
        (state_1[53] - state_0[53]), I_kick, 3*(ball_goal_dist(state_0) - ball_goal_dist(state_1)), "| INCLUDED |" if ball.visited else "| ** IGNORED ** |" ,5*I_goal
    ))
    return((state_1[53] - state_0[53]) + 2*I_kick + int(ball.visited)*5*(ball_goal_dist(state_0) - ball_goal_dist(state_1)) + 7*I_goal)


def invert_grads(g, a):
    tmp_a = np.array(a)
    tmp = np.array(g)
    tmp[tmp >= 0] = tmp[tmp >= 0]*(1- a[tmp >= 0])/2
    tmp[tmp < 0] = tmp[tmp < 0]*(a[tmp < 0] + 1)/2
    return tmp
    
def playGame(train_indicator=0):    # 1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000.
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     # Target Network HyperParameters
    LRA = 0.0005    # Learning rate for Actor
    LRC = 0.001     # Lerning rate for Critic

    action_dim = 10  # 4 actions and their 6 continuous parameters
    state_dim = 58   # of sensors input

    np.random.seed(1337)

    EXPLORE = 100000
    episode_count = 20000
    max_steps = 100000
    reward = 0
    step = 0
    epsilon = 1
    indicator = 0

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a HFO environment
    env = hfo.HFOEnvironment()
    env.connectToServer(hfo.LOW_LEVEL_FEATURE_SET, config_dir='./')

    #Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights("actormodel.h5")
        critic.model.load_weights("criticmodel.h5")
        actor.target_model.load_weights("actormodel.h5")
        critic.target_model.load_weights("criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

        
    print("Soccer Experiment Start.")
    for episode in range(episode_count):
        print("Episode : " + str(episode) + " Replay Buffer " + str(buff.count()))
        isBall = Locker()  
        s_t = np.hstack(env.getState())
        status = env.step()
        total_reward = 0.
        total_target_q_values = 0
        for j in range(max_steps):
            
            # time.sleep(.1)
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            
            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.60 , 0.15, 0.20)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.25 , 0.15, 0.20)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.20, 0.15, 0.20)
            noise_t[0][3] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][3],  0.40 , 0.15, 0.20)
            noise_t[0][4] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][4],  0.0 , 0.15, 0.20)
            noise_t[0][5] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][5],  0.0 , 0.15, 0.20)
            noise_t[0][6] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][6],  0.0 , 0.15, 0.20)
            noise_t[0][7] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][7],  0.0 , 0.15, 0.20)
            noise_t[0][8] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][8],  0.0 , 0.15, 0.20)
            noise_t[0][9] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][9],  0.0 , 0.15, 0.20)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][4] = a_t_original[0][4] + noise_t[0][4]
            a_t[0][5] = a_t_original[0][5] + noise_t[0][5]

            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][6] = a_t_original[0][6] + noise_t[0][6]

            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
            a_t[0][7] = a_t_original[0][7] + noise_t[0][7]

            a_t[0][3] = a_t_original[0][3] + noise_t[0][3]
            a_t[0][8] = a_t_original[0][8] + noise_t[0][8]
            a_t[0][9] = a_t_original[0][9] + noise_t[0][9]
            
            dash_tuple = namedtuple('Dash', ['SOFTMAX', 'PWR','ANGLE'])
            turn_tuple = namedtuple('Turn', ['SOFTMAX', 'ANGLE'])
            tackle_tuple = namedtuple('Tackle', ['SOFTMAX', 'ANGLE'])
            kick_tuple = namedtuple('Kick',['SOFTMAX','PWR','ANGLE'])

            dash = dash_tuple(a_t[0][0], 100*a_t[0][4], 180*a_t[0][5])
            turn = turn_tuple(a_t[0][1], 180*a_t[0][6])
            tackle = tackle_tuple(a_t[0][2], 180*a_t[0][7])
            kick = kick_tuple(a_t[0][3], 100*a_t[0][8], 180*a_t[0][9])

            print("Actions:\n--{}\n--{}\n--{}\n--{}".format(dash, turn, tackle, kick))
            # if 0 <= episode <= 200:
            #     r = .6
            # elif 201 < episode <=500:
            #     r = 0.7
            # elif 501 <= episode < 1000:
            #     r = 0.75
            # elif 1001 <- episode < 2000:
            #     r = .8
            # else:
            #     r = .9
            
            actions = sorted([dash, turn, tackle, kick], key=lambda x: x.SOFTMAX, reverse=True)
            # action = actions[0 if random.random() < r else random.randint(0,3)]
            action = actions[0]
            print(action)
            if type(action) == type(dash):
                env.act(hfo.DASH, dash.PWR, dash.ANGLE)
            elif type(action) == type(turn):
                env.act(hfo.TURN, turn.ANGLE)
            elif type(action) == type(tackle):
                env.act(hfo.TACKLE, tackle.ANGLE)
            elif type(action) == type(kick):
                env.act(hfo.KICK, kick.PWR,kick.ANGLE)
            else:
                print('I am not acting')
            player = env.playerOnBall()
            status = env.step()

            s_t1 = np.array(env.getState())
            r_t =   low_level_reward_function(s_t1, s_t, isBall, status)

            buff.add(s_t, a_t[0], r_t , s_t1, status)
            
            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            
            predicted_actions = actor.target_model.predict(new_states)
            target_q_values = critic.target_model.predict([new_states, predicted_actions])
            
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
            
            for k in range(len(batch)):
                total_target_q_values = total_target_q_values + target_q_values[k]
       
            if (train_indicator):
                loss += critic.model.train_on_batch([states,actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                inverted_grads = invert_grads(grads, a_for_grad) # Invert the gradients if they exceed the parameter max and min values
                actor.train(states, inverted_grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1
            step += 1
            if status != hfo.IN_GAME:
                break
                
            if status == hfo.SERVER_DOWN:
                env.act(hfo.QUIT)
                break
        with open('rewards.csv', 'a') as f:
            f.writelines("{},{}\n".format(episode, total_reward))
        with open('q_values.csv', 'a') as g:
            g.write("{},{}\n".format(episode, sum(total_target_q_values)))
        if np.mod(episode, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(episode) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")
    
    print("Finish.")

if __name__ == "__main__":
    playGame(train_indicator=1)
    
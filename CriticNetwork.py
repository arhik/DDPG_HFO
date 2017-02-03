import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense, Input, merge, Lambda, Activation, ELU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        K.set_session(sess)
        #Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)  
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size,action_dim):
        S = Input(shape=[state_size], name='state_space')
        A = Input(shape=[action_dim],name='action_space')
        w1 = Dense(1024, activation=K.elu)(S)
        a1 = Dense(512, activation=K.elu)(A)
        h1 = Dense(512, activation=K.elu)(w1)
        h2 = merge([h1,a1],mode='sum')
        h3 = Dense(256, activation=K.elu)(h2)
        h4 = Dense(128, activation=K.elu)(h3)
        # V_a = Dense(4, activation='softmax')(h4)
        # V_p = Dense(6, activation='softsign')(h4)
        # V = merge([V_a, V_p], mode='concat')
        O = Dense(1, activation = K.elu)(h4)
        model = Model(input=[S,A],output=O)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        model.summary()
        return model, A, S
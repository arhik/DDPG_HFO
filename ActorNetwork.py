import numpy as np
import math
from keras.initializations import normal
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Merge, ELU, merge
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)   
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size,action_dim):
        S = Input(shape=[state_size])
        h0 = Dense(1024, activation=K.elu)(S)
        h1 = Dense(512, activation=K.elu)(h0)
        h2 = Dense(256, activation=K.elu)(h1)
        h3 = Dense(128, activation=K.elu)(h2)
        action_V = Dense(4, activation = 'softmax')(h3)
        action_P = Dense(6, activation='softsign', init=lambda shape, name: normal(shape, scale=1e-2, name=name))(h3)
        V = merge([action_V, action_P], mode='concat')
        model = Model(input=S,output=V)
        return model, model.trainable_weights, S
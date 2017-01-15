import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, Merge, Lambda, ELU, merge, PReLU, LeakyReLU
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
# from keras.layers.advanced_activations import ELU

# elu = LeakyReLU()

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
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
        print("Now we build the model")
        
        
        # input_state_model = Sequential()
        # in_state_layer = Dense(1024, input_dim=state_size)
        # input_state_model.add(in_state_layer)

        # input_action_model = Sequential()
        # in_action_layer = Dense(512, input_dim=action_dim)
        # input_action_model.add(in_action_layer)

        # input_model = Sequential()
        # input_layer = Merge([input_state_model,input_action_model], mode='concat',concat_axis=1)
        # input_model.add(input_layer)

        # model = Sequential()
        # model.add(input_state_model)
        # model.add(ELU())
        # model.add(Dense(512))
        # model.add(ELU())
        # model.add(Dense(256))
        # model.add(ELU())
        # model.add(Dense(128))
        # model.add(ELU())
        
        # a_model = Sequential()
        # a_model.add(model)
        # a_model.add(Dense(4,activation='softmax'))

        # p_model = Sequential()
        # p_model.add(model)
        # p_model.add(Dense(6, activation='linear'))

        # final_model = Sequential()

        # final_model.add(Merge([a_model,p_model], mode='concat'))



        # final_model = Sequential()
        # final_model.add(merged)
        # model.add(final_model)


        S = Input(shape=[state_size])
        h0 = Dense(1024, activation=K.elu)(S)
        h1 = Dense(512, activation=K.elu)(h0)
        h2 = Dense(256, activation=K.elu)(h1)
        h3 = Dense(128, activation=K.elu)(h2)
        action_V = Dense(4, activation = 'softmax')(h3)
        action_P = Dense(6, activation='linear', init=lambda shape, name: normal(shape, scale=1e-2, name=name))(h3)
        # Dash_power = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-2, name=name))(h3)  
        # Dash_degree = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-2, name=name))(h3)
        # Turn_degree = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-2, name=name))(h3)
        # Tackle_degree = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-2, name=name))(h3)
        # Kick_degree = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-2, name=name))(h3)
        # Kick_power = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-2, name=name))(h3)
        # V = merge([action_V, Dash_power, Dash_degree, Turn_degree, Tackle_degree, Kick_power, Kick_degree],mode='concat')
        V = merge([action_V, action_P], mode='concat')
        model = Model(input=S,output=V)
        # final_model.compile(optimizer='adam',
        #       loss='mse')
        # final_model.summary()
        # return final_model, final_model.trainable_weights, in_state_layer
        return model, model.trainable_weights, S
import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import model_from_json, load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation, ELU, Merge
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

# from keras.layers.advanced_activations import ELU

elu = ELU()
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
        print("Now we build the model")
        S = Input(shape=[state_size])  
        A = Input(shape=[action_dim],name='action2')
        # L = merge([S,A], mode='concat')
        w1 = Dense(1024)(S)
        a1 = Dense(512, activation=K.elu)(A)
        h1 = Dense(512, activation=K.elu)(w1)
        h2 = merge([h1,a1],mode='sum')
        h3 = Dense(256, activation=K.elu)(h2)
        h4 = Dense(128, activation=K.elu)(h3)
        V = Dense(action_dim,activation=K.elu)(h4)
        model = Model(input=[S,A],output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        model.summary()
        # input_state_model = Sequential()
        # in_state_layer = Dense(1024, input_dim=state_size)
        # input_state_model.add(in_state_layer)
        # input_state_model.add(Dense(512))

        # input_action_model = Sequential()
        # in_action_layer = Dense(512, input_dim=action_dim)
        # input_action_model.add(in_action_layer)

        # input_model = Sequential()
        # input_layer = Merge([input_state_model,input_action_model], mode='concat',concat_axis=1)
        # input_model.add(input_layer)

        # model = Sequential()
        # model.add(input_model)
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
        # final_model.compile(optimizer='adam',
        #       loss='mse')

        # return final_model, input_action_model, input_state_model
        return model, A, S
# This report on capstone project

# Title
## Actor Critic based Deep Deterministic Policy Gradient approach to Soccer World game under partially observed environment

### Introduction:

Since the  advent of AI I always strived to work on algorithms which are more
promising and more scalable and possibly better than biological versions. 
This project is an attempt to understand current approaches of reinforcement
learning algorithms and weigh their cons and pros. In this project I was primarily
focusing on the case where the action space which is larger unlike the toy 
version of self driving car in the Udacity Nano-degree program. The course 
project made me think about its limitations and searched for better version to 
solve robotics learning case. While hunting for current research on such 
situations I came across papers from google deepmind, blogs on it, and 
youtube videos. All these game me an insight to tackle it and starting 
working on it. Though half way through I figured out I will end up working 
on the simulator with complex math for scoring right moves. Instead I 
happen to choose an already existing simulator HFO (Half Field Offense) 
soccer simulator. This environment has scoring mechanism and other features 
encoded for the agent to learn by partial observation. Also the action space is 
too large for the Q-learning approach. This is a report on how DDPG algorithm 
works and how it did in this Environment. Though there is a work done already 
done on I will explore if there is a better way to deal with this


### Approach:

The approach used in this project is Deep Deterministic Policy Gradient Algorithm.
This algorithms helps us deal with the situation where the action space is too large
for lookup. The neural networks are good for encoding large statespace by approximating
and capturing the underlying statespace using weight parameters and could generalize well
to unseen statespaces. This made me choose the Deep  Neural Networks to tackle this 
problem. Among the number of frameworks to choose from I preferred Keras for its backend 
agnostic nature and ability to take advantage of GPU. The neural network architecture
consists of actor and critic networks which play a role in adjusting weights of the 
neural network to better model the partially observable world. The Actor network acts on the
network and the critic helps actor to choose better actions for better performance.


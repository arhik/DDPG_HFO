
# Title
## Actor Critic based Deep Deterministic Policy Gradient approach to Soccer World game under partially observed environment
****
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



![Rewards per each iteration](../images/figure_1-41.png)

#checking math latex
$$ c^2 = a^2 + b^2 $$
$$ \delta^2 + \gamma^2 = 1 $$
$$ a_1 + a_2 = 0$$

$$ d_g + d_b + `3*d_{bg}` $$

### Reward function:

#checking code latex
```{python}
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
        ball_dist = 1 - s[53]
        goal_dist = 1 - s[15]
        return math.sqrt(ball_dist**2 + goal_dist**2 - \
                    2*ball_dist*goal_dist* \
                    math.cos(max(theta1, theta2)- min(theta1,theta2)))
    
    print("| TowardsError: {}\n \
           | Kickable: {}\n\
           | \ball_goal_delta: {}{}\n\
           | bool GOAL:{}".\
           format(
                (state_1[53] - state_0[53]),\
                I_kick, 3*(ball_goal_dist(state_0) - ball_goal_dist(state_1)),\
                "| INCLUDED |" if ball.visited else "| ** IGNORED ** |" ,\
                5*I_goal))
    return((state_1[53] - state_0[53]) + \
        I_kick + int(ball.visited)*3*(ball_goal_dist(state_0) - ball_goal_dist(state_1)) \ +
        5*I_goal)

```
### Approach:

The approach used in this project is Deep Deterministic Policy Gradient Algorithm.
This algorithms helps us deal with the situation where the action space is 
too large for lookup. The neural networks are good for encoding large statespace
by approximating and capturing the underlying statespace using weight parameters
and could generalize well to unseen statespaces. This made me choose the Deep
Neural Networks to tackle this problem. Among the number of frameworks to choose
from I preferred Keras for its backend agnostic nature and ability to take advantage
of GPU. The neural network architecture consists of actor and critic networks 
which play a role in adjusting weights of the neural network to better model 
the partially observable world. The Actor network acts on the network and the 
critic helps actor to choose better actions for better performance.


### Variations and results:

Different architectures tried
results
Why?

# Just a template to make tables

Possible reasons
What could have worked more.

trials | frames per trial | score
---------|----------|---------
 A1 | B1 | C1
 A2 | B2 | C2
 A3 | B3 | C3


### Citations:

1. Tweaked Reward function based on https://github.com/mhauskn/dqn-hfo/blob/master/src/hfo_game.cpp while trying other weights.
But the weight choice made in the papers is good.

### Remarks:
2. The paper used for this project.

3. Numenta SOC concept to stop

4. The AI concepts in generalize

5. The cons of current approach.

6. The DDPG alternatives.

7. should investigate on s[0] and its impact of sign on this.

8. e-greedy may not work for the firt architecture, uhlenbeck and combination of annealing works in this case. Or the exponentially decreasing exploration would equally help.

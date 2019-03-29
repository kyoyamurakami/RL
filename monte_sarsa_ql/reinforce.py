import numpy as np
import random
import matplotlib.pyplot as plt

def graph_plot(xlist, ylist):
    plt.plot(xlist, ylist)
    plt.xlabel("the number of games")
    plt.ylabel("rate of win")


def check(s):
    pos = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
    pos = np.array(pos)
    for i in range(pos.shape[0]):
        val = np.prod(s[0, pos[i, :]])
        if val == 1:
            return 1
        elif val == 2 ** 3:
            return 2
    if np.prod(s): return 3
    return 0


def train(policy, step, state3):
    p = np.copy(policy)
    if (step == 0):
        a = 0
    else:
        while (True):
            p = p / np.sum(p)

            a = np.random.choice(np.size(policy), p=p[0, :])
            if state3[0, a] == 0:
                break
            p[0, a] = 0
    action = a
    state3[0, a] = 2
    fin = check(state3)
    if fin == 2:
        reward = 10
        return action, reward, state3, fin
    elif fin == 3:
        reward = 0
        return action, reward, state3, fin

    reach = 0
    pos = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
    pos = np.array(pos)
    for i in range(pos.shape[0]):
        val = sum(state3[0, pos[i, :]])
        num = np.shape(np.where(state3[0, pos[i, :]] == 0))[1]

        if val == 2 and num == 1:
            a = pos[i, state3[0,pos[i, :]] == 0] # review
            reach = 1
            break
    if reach == 0:
        while (True):
            a = int(random.random() * 10)
            if a == 9:
                continue
            if state3[0, a] == 0:
                break

    state3[0, a] = 1

    fin = check(state3)
    if fin == 1:
        reward = -10
        return action, reward, state3, fin
    elif fin == 0:
        reward = 0
        return action, reward, state3, fin


def conv(state3):
    con = [[0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 1, 0, 5, 4, 3, 8, 7, 6], [6, 3, 0, 7, 4, 1, 8, 5, 2],
           [0, 3, 6, 1, 4, 7, 2, 5, 8], [8, 7, 6, 5, 4, 3, 2, 1, 0], [6, 7, 8, 3, 4, 5, 0, 1, 2],
           [2, 5, 8, 1, 4, 7, 0, 3, 6], [8, 5, 2, 7, 4, 1, 6, 3, 0]]
    convert = np.array(con)

    con3_10 = [3 ** 8, 3 ** 7, 3 ** 6, 3 ** 5, 3 ** 4, 3 ** 3, 3 ** 2, 3 ** 1, 3 ** 0]
    convert3_10 = np.array(con3_10)

    candidates = np.zeros(8)

    for i in range(8):
        candidates[i] = sum(state3[0, convert[i, :]] * convert3_10)

    state = min(candidates)

    return state

def policy_select(self,  state, Q):
    policy = np.zeros((1, self.actions))
    if self.mode == 1:
        a = np.argmax(Q[state, :])
        policy[0, a] = 1
        return policy

    elif self.mode == 2:
        a = np.argmax(Q[state, :])
        policy = np.ones((1, self.actions)) * self.epsilon / self.actions
        policy[0, a] = 1 - self.epsilon + self.epsilon / self.actions
        return policy

    else:
        policy = np.exp(Q[state, :] / self.softmax) / sum(np.exp(Q[state, :] / self.softmax))
        return policy



def display_rate(Q, count, results, l):
    Q = Q / count
    win = len(np.where(results[0, :] == 2)[0])
    lose = len(np.where(results[0, :] == 1)[0])
    draw = len(np.where(results[0, :] == 3)[0])
    print((l + 1) * 100, "win = ", win, "lose", lose, "draw", draw)
    return win/ (win+lose+draw), (l+1) * 100


class MonteCarloPolicy():

    def __init__(self, T=5, states=3 ** 9, actions=9, mode=2, gamma=0.9, softmax=0.5, epsilon=0.1):
        self.states = states
        self.actions = actions
        self.T = T
        self.mode = mode
        self.gamma = gamma
        self.softmax = softmax
        self.epsilon = epsilon
        random.seed(0)

    def update_q(self, M, states, actions, drewards):
        Q = np.zeros((self.states, self.actions))

        for m in range(M):
            for t in range(np.shape(states)[1]):
                s = int(states[m, t])
                a = int(actions[m, t])
                if s == 0:
                    continue
                Q[s, a] = Q[s, a] + drewards[m, t]
        return Q

    def learn(self, L=10, M=1000):
        states = np.zeros((M, self.T))
        actions = np.zeros((M, self.T))
        rewards = np.zeros((M, self.T))
        drewards = np.zeros((M, self.T))
        xlist = []
        ylist = []
        Q = np.zeros((self.states, self.actions))
        for l in range(L):
            count = np.ones((self.states, self.actions))
            results = np.zeros((1, M))

            for m in range(M):
                state3 = np.zeros((1, 9))

                for t in range(self.T):
                    state = int(conv(state3))
                    policy = policy_select(self, state, Q)
                    action, reward, state3, fin = train(policy, t, state3)

                    states[m, t] = state
                    actions[m, t] = action
                    rewards[m, t] = reward
                    count[state, action] = count[state, action] + 1

                    if fin > 0:
                        results[0, m] = fin

                        drewards[m, t] = rewards[m, t]
                        for pstep in range(t-1,-1,-1):
                            drewards[m, pstep] = self.gamma * drewards[m, pstep + 1]

                        break


            Q = self.update_q(M, states, actions,drewards)

            rate, num = display_rate(Q, count, results, l)
            xlist.append(num)
            ylist.append(rate)

        graph_plot(xlist, ylist)

class SARSA():

    def __init__(self, T=5, states=3 ** 9, actions=9, mode=2, gamma=0.9, softmax=0.5, epsilon=0.1, alpha=1):
        self.states = states
        self.actions = actions
        self.T = T
        self.mode = mode
        self.gamma = gamma
        self.softmax = softmax
        self.epsilon = epsilon
        self.alpha = alpha
        random.seed(0)

    def update_q(self, newQ, pstate, paction, state, action, reward):
        newQ[pstate,paction] = newQ[pstate, paction] + self.alpha * (reward - newQ[pstate, paction] + self.gamma * newQ[state, action])
        return newQ



    def learn(self,  L=10, M=1000):
        xlist = []
        ylist = []
        Q = np.zeros((self.states, self.actions))
        for l in range(L):
            count = np.ones((self.states, self.actions))
            results = np.zeros((1, M))
            newQ = np.zeros((self.states, self.actions))
            for m in range(M):
                state3 = np.zeros((1, 9))
                for t in range(self.T):
                    state = int(conv(state3))
                    policy = policy_select(self, state, Q)
                    action, reward, state3, fin = train(policy, t, state3)

                    if t > 0:
                        newQ = self.update_q(newQ, pstate, paction, state, action, reward)

                    if fin >0:
                        results[0,m] = fin
                        break

                    pstate = state
                    paction = action
            Q = np.copy(newQ)


            rate, num = display_rate(Q, count, results, l)
            xlist.append(num)
            ylist.append(rate)

        graph_plot(xlist, ylist)

class Qlearn():

    def __init__(self, T=5, states=3 ** 9, actions=9, mode=2, gamma=0.9, softmax=0.5, epsilon=0.1, alpha=1):
        self.states = states
        self.actions = actions
        self.T = T
        self.mode = mode
        self.gamma = gamma
        self.softmax = softmax
        self.epsilon = epsilon
        self.alpha = alpha
        random.seed(0)

    def update_q(self, Q, pstate, paction, state, action, reward):
        Q[pstate, paction] = Q[pstate, paction] + self.alpha * (
                    reward - Q[pstate, paction] + self.gamma * np.amax(Q[state, :]))
        return Q

    def learn(self, L=10, M=1000):
        xlist = []
        ylist = []
        Q = np.zeros((self.states, self.actions))
        for l in range(L):
            count = np.ones((self.states, self.actions))
            results = np.zeros((1, M))
            for m in range(M):
                state3 = np.zeros((1, 9))
                for t in range(self.T):
                    state = int(conv(state3))
                    policy = policy_select(self, state, Q)
                    action, reward, state3, fin = train(policy, t, state3)

                    if t > 0:
                        Q = self.update_q(Q, pstate, paction, state, action, reward)

                    if fin > 0:
                        results[0, m] = fin
                        break

                    pstate = state
                    paction = action

            rate, num = display_rate(Q, count, results, l)
            xlist.append(num)
            ylist.append(rate)

        graph_plot(xlist, ylist)


if __name__ ==  "__main__":
    monte = MonteCarloPolicy()
    monte.learn()
    sarsa = SARSA()
    sarsa.learn()
    ql = Qlearn()
    ql.learn()
    plt.show()

import numpy as np
import math
import random

def check(s):
	pos = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
	pos = np.array(pos)
	for i in range(pos.shape[0]):
		val = np.prod(s[0,pos[i,:]])
		if val == 1: return 1
		elif val == 2**3: return 2
	if np.prod(s): return 3
	return 0
	
	

def train(policy, step, state3):
	reward = 0
	random.seed(0)
	p = np.copy(policy)
	if(step == 0):
		a=0
	else:
		while(True):
			p = p / np.sum(p)
			a = np.random.choice(np.size(policy),p=p[0,:])
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
	pos = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
	pos = np.array(pos)
	for i in range(pos.shape[0]):
		val = sum(state3[0,pos[i,:]])
		num = np.shape(np.where(state3[0,pos[i,:]]))[1]
		if val==2 and num == 1:
			a = pos[i, state3[pos[i,:]==0] 
			reach = 1
			break
	if reach == 0:
		while(True)
			a = int(random.random()*10)
			if a == 10:
				continue
			if state3[0,a] == 0:
				break
	
	state3[0,a] = 1
	
	fin = check(state3)
	if fin == 1:
		reward = -10
		return action, reward, state3, fin
	elif fin == 0:
		reward = 0
		return action, reward, state3, fin





def conv(state3):
	con = [[0,1,2,3,4,5,6,7,8],[2,1,0,5,4,3,8,7,6],[6,3,0,7,4,1,8,5,2],[0,3,6,1,4,7,2,5,8],[8,7,6,5,4,3,2,1,0],[6,7,8,3,4,5,0,1,2],[2,5,8,1,4,7,0,3,6],[8,5,2,7,4,1,6,3,0]]
	convert = np.array(con)
	
	con3_10 = [3**8,3**7,3**6,3**5,3**4,3**3,3**2,3**1,3**0]
	convert3_10 = np.array(con3_10)
	
	candidates = np.zeros(8)
	
	for i in range(8):
		candidates[i] = sum(state3[0,convert[i,:]] * convert3_10)
		
	state = min(candidates)
	
	return state
	
class MonteCarloPolicy():
	 
	def __init__(self, T=5,states = 3**9, actions = 9, mode = 1,gamma = 0.9, softmax = 0.5, epsilon = 0.1):
		self.states = states
		self.actions = actions
		self.T = 5
		self.Q = np.zeros((states, actions))
		self.mode = mode
		self.gamma = gamma
		self.softmax = softmax
		self.epsilon = epsilon
		
	def __policy_select(self, mode, state):
		policy = np.zeros((1,self.actions))
		if mode == 1:
			a = np.argmax(self.Q[state, :])
			policy[0,a] = 1
			return policy
			
		elif mode == 2:
			a = np.argmax(self.Q[state, :])
			policy = np.ones((1, self.actions)) * self.epsilon/self.actions
			policy[0,a] = 1-self.epsilon + self.epsilon/self.actions
			return policy
			
		else:
			policy = np.exp(self.Q[state, :]/ self.softmax)/sum(np.exp(self.Q[state, :] / self.softmax))
			return policy
			
		
		
	def learn(self, L = 10, M =1000):
		 
		 for l in range(L):
		 	count = np.ones((self.states, self.actions))
		 	results = np.zeros((M, 1))
		 	
		 	for m in range(M):
		 		state3 = np.zeros((1, 9))
		 		
		 		for t in range(self.T):
		 			state = int(conv(state3))
		 			policy = self.__policy_select(self.mode, state)
		 			action, reward, state3, fin = train(policy, t, state3)
		 			
		 			
	

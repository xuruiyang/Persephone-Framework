from abc import ABC, abstractmethod
import copy
import numpy as np

class TransitionSystem(ABC):

	@abstractmethod
	def eval(self, action):
		...

	def next(self, action):
		nextState = copy.deepcopy(self)
		nextState.action = action
		return nextState

	@abstractmethod
	def getStateRep(self):
		...

	@abstractmethod
	def f2str(self):
		...

	@abstractmethod
	def getActions(self):
		...


class HSRTranSys(TransitionSystem):

	def __init__(self, k=2, q=2, n=4):
		self.k = k
		self.q = q
		self.n = n
		self.m = -1
		self.action = None

	def eval(self):
		# breakpoint()
		from persephone.PersephoneMu import Variable
		if self.action is None:
			return self
		else:
			assert isinstance(self.action, Variable)
		action = self.action.eval()
		self.action = None
		if self.m==-1:
			self.m = action
		else:
			if action==0:
				self.k -= 1
				self.q -= 1
				self.n = self.m
			elif action==1:
				self.q -= 1
				self.n = self.n - self.m
			else:
				assert False
			self.m = -1
		return self

	def getStateRep(self):
		return [self.k, self.q, self.n, self.m]

	def f2str(self):
		if self.action is None:
			return str((self.k, self.q, self.n, self.m))
		else:
			return str((self.k, self.q, self.n, self.m))+".next"
		# if self.action is None:
		# 	return str(self)
		# else:
		# 	return str(self)+".next"

	def getActions(self):
		from persephone.PersephoneMu import Domain
		if self.m==-1:
			if self.n>1 and self.k>0 and self.q>0:
				return Domain(1,self.n)
			else:
				return Domain(1,0)
		else:
			return Domain(0,2)
		

class DinePhiTranSys(TransitionSystem):

	def __init__(self, n=2):
		self.fork = [1 for _ in range(n)]
		self.dine_phi = [DinePhi(pid, self.fork, n) for pid in range(n)]
		self.active_pid = 0
		self.action = None
		self.n = n

	def eval(self):
		from persephone.PersephoneMu import Variable
		if self.action is None:
			return self
		else:
			assert isinstance(self.action, Variable)
		action = self.action.eval()
		self.action = None
		self.dine_phi[action].eval(action)
		self.active_pid = action
		return self

	def f2str(self):
		if self.action is None:
			return str(self.getStateRep())
		else:
			return str(self.getStateRep())+".next"

	def getStateRep(self):
		# return [self.active_pid]+[p.current_state for p in self.dine_phi]+self.fork[:]
		return [p.current_state for p in self.dine_phi]+self.fork[:]
		# return [self.active_pid]+[p.current_state for p in self.dine_phi]

	def getActions(self):
		return self.dine_phi[self.active_pid].getActions()

	def someone_hungry(self):
		ret = False
		for p in self.dine_phi:
			ret = ret or (p.current_state>0 and p.current_state<7)
		return 1 if ret else 0

	def nobody_hungry(self):
		return 0 if self.someone_hungry() else 1

	def someone_eat(self):
		ret = False
		for p in self.dine_phi:
			ret = ret or (p.current_state==7)
		return 1 if ret else 0

	def phi0_hungry(self):
		ret = False
		p=self.dine_phi[0]
		ret = ret or (p.current_state>0 and p.current_state<7)
		return 1 if ret else 0

	def phi0_not_hungry(self):
		return 0 if self.phi0_hungry() else 1

	def phi0_eat(self):
		ret = False
		p=self.dine_phi[0]
		ret = ret or (p.current_state==7)
		return 1 if ret else 0


class DinePhi():

	def __init__(self, pid, fork, n):
		self.pid = pid
		self.current_state = 0
		self.fork = fork
		self.n = n
		self.has_fork = 0

	def getActions(self):
		from persephone.PersephoneMu import Domain
		return Domain(0,self.n)

	def eval(self, action):
		# no fair required
		n = self.n
		if self.current_state == 0:
			self.current_state = 1 if np.random.rand()>0.5 else 2
		elif self.current_state == 1:
			if self.fork[(self.pid-1)%n]:
				self.current_state = 3
				self.fork[(self.pid-1)%n] = 0
				self.has_fork += 1
		elif self.current_state == 2:
			if self.fork[self.pid]:
				self.current_state = 4
				self.fork[self.pid] = 0
				self.has_fork += 1
		elif self.current_state == 3:
			if self.fork[self.pid%n]:
				self.current_state = 7
				self.fork[self.pid] = 0
				self.has_fork += 1
			else: 
				self.current_state = 5
		elif self.current_state == 4:
			if self.fork[(self.pid-1)%n]:
				self.current_state = 7 
				self.fork[(self.pid-1)%n] = 0
				self.has_fork += 1 
			else: 
				self.current_state = 6
		elif self.current_state == 5:
			self.current_state = 0
			self.fork[(self.pid-1)%n] = 1
			self.has_fork -= 1
		elif self.current_state == 6:
			self.current_state = 0
			self.fork[self.pid] = 1
			self.has_fork -= 1
		elif self.current_state == 7:
			if np.random.rand()>0.5:
				self.current_state = 8
				self.fork[(self.pid-1)%n] = 1
			else: 
				self.current_state = 9
				self.fork[self.pid] = 1
			self.has_fork -= 1
		elif self.current_state == 8:
			self.current_state = 0
			self.fork[self.pid] = 1
			self.has_fork -= 1
		elif self.current_state == 9:
			self.current_state = 0
			self.fork[(self.pid-1)%n] = 1
			self.has_fork -= 1
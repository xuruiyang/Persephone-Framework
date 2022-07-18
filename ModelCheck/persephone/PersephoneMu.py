from abc import ABC, abstractmethod
from typing import List
from typing import Union
from ts.TransitionSystem import TransitionSystem
import conf

P_ID = 1
P_PARAMS = conf.NUM_P*3
F_VARS = 5
S_TOTAL = P_ID+P_PARAMS+F_VARS

class Formula(ABC):
	@property
	def f_type(self) -> str:
		return self._f_type
	@f_type.setter
	def f_type(self, val: str):
		self._f_type = val

	@abstractmethod
	def f2str(self,expand=False):
		...

	# @abstractmethod
	# def count_operators(self):
	# 	...

	@abstractmethod
	def getActions(self):
		...

	@abstractmethod
	def move(self,act,player1,player2):
		...

	@abstractmethod
	def getRole(self):
		...

	@abstractmethod
	def traverseFormula(self, preorder):
		...

class TrueConst(Formula):

	def __init__(self):
		self._f_type = 't'
		self.preorder = -1
		self.paramVec = [-1]*P_PARAMS
		self.pidVec = [-1]*P_ID
		self.varVec = [-1]*F_VARS

	def f2str(self,expand=False):
		return 'True'

	def getActions(self):
		return None

	def move(self,act,player1,player2):
		return None

	def getRole(self):
		return None

	def getStateRep(self):
		return self.pidVec+self.paramVec+self.varVec

	def traverseFormula(self, preorder):
		self.preorder = preorder
		return preorder

class FalseConst(Formula):

	def __init__(self):
		self._f_type = 'f'
		self.preorder = -1
		self.paramVec = [-1]*P_PARAMS
		self.pidVec = [-1]*P_ID
		self.varVec = [-1]*F_VARS

	def f2str(self,expand=False):
		return 'False'

	def getActions(self):
		return None

	def move(self,act,player1,player2):
		return None

	def getRole(self):
		return None

	def getStateRep(self):
		return self.pidVec+self.paramVec+self.varVec

	def traverseFormula(self, preorder):
		self.preorder = preorder
		return preorder

class Algebra(ABC):
	@abstractmethod
	def eval(self):
		...

class Variable(Formula):
	@property
	def symbol(self) -> str:
		return self._symbol
	@symbol.setter
	def symbol(self, val: str):
		self._symbol = val

	@property
	def isFree(self) -> bool:
		return self._isFree
	@isFree.setter
	def isFree(self, val: bool):
		self._isFree = val

	@property
	def value(self) -> int:
		return self._value
	@value.setter
	def value(self, val: int):
		self._value = val

	def __init__(self,s):
		self._symbol = s
		self._isFree = True
		self._value = None
		self._f_type = 'var'
		self.preorder = -1

	def f2str(self,expand=False):
		return self.symbol if self._value is None else str(self._value)

	def count_operators(self):
		return 0

	def getActions(self):
		return []

	def move(self,act,player1,player2):
		return None

	def eval(self):
		return self._value if self._value is not None else self.symbol

	def getRole(self):
		return None

	def traverseFormula(self, preorder):
		self.preorder = preorder
		return preorder

class ITE(Formula):
	@property
	def i_f_type(self) -> Union[Algebra,int]:
		return self._i_f_type
	@i_f_type.setter
	def i_f_type(self, val: Union[Algebra,int]):
		self._i_f_type = val

	@property
	def then_form(self) -> Union[Formula,int]:
		return self._then_form
	@then_form.setter
	def then_form(self, val: Union[Formula,int]):
		self._then_form = val

	@property
	def else_form(self) -> Union[Formula,int]:
		return self._else_form
	@else_form.setter
	def else_form(self, val: Union[Formula,int]):
		self._else_form = val

	def __init__(self, iff:Union[Algebra,int] , thenf:Union[Formula,int], elsef:Union[Formula,int]):
		self._i_f_type = iff
		self._then_form = thenf
		self._else_form = elsef
		self._f_type = 'ite'
		self.preorder = -1
		self.paramVec = [-1]*P_PARAMS
		self.pidVec = [-1]*P_ID
		self.varVec = [-1]*F_VARS


	def f2str(self,expand=False):
		return '(if '+self._i_f_type.f2str()+' then '+self._then_form.f2str()+' else '+self._else_form.f2str()+')'

	def getActions(self):
		return []

	def move(self,act,player1,player2):
		# assert(self.preorder>-1 and self.preorder<=F_VARS)
		self._then_form.varVec = self.varVec[:] if self._then_form._f_type!='pred' else self._then_form.varVec
		self._then_form.pidVec = self.pidVec[:] if self._then_form._f_type!='pred' else self._then_form.pidVec
		self._then_form.paramVec = self.paramVec[:] if self._then_form._f_type!='pred' else self._then_form.paramVec
		self._else_form.varVec = self.varVec[:] if self._else_form._f_type!='pred' else self._else_form.varVec
		self._else_form.pidVec = self.pidVec[:] if self._else_form._f_type!='pred' else self._else_form.pidVec
		self._else_form.paramVec = self.paramVec[:] if self._else_form._f_type!='pred' else self._else_form.paramVec
		# self._then_form.varVec[self.preorder] = 1
		# self._else_form.varVec[self.preorder] = 0
		return self._then_form if self._i_f_type.eval()==1 else self._else_form

	def getRole(self):
		return None

	def getStateRep(self):
		return self.pidVec+self.paramVec+self.varVec

	def traverseFormula(self, preorder):
		# transparent to game
		# if self._i_f_type.eval()==1:
		preorder1 = self._then_form.traverseFormula(preorder)
		# else:
		preorder2 = self._else_form.traverseFormula(preorder)
		return max(preorder1, preorder2)

class BinOP(Formula):
	@property
	def left_form(self) -> Union[Formula,int]:
		return self._left_form
	@left_form.setter
	def left_form(self, val: Union[Formula,int]):
		self._left_form = val

	@property
	def right_form(self) -> Union[Formula,int]:
		return self._right_form
	@right_form.setter
	def right_form(self, val: Union[Formula,int]):
		self._right_form = val

	@abstractmethod
	def __init__():
		...

	def f2str(self,expand=False):
		return '('+self._left_form.f2str()+' '+self._f_type+' '+self._right_form.f2str()+')'

	def count_operators(self):
		return 1+self.left_form.count_operators()+self.right_form.count_operators()

	def getActions(self):
		return [0,1]

	def move(self,act,player1,player2):
		assert(self.preorder>-1 and self.preorder<=F_VARS)
		self._left_form.varVec = self.varVec[:] if self._left_form._f_type!='pred' else self._left_form.varVec
		self._left_form.pidVec = self.pidVec[:] if self._left_form._f_type!='pred' else self._left_form.pidVec
		self._left_form.paramVec = self.paramVec[:] if self._left_form._f_type!='pred' else self._left_form.paramVec
		self._right_form.varVec = self.varVec[:] if self._right_form._f_type!='pred' else self._right_form.varVec
		self._right_form.pidVec = self.pidVec[:] if self._right_form._f_type!='pred' else self._right_form.pidVec
		self._right_form.paramVec = self.paramVec[:] if self._right_form._f_type!='pred' else self._right_form.paramVec
		if act == 0:
			if self._left_form._f_type!='pred':
				self._left_form.varVec[self.preorder] = 0
			return self._left_form
		elif act == 1:
			if self._right_form._f_type!='pred':
				self._right_form.varVec[self.preorder] = 1
			return self._right_form
		else:
			print('illegal action',act,type(act))
			assert False

	def traverseFormula(self, preorder):
		self.preorder = preorder
		preorder+=1
		preorder = self._left_form.traverseFormula(preorder)
		preorder+=1
		preorder = self._right_form.traverseFormula(preorder)
		return preorder

class UnaOP(Formula):
	@property
	def sub_form(self) -> Formula:
		return self._sub_form
	@sub_form.setter
	def sub_form(self, val: Formula):
		self._sub_form = val

	@abstractmethod
	def __init__():
		...

	def f2str(self,expand=False):
		return '('+self._f_type+' '+self._sub_form.f2str()+')'

	def count_operators(self):
		return 1+self.sub_form.count_operators()

	def getActions(self):
		return []

	def move(self,act,player1,player2):
		self._sub_form.varVec[self.preorder] = 0
		return self._sub_form

	def traverseFormula(self, preorder):
		self.preorder = preorder
		preorder+=1
		preorder = self._sub_form.traverseFormula(preorder)
		return preorder

class Domain():
	@property
	def low(self) -> Union[int,Variable,Algebra]:
		return self._low
	@low.setter
	def low(self, val: Union[int,Variable,Algebra]):
		self._low = val

	@property
	def high(self) -> Union[int,Variable,Algebra]:
		return self._high
	@high.setter
	def high(self, val: Union[int,Variable,Algebra]):
		self._high = val

	def __init__(self,low:Union[int,Variable,Algebra],high:Union[int,Variable,Algebra]):
		self._low=low
		self._high=high

	def toList(self):
		l_int=isinstance(self._low,int)
		h_int=isinstance(self._high,int)
		l_val = self._low.eval() if not l_int else self._low
		h_val = self._high.eval() if not h_int else self._high
		return list(range(l_val,h_val))

	def toStr(self):
		l_int=isinstance(self._low,int)
		h_int=isinstance(self._high,int)
		l_str = self._low.f2str() if not l_int else str(self._low)
		h_str = self._high.f2str() if not h_int else str(self._high)
		return '['+l_str+','+h_str+']'

class Quant(UnaOP):
	@property
	def var(self) -> Variable:
		return self._var
	@var.setter
	def var(self, val: Variable):
		self._var = val

	@property
	def domain(self) -> Domain:
		return self._domain
	@domain.setter
	def domain(self, val: Domain):
		self._domain = val

	@abstractmethod
	def __init__():
		...

	def f2str(self,expand=False):
		return '('+self._f_type+' '+self._var.f2str()+' in '+self._domain.toStr()+': '+self._sub_form.f2str()+')'

	def getActions(self):
		return self._domain.toList()

	def move(self,act,player1,player2):
		if act in [x for x in self._domain.toList()]:
			self._var._value = int(act)
			self._sub_form.varVec = self.varVec[:] if self._sub_form._f_type!='pred' else self._sub_form.varVec
			self._sub_form.pidVec = self.pidVec[:] if self._sub_form._f_type!='pred' else self._sub_form.pidVec
			self._sub_form.paramVec = self.paramVec[:] if self._sub_form._f_type!='pred' else self._sub_form.paramVec
			if  self._sub_form._f_type!='pred' and self._sub_form._f_type!='not':
				self._sub_form.varVec[self.preorder] = int(act)
			return self._sub_form
		else:
			# print("illegal action")
			return FalseConst() if self.getRole()==1 else TrueConst()

class Exists(Quant):
	def __init__(self, v:Variable, dom:Domain, f:Formula):
		self._sub_form = f
		self._var = v
		self._f_type = 'e'
		self._domain = dom
		self.preorder = -1
		self.paramVec = [-1]*P_PARAMS
		self.pidVec = [-1]*P_ID
		self.varVec = [-1]*F_VARS
		# print("Exists",self._var)

	def getRole(self):
		return 1

	def getStateRep(self):
		return self.pidVec+self.paramVec+self.varVec

class Forall(Quant):
	def __init__(self, v:Variable, dom:Domain, f:Formula):
		self._sub_form = f
		self._var = v
		self._f_type = 'a'
		self._domain = dom
		self.preorder = -1
		self.paramVec = [-1]*P_PARAMS
		self.pidVec = [-1]*P_ID
		self.varVec = [-1]*F_VARS

	def getRole(self):
		return -1

	def getStateRep(self):
		return self.pidVec+self.paramVec+self.varVec

class And(BinOP):
	def __init__(self, lf:Formula, rf:Formula):
		self._left_form = lf
		self._right_form = rf
		self._f_type = 'and'
		self.preorder = -1
		self.paramVec = [-1]*P_PARAMS
		self.pidVec = [-1]*P_ID
		self.varVec = [-1]*F_VARS

	def getRole(self):
		return -1

	def getStateRep(self):
		return self.pidVec+self.paramVec+self.varVec

class Or(BinOP):
	def __init__(self, lf:Formula, rf:Formula):
		self._left_form = lf
		self._right_form = rf
		self._f_type = 'or'
		self.preorder = -1
		self.paramVec = [-1]*P_PARAMS
		self.pidVec = [-1]*P_ID
		self.varVec = [-1]*F_VARS

	def getRole(self):
		return 1

	def getStateRep(self):
		return self.pidVec+self.paramVec+self.varVec

class Sub(BinOP,Algebra):
	def __init__(self, lf:Union[Variable,int], rf:Union[Variable,int]):
		self._left_form = lf
		self._right_form = rf
		self._f_type = '-'
		self.preorder = -1

	def f2str(self,expand=False):
		# l_int=isinstance(self._left_form,int)
		# r_int=isinstance(self._right_form,int)
		# l_str = self._left_form.f2str() if not l_int else str(self._left_form)
		# r_str = self._right_form.f2str() if not r_int else str(self._right_form)
		# return l_str +self._f_type +r_str
		l_int=isinstance(self._left_form,int)
		r_int=isinstance(self._right_form,int)
		l_val = self._left_form.eval() if not l_int else self._left_form
		r_val = self._right_form.eval() if not r_int else self._right_form
		return str(l_val) +self._f_type +str(r_val)

	def eval(self):
		l_int=isinstance(self._left_form,int)
		r_int=isinstance(self._right_form,int)
		l_val = self._left_form.eval() if not l_int else self._left_form
		r_val = self._right_form.eval() if not r_int else self._right_form
		return l_val-r_val

	def getRole(self):
		return None

class Add(BinOP,Algebra):
	def __init__(self, lf:Union[Variable,int], rf:Union[Variable,int]):
		self._left_form = lf
		self._right_form = rf
		self._f_type = '+'
		self.preorder = -1

	def f2str(self,expand=False):
		# l_int=isinstance(self._left_form,int)
		# r_int=isinstance(self._right_form,int)
		# l_str = self._left_form.f2str() if not l_int else str(self._left_form)
		# r_str = self._right_form.f2str() if not r_int else str(self._right_form)
		# return l_str +self._f_type +r_str
		l_int=isinstance(self._left_form,int)
		r_int=isinstance(self._right_form,int)
		l_val = self._left_form.eval() if not l_int else self._left_form
		r_val = self._right_form.eval() if not r_int else self._right_form
		return str(l_val) +self._f_type +str(r_val)

	def eval(self):
		l_int=isinstance(self._left_form,int)
		r_int=isinstance(self._right_form,int)
		l_val = self._left_form.eval() if not l_int else self._left_form
		r_val = self._right_form.eval() if not r_int else self._right_form
		return l_val+r_val

	def getRole(self):
		return None

class EQ(BinOP,Algebra):
	def __init__(self, lf:Union[Variable,int], rf:Union[Variable,int]):
		self._left_form = lf
		self._right_form = rf
		self._f_type = '=='
		self.preorder = -1

	def f2str(self,expand=False):
		l_int=isinstance(self._left_form,int)
		r_int=isinstance(self._right_form,int)
		l_str = self._left_form.f2str() if not l_int else str(self._left_form)
		r_str = self._right_form.f2str() if not r_int else str(self._right_form)
		return l_str +self._f_type +r_str

	def eval(self):
		l_int=isinstance(self._left_form,int)
		r_int=isinstance(self._right_form,int)
		l_val = self._left_form.eval() if not l_int else self._left_form
		r_val = self._right_form.eval() if not r_int else self._right_form
		return 1 if l_val==r_val else 0

	def getRole(self):
		return None

class NEQ(BinOP,Algebra):
	def __init__(self, lf:Union[Variable,int], rf:Union[Variable,int]):
		self._left_form = lf
		self._right_form = rf
		self._f_type = '!='
		self.preorder = -1

	def f2str(self,expand=False):
		l_int=isinstance(self._left_form,int)
		r_int=isinstance(self._right_form,int)
		l_str = self._left_form.f2str() if not l_int else str(self._left_form)
		r_str = self._right_form.f2str() if not r_int else str(self._right_form)
		return l_str +self._f_type +r_str

	def eval(self):
		l_int=isinstance(self._left_form,int)
		r_int=isinstance(self._right_form,int)
		l_val = self._left_form.eval() if not l_int else self._left_form
		r_val = self._right_form.eval() if not r_int else self._right_form
		return 1 if l_val!=r_val else 0

	def getRole(self):
		return None

class GE(BinOP,Algebra):
	def __init__(self, lf:Union[Variable,int], rf:Union[Variable,int]):
		self._left_form = lf
		self._right_form = rf
		self._f_type = '>='
		self.preorder = -1

	def f2str(self,expand=False):
		l_int=isinstance(self._left_form,int)
		r_int=isinstance(self._right_form,int)
		l_str = self._left_form.f2str() if not l_int else str(self._left_form)
		r_str = self._right_form.f2str() if not r_int else str(self._right_form)
		return l_str +self._f_type +r_str

	def eval(self):
		l_int=isinstance(self._left_form,int)
		r_int=isinstance(self._right_form,int)
		l_val = self._left_form.eval() if not l_int else self._left_form
		r_val = self._right_form.eval() if not r_int else self._right_form
		return 1 if l_val>=r_val else 0

	def getRole(self):
		return None

class LE(BinOP,Algebra):
	def __init__(self, lf:Union[Variable,int], rf:Union[Variable,int]):
		self._left_form = lf
		self._right_form = rf
		self._f_type = '<='
		self.preorder = -1

	def f2str(self,expand=False):
		l_int=isinstance(self._left_form,int)
		r_int=isinstance(self._right_form,int)
		l_str = self._left_form.f2str() if not l_int else str(self._left_form)
		r_str = self._right_form.f2str() if not r_int else str(self._right_form)
		return l_str +self._f_type +r_str

	def eval(self):
		l_int=isinstance(self._left_form,int)
		r_int=isinstance(self._right_form,int)
		l_val = self._left_form.eval() if not l_int else self._left_form
		r_val = self._right_form.eval() if not r_int else self._right_form
		return 1 if l_val<=r_val else 0

	def getRole(self):
		return None

class GT(BinOP,Algebra):
	def __init__(self, lf:Union[Variable,int], rf:Union[Variable,int]):
		self._left_form = lf
		self._right_form = rf
		self._f_type = '>'
		self.preorder = -1

	def f2str(self,expand=False):
		l_int=isinstance(self._left_form,int)
		r_int=isinstance(self._right_form,int)
		l_str = self._left_form.f2str() if not l_int else str(self._left_form)
		r_str = self._right_form.f2str() if not r_int else str(self._right_form)
		return l_str +self._f_type +r_str

	def eval(self):
		l_int=isinstance(self._left_form,int)
		r_int=isinstance(self._right_form,int)
		l_val = self._left_form.eval() if not l_int else self._left_form
		r_val = self._right_form.eval() if not r_int else self._right_form
		return 1 if l_val>r_val else 0

	def getRole(self):
		return None

class LS(BinOP,Algebra):
	def __init__(self, lf:Union[Variable,int], rf:Union[Variable,int]):
		self._left_form = lf
		self._right_form = rf
		self._f_type = '<'
		self.preorder = -1

	def f2str(self,expand=False):
		l_int=isinstance(self._left_form,int)
		r_int=isinstance(self._right_form,int)
		l_str = self._left_form.f2str() if not l_int else str(self._left_form)
		r_str = self._right_form.f2str() if not r_int else str(self._right_form)
		return l_str +self._f_type +r_str

	def eval(self):
		l_int=isinstance(self._left_form,int)
		r_int=isinstance(self._right_form,int)
		l_val = self._left_form.eval() if not l_int else self._left_form
		r_val = self._right_form.eval() if not r_int else self._right_form
		return 1 if l_val<r_val else 0

	def getRole(self):
		return None

class Alg_Or(BinOP,Algebra):
	def __init__(self, lf:Union[Variable,int], rf:Union[Variable,int]):
		self._left_form = lf
		self._right_form = rf
		self._f_type = '|'
		self.preorder = -1

	def f2str(self,expand=False):
		l_int=isinstance(self._left_form,int)
		r_int=isinstance(self._right_form,int)
		l_str = self._left_form.f2str() if not l_int else str(self._left_form)
		r_str = self._right_form.f2str() if not r_int else str(self._right_form)
		return l_str +self._f_type +r_str

	def eval(self):
		l_int=isinstance(self._left_form,int)
		r_int=isinstance(self._right_form,int)
		l_val = self._left_form.eval() if not l_int else self._left_form
		r_val = self._right_form.eval() if not r_int else self._right_form
		return 1 if l_val or r_val else 0

	def getRole(self):
		return None

class Alg_Xor(BinOP,Algebra):
	def __init__(self, lf:Union[Variable,int], rf:Union[Variable,int]):
		self._left_form = lf
		self._right_form = rf
		self._f_type = '^'
		self.preorder = -1

	def f2str(self,expand=False):
		l_int=isinstance(self._left_form,int)
		r_int=isinstance(self._right_form,int)
		l_str = self._left_form.f2str() if not l_int else str(self._left_form)
		r_str = self._right_form.f2str() if not r_int else str(self._right_form)
		return l_str +self._f_type +r_str

	def eval(self):
		l_int=isinstance(self._left_form,int)
		r_int=isinstance(self._right_form,int)
		l_val = self._left_form.eval() if not l_int else self._left_form
		r_val = self._right_form.eval() if not r_int else self._right_form
		return 1 if l_val ^ r_val else 0

	def getRole(self):
		return None

class Alg_And(BinOP,Algebra):
	def __init__(self, lf:Union[Variable,int], rf:Union[Variable,int]):
		self._left_form = lf
		self._right_form = rf
		self._f_type = '&'
		self.preorder = -1

	def f2str(self,expand=False):
		l_int=isinstance(self._left_form,int)
		r_int=isinstance(self._right_form,int)
		l_str = self._left_form.f2str() if not l_int else str(self._left_form)
		r_str = self._right_form.f2str() if not r_int else str(self._right_form)
		return l_str +self._f_type +r_str

	def eval(self):
		l_int=isinstance(self._left_form,int)
		r_int=isinstance(self._right_form,int)
		l_val = self._left_form.eval() if not l_int else self._left_form
		r_val = self._right_form.eval() if not r_int else self._right_form
		return 1 if l_val and r_val else 0

	def getRole(self):
		return None

class Not(UnaOP):
	def __init__(self, f:Formula):
		self._sub_form = f
		self._f_type = 'not'
		self.preorder = -1
		self.paramVec = [-1]*P_PARAMS
		self.pidVec = [-1]*P_ID
		self.varVec = [-1]*F_VARS

	def move(self,act,player1,player2):
		# transparent to game
		player1.switch()
		player2.switch()
		# assert(self.preorder>-1 and self.preorder<=F_VARS)
		self._sub_form.varVec = self.varVec[:] if self._sub_form._f_type!='pred' else self._sub_form.varVec
		self._sub_form.pidVec = self.pidVec[:] if self._sub_form._f_type!='pred' else self._sub_form.pidVec
		self._sub_form.paramVec = self.paramVec[:] if self._sub_form._f_type!='pred' else self._sub_form.paramVec
		return self._sub_form

	def getRole(self):
		return None

	def getStateRep(self):
		return self.pidVec+self.paramVec+self.varVec

	def traverseFormula(self, preorder):
		# transparent to game
		preorder = self._sub_form.traverseFormula(preorder)
		return preorder

class Predicate(Formula):
	@property
	def symbol(self) -> str:
		return self._var
	@symbol.setter
	def symbol(self, val: str):
		self._symbol = val

	@property
	def args(self) -> List[Union[Variable,Algebra,int]]:
		return self._args
	@args.setter
	def args(self, val: List[Union[Variable,Algebra,int]]):
		self._args = val

	def __init__(self,s,args):
		self._symbol = s
		self._args = args
		self._f_type = 'pred'
		self.preorder = -1

	def f2str(self,expand=False):
		return self._symbol+'('+','.join([v.f2str() if not isinstance(v,int) else str(v) for v in self._args])+')'

	def count_operators(self):
		return 0

	def getActions(self):
		return []

	def move(self,act,player1,player2):
		return None

	def getRep(self):
		return []

	def traverseFormula(self, preorder):
		self.preorder = 0
		# Terminal
		return preorder-1

class Role():
	def __init__(self, role_name:str):
		self.role_name = role_name

class Role_P(Role):
	def __init__(self):
		super().__init__(1)

	def switch(self):
		return Role_OP()

class Role_OP(Role):
	def __init__(self):
		super().__init__(-1)

	def switch(self):
		return Role_P()

class Player():
	def __init__(self, role:Role, name:str):
		self.role = role
		self.name = name

	def switch(self):
		self.role = self.role.switch()

	def getRoleName(self):
		return self.role.role_name

class SemanticGame():
	def __init__(self,f:Formula,p1:Player,p2:Player):
		self.f = f
		self.player1 = p1
		self.player2 = p2
		self.history = []
		self.last_moved_player = p1
		self.game_outcome = None

	def play(self):
		while self.f is not None:
			role2player={}
			role2player[self.player1.role.role_name] = self.player1.name
			role2player[self.player2.role.role_name] = self.player2.name
			cur_role = self.f.getRole()
			cur_state = self.f.f2str(True)
			self.game_outcome = True if cur_state=='True' else False if cur_state=='False' else None
			self.last_moved_player = role2player[cur_role] if cur_role else self.last_moved_player
			if len(self.f.getActions())>0:
				print(role2player)
				print("{}:{} -> {}".format(role2player[cur_role] if cur_role else None,cur_role,cur_state))
				print("stateRep:{}".format(self.f.getStateRep()))
				print("order:{}".format(self.f.preorder))
				print(self.f.getActions())
			if self.f.getActions is not None and len(self.f.getActions())>0:
				act = input("take action : ")
				self.history.append((role2player[cur_role] if cur_role else None,act))
				self.f = self.f.move(act,self.player1,self.player2)
			else:
				# self.history.append(None)
				self.f = self.f.move(None,self.player1,self.player2)

		winner = None
		if self.game_outcome:
			winner = role2player[1]
		else:
			winner = role2player[-1]
		print("last move by:{}".format(self.last_moved_player))
		print("winner:{}".format(winner))
		print("action history:{}".format(self.history))

import re

def generatePredicate(name, params, body, fp_type=None):
	# generate init params
	global predicate_register
	if name in predicate_register.values():
		return
	else:
		global predicate_id
		predicate_id += 1
		predicate_register[predicate_id]=name

	init_params=''
	for p in params:
		init_params += ',{}:Union[Variable,Algebra,int]'.format(p)

	# generate init body
	init_body='self._args = ['
	for p in params:
		init_body += p+','
	init_body=init_body[:-1]+']'

	# generate definition params decleration
	params_def=''
	i=0
	for p in params:
		params_def += """
		{0}=self._args[{1}]
		if isinstance(self._args[{1}], Variable):
			self.paramVec += [self._args[{1}].eval()]
		elif isinstance(self._args[{1}],int):
			self.paramVec += [self._args[{1}]]
		elif isinstance(self._args[{1}], Algebra):
			self.paramVec += [self._args[{1}].eval()]
		elif isinstance(self._args[{1}], TransitionSystem):
			self.paramVec += self._args[{1}].eval().getStateRep()[:]
		# print(self._args[{1}])
		""".format(p,i)
		i+=1

	# generate definition body
	e = set([s[7:-1] for s in re.findall("Exists\(.*?,", body)])
	a = set([s[7:-1] for s in re.findall("Forall\(.*?,", body)])
	v = a.union(e)
	body_def = ''
	for p in v:
		body_def += """
		{0}=Variable('{0}')
		""".format(p) 

	# synthesis class 
	class_str="""
global {0}
class {0}(Predicate):
	def __init__(self{1}):
		self.pid = {6}
		self._symbol = '{0}'
		{2}
		self._f_type = 'pred'
		self.preorder = 0
		self.paramVec = [-1]*P_PARAMS
		self.pidVec = [{6}]
		# self.pidVec = []
		self.varVec = [-1]*F_VARS
		self.define = None
		self.fp_type = {7}

	def definition(self):
		self.paramVec = []
		{3}
		{4}
		body = {5}
		init_statePreorder(body)
		return body

	def move(self,act,player1,player2):
		if self.define is None:
			self.define = self.definition()
		pred = self.define
		if isinstance(pred,bool):
			return None
		else:
			pred.varVec = self.varVec[:]
			pred.varVec[self.preorder] = 0
			pred.pidVec = self.pidVec[:]
			pred.paramVec = self.paramVec[:]
			# return pred.move(act,player1,player2)

			return pred

	def getActions(self):
		if self.define is None:
			self.define = self.definition()
		pred = self.define
		if isinstance(pred,bool):
			return []
		else:
			# return pred.getActions()
			return [0]

	def f2str(self,expand=False):
		if expand:
			if self.define is None:
				self.define = self.definition()
			pred = self.define
			return pred.f2str() if not isinstance(pred,bool) else str(pred)
		else:
			return super().f2str()

	def getRole(self):
		if self.define is None:
			self.define = self.definition()
		pred = self.define
		if isinstance(pred,bool):
			return None
		else:
			return pred.getRole()

	def getStateRep(self):
		return self.pidVec+self.paramVec+self.varVec

	""".format(name,init_params,init_body,params_def,body_def,body,predicate_id,fp_type)
	# print(class_str)
	exec(class_str)

def generateGFP(name, params, body):
	generatePredicate(name, params, body, 1)
def generateLFP(name, params, body):
	generatePredicate(name, params, body, 0)


def init_statePreorder(formula):
	max_preorder = formula.traverseFormula(1)
	return max_preorder

predicate_id = 0
predicate_register={}

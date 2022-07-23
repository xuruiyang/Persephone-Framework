from . import Persephone
import sys
sys.path.append('..')
from Game import Game
import numpy as np
import copy


class PersephoneGame(Game):
    def __init__(self, n=3):
        self.n = n
        self.MAX_VEC_SIZE = Persephone.S_TOTAL+1
        self.MAX_ACT_SIZE = 4
        # Persephone.generatePredicate('HSR', ['k','q','n'],"ITE(EQ(n,1),TrueConst(),ITE(Alg_Or(EQ(k,0),EQ(q,0)),FalseConst(),Exists(m,Domain(1,n),And(HSR(Sub(k,1),Sub(q,1),m),HSR(k,Sub(q,1),Sub(n,m))))))")
        # Persephone.generatePredicate('Legal', ['m','n'],"ITE(Alg_And(LS(m,n),GT(m,0)),TrueConst(),FalseConst())")
        # Persephone.generatePredicate('HSR', ['k','q','n'],"ITE(EQ(n,1),TrueConst(),ITE(Alg_Or(LS(n,1),Alg_Or(EQ(k,0),EQ(q,0))),FalseConst(),Exists(m,Domain(1,35),And(Legal(m,n),And(HSR(Sub(k,1),Sub(q,1),m),HSR(k,Sub(q,1),Sub(n,m)))))))")
        # Persephone.generatePredicate('HSR_COMP', ['k','q'],"Exists(n,Domain(1,19),And(HSR(k,q,n),Forall(n1,Domain(Add(n,1),20),Not(HSR(k,q,n1)))))")
        # self.init_state = Persephone.HSR_COMP(4,4)
        # Persephone.generatePredicate('H', ['n','q'],"ITE(EQ(n,2),TrueConst(),ITE(Alg_Or(LS(n,2),Alg_And(EQ(q,0),GT(n,2))),FalseConst(),Exists(m,Domain(1,n),And(H(m,Sub(q,1)),H(Sub(n,m),Sub(q,1))))))")
        # Persephone.generatePredicate('FB', ['x'],"Forall(b,Domain(3,7),HSR(2,Mult(2,b),Mult(b,b)))")
        # self.init_state = Persephone.FB(0)
        # self.init_state = Persephone.HSR(2,24,144)
        # self.init_state = Persephone.H(100,30)
        
        # Persephone.generatePredicate('MOD', ['m','n'],"ITE(Mod(m,n),TrueConst(),FalseConst())")
        # Persephone.generatePredicate('NMOD', ['m','n'],"ITE(Mod(m,n),FalseConst(),TrueConst())")
        # Persephone.generatePredicate('CD', ['m','n','x'],"And(MOD(m,x),MOD(n,x))")
        # Persephone.generatePredicate('NCD', ['m','n','x'],"Or(NMOD(m,x),NMOD(n,x))")
        # Persephone.generatePredicate('GCD', ['m','n'],"Exists(x,Domain(1,m),And(CD(m,n,x),Forall(y,Domain(Add(x,1),n),NCD(m,n,y))))")
        # Persephone.generatePredicate('GCD_WRAPPER', ['m','n'],"ITE(LS(m,n),GCD(m,n),GCD(n,m))")
        # self.init_state = Persephone.GCD_WRAPPER(53,67)

        # NG: missing player info when entring the predicate
        # Persephone.generatePredicate('NIM', ['k','n'],"ITE(LE(n,k),TrueConst(),Exists(m,Domain(1,Add(k,1)),Not(NIM(k,Sub(n,m)))))")
        
        # self.init_state = Persephone.NIM(3,50)

    def getInitBoard(self):
        # return initial board (numpy board)
        # the init state must be a playable state, otherwise, auto run until a playable state
        b = copy.deepcopy(self.init_state)
        player1,player2 = self.alignPlayers()
        while b.getActions()==[]:
            b=b.move(None,player1,player2)
        # define player info
        return b,player1,player2

    def alignPlayers(self, curPlayer=1, curRole=1):
        if curPlayer==1 and curRole==1:
            player1 = Persephone.Player(Persephone.Role_P(),1)
            player2 = Persephone.Player(Persephone.Role_OP(),-1)
        elif curPlayer==1 and curRole==-1:
            player1 = Persephone.Player(Persephone.Role_OP(),1)
            player2 = Persephone.Player(Persephone.Role_P(),-1)
        elif curPlayer==-1 and curRole==1:
            player1 = Persephone.Player(Persephone.Role_P(),-1)
            player2 = Persephone.Player(Persephone.Role_OP(),1)
        else:
            player1 = Persephone.Player(Persephone.Role_OP(),-1)
            player2 = Persephone.Player(Persephone.Role_P(),1)
        return player1,player2

    def getBoardSize(self):
        # (a,b) tuple
        return (1, self.MAX_VEC_SIZE)

    def getActionSize(self):
        # return number of actions + 1 none action
        return self.MAX_ACT_SIZE + 1

    def getNextState(self, board, player, role, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.MAX_ACT_SIZE:
            return (copy.deepcopy(board), -player, -role)
        b = copy.deepcopy(board)
        player1,player2 = self.alignPlayers(player,role)
        b = b.move(action, player1, player2)
        while b.getActions()==[]:
            b=b.move(None,player1,player2)
        if b.getRole() is None:
            assert(b.f_type=='t' or b.f_type=='f')
        next_role = b.getRole() if b.getRole() is not None else role
        if player1.getRoleName() == next_role:
            next_player = player1.name
        else:
            next_player = player2.name
        return (b, next_player, next_role)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        if board.getActions() is None:
            print(board.f2str())
        assert(board.getActions() is not None)
        valids = [0]*self.getActionSize()
        b = copy.deepcopy(board)
        legalMoves =  b.getActions()
        if len(legalMoves)==0:
            valids[-1]=1
            return np.array(valids)
        for m in legalMoves:
            valids[m]=1
        return np.array(valids)

    def getGameEnded(self, board, player, role):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        if board.f_type == 't':
            return 1
        elif board.f_type == 'f':
            return -1
        else:
            return 0

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return board

    def getSymmetries(self, board, pi):
        return [([board.getStateRep()],pi)]

    def stringRepresentation(self, board, player):
        # 8x8 numpy array (canonical board)
        return str(board.getStateRep()+[player])

    @staticmethod
    def display(board):
        print(board.f2str())

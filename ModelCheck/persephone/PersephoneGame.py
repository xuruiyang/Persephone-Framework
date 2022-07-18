from . import PersephoneMu as Persephone
import sys
sys.path.append('..')
from Game import Game
import numpy as np
import copy
from ts.TransitionSystem import HSRTranSys, DinePhiTranSys
import conf


class PersephoneGame(Game):
    def __init__(self, n=3):
        self.n = n
        self.MAX_VEC_SIZE = Persephone.S_TOTAL+1
        self.MAX_ACT_SIZE = conf.MAX_ACT_SIZE
        Persephone.generateGFP('Z',['s'],"And(Or(P(s),X(s)),Forall(a,s.getActions(),Z(s.next(a))))")
        Persephone.generateLFP('X',['s'],"Or(Q(s),Forall(a,s.getActions(),X(s.next(a))))")
        Persephone.generatePredicate('P',['s'],'ITE(EQ(s.phi0_hungry(),0),TrueConst(),FalseConst())')
        Persephone.generatePredicate('Q',['s'],'ITE(EQ(s.phi0_eat(),1),TrueConst(),FalseConst())')
        self.init_state = Persephone.Z(DinePhiTranSys(conf.NUM_P))

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
            print("action == self.MAX_ACT_SIZE? Please Check!")
            return (board, -player, -role)
        b = copy.deepcopy(board)
        player1,player2 = self.alignPlayers(player,role)
        b = b.move(action, player1, player2)
        while b.getActions()==[]:
            b=b.move(None,player1,player2)
        next_role = b.getRole() if b.getRole() is not None else role
        assert next_role is not None
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

    def getGameEnded(self, board, player, role, changes):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost

        def md(changes):
            return np.max(changes)-np.min(changes)>conf.MAX_D 

        if board.f_type == 't':
            return 1 if role==1 else -1
        elif board.f_type == 'f':
            return 1 if role==-1 else -1
        elif md(changes):
            return 1 if role==1 else -1
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

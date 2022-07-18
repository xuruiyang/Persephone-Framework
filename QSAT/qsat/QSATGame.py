from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .QSATLogic import Board
import numpy as np
import networkx as nx
import conf

class QSATGame(Game):
    def __init__(self, qdimacs):
        self.qdimacs = qdimacs

    def getInitBoard(self):
        # return initial board (numpy board)
        # only one problem instance for the time being (maybe more later...)

        # read in qdimacs
        # parse qdimacs into 4 graphs
        # a board is a tuple of 4 graphs (e2a,a2e,l2c, and reflexive) with an active variable (i.e. head)
        b = Board(self.qdimacs)
        return b

    def getBoardSize(self):
        # no use of this interface
        return -1

    def getActionSize(self):
        # return number of actions, it must be 2
        return 2

    def getNextState(self, board, player, action):
        # proceed graph
        if isinstance(board, list):
            b = Board(board[-1],True)
        else:
            b = Board(self.getCanonicalForm(board, player)[-1],True)
        next_player = b.execute_move(action, player)
        return (b, next_player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        if board[-2]==0:
            valids = [1]*self.getActionSize()
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        if isinstance(board, list):
            return board[-2]
        else:
            return self.getCanonicalForm(board, player)[-2]

    def getCanonicalForm(self, board, player):
        # use 3D matrix as CanonicalBoard
        # CanonicalBoard will only be used by NN
        (g1,g2,g3,g4)=board.graphs
        num_vertices = conf.NUM_V
        # print(len(g3.nodes()))
        assert len(g1.nodes)==len(g2.nodes)
        assert len(g2.nodes)==len(g3.nodes)
        assert len(g3.nodes)==len(g4.nodes)
        hidden_size = 128
        num_edge_types = 4

        # It is important to make sure all input data has the same shape!

        init_node_reps = np.zeros((num_vertices, hidden_size))
        i=0
        for node in sorted(g1.nodes()):
            n_type = g1.nodes[node]['n_type']
            if n_type=='elit':
                init_node_reps[i]=np.pad(np.asarray([1,0,0]),pad_width=[0,hidden_size-3],mode='constant')
            elif n_type=='alit':
                init_node_reps[i]=np.pad(np.asarray([0,1,0]),pad_width=[0,hidden_size-3],mode='constant')
            else:
                init_node_reps[i]=np.pad(np.asarray([0,0,1]),pad_width=[0,hidden_size-3],mode='constant')
            i+=1

        amats = np.zeros((num_edge_types, num_vertices, num_vertices))
        amats[0]=np.pad(nx.to_numpy_matrix(g1,nodelist=sorted(g1.nodes())),pad_width=[[0,num_vertices-len(g1.nodes)],[0,num_vertices-len(g1.nodes)]],mode='constant')
        amats[1]=np.pad(nx.to_numpy_matrix(g2,nodelist=sorted(g2.nodes())),pad_width=[[0,num_vertices-len(g1.nodes)],[0,num_vertices-len(g1.nodes)]],mode='constant')
        amats[2]=np.pad(nx.to_numpy_matrix(g3,nodelist=sorted(g3.nodes())),pad_width=[[0,num_vertices-len(g1.nodes)],[0,num_vertices-len(g1.nodes)]],mode='constant')
        amats[3]=np.pad(nx.to_numpy_matrix(g4,nodelist=sorted(g4.nodes())),pad_width=[[0,num_vertices-len(g1.nodes)],[0,num_vertices-len(g1.nodes)]],mode='constant')

        # num_vertices = len(g1.nodes)
        game_state = board.game_state

        return [init_node_reps,amats,num_vertices,game_state,board]

    def getSymmetries(self, board, pi):
        # dummy function, symmetry will handle by GNN
        return [(board,pi)]

    def stringRepresentation1(self, board):
        # [4,n,n] (canonical board 3D matrix)
        b= board[:-1]
        # return np.array2string(np.asarray(b)) This is in correct since it will use ... to fill the string!
        return np.asarray(b).tobytes()

    def stringRepresentation(self, board):
        # [4,n,n] (canonical board 3D matrix)
        b= board[-1]
        return b.assign_seq

def display(board):
    print("TBD")

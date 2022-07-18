import Arena
from MCTS import MCTS

from qsat.QSATPlayers import *
from qsat.QSATGame import QSATGame as Game
from qsat.tensorflow.NNet import NNetWrapper as nn

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = Game('qsat/qdimacs_test4.txt')

# all players
rp = RandomPlayer(g).play
hp = HumanQSATPlayer(g).play

# nnet players
nnet = nn(g)
nnet2 = nn(g,identity=2)

nnet.load_checkpoint('./temp/', 'best1.pth.tar')
nnet2.load_checkpoint('./temp/', 'best2.pth.tar')

args1 = dotdict({'numMCTSSims': 25, 'cpuct':1.2})
# mcts1 = MCTS(g, nnet, nnet2, args1)
# n1p = lambda x,p: np.argmax(mcts1.getActionProb(x, p, temp=0))


#n2 = NNet(g)
#n2.load_checkpoint('/dev/8x50x25/','best.pth.tar')
#args2 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
#mcts2 = MCTS(g, n2, args2)
#n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

# arena = Arena.Arena(n1p, rp, g)
# arena.playGames(10, verbose=False)


def enumerator(seq_len, seq):
	if seq_len==10:
		print()
		print(seq)
		sp = Seqlayer(g, seq).play
		mcts1 = MCTS(g, nnet, nnet2, args1)
		n1p = lambda x,p: np.argmax(mcts1.getActionProb(x, p, temp=0))
		arena = Arena.Arena(n1p, sp, g)
		arena.playGame(verbose=False)
	else:
		enumerator(seq_len+1, seq+[0])
		enumerator(seq_len+1, seq+[1])

def enumerator2(seq_len, seq):
	if seq_len==11:
		print()
		print(seq)
		sp = Seqlayer(g, seq).play
		mcts2 = MCTS(g, nnet, nnet2, args1)
		n2p = lambda x,p: np.argmax(mcts2.getActionProb(x, p, temp=0))
		arena = Arena.Arena(sp, n2p, g)
		arena.playGame(verbose=False)
	else:
		enumerator2(seq_len+1, seq+[0])
		enumerator2(seq_len+1, seq+[1])

states = set()
def enumerator3(board, curPlayer, g, act_seq):
	states.add(act_seq)
	result = g.getGameEnded(board, curPlayer)
	if result!=0:
		print(act_seq+" : "+str(result))
		return result==1
	m_board, m_curPlayer = g.getNextState(board, curPlayer, 0)
	r1 = enumerator3(m_board, m_curPlayer, g, act_seq+str(0))
	if ((curPlayer==1) and r1) or ((curPlayer==-1) and not r1):
		return r1
	m_board, m_curPlayer = g.getNextState(board, curPlayer, 1)
	r2 = enumerator3(m_board, m_curPlayer, g, act_seq+str(1))
	return r1 or r2 if curPlayer==1 else r1 and r2

enumerator3(g.getInitBoard(), 1, g, "")
print(len(states))



# enumerator(0, [1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
# enumerator2(0, [])


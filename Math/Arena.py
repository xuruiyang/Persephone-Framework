import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time
import pdb
import logging

logging.basicConfig(filename='experiment-7-7-128-2nn.log',format='%(asctime)s %(message)s')
# logging.basicConfig(format='%(asctime)s %(message)s')

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, display=None, mcts=None, main_player=1):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.mcts=mcts
        self.main_player=main_player
    
    def playGame(self, verbose=False, greedy=True):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]

        board,player1,player2 = self.game.getInitBoard()
        if board.getActions() is None:
            print("Formula trivially true, terminate")
            exit()
        curRole = board.getRole()
        curPlayer = player1.name if player1.getRoleName() == curRole else player2.name

        it = 0
        # print()
        self.preFault = False
        while self.game.getGameEnded(board, curPlayer, curRole)==0:
            it+=1
            pi = players[curPlayer+1](self.game.getCanonicalForm(board, curPlayer), curPlayer, curRole)
            action = np.argmax(pi) if greedy else np.random.choice(len(pi), p=pi)
            signal = None

            vec_cur = board.getStateRep()
            vec_cur.append(curPlayer)
            if curPlayer==1:
                ps, v1, concept1 = self.mcts.nnet.predict(np.array([vec_cur]))
                _, v2, concept2 = self.mcts.nnet2.predict(np.array([vec_cur]))
            else:
                _, v1, concept1 = self.mcts.nnet.predict(np.array([vec_cur]))
                ps, v2, concept2 = self.mcts.nnet2.predict(np.array([vec_cur]))
            valids = self.game.getValidMoves(board, curPlayer)
            ps = ps*valids      # masking invalid moves
            sum_Ps_s = np.sum(ps)
            if sum_Ps_s > 0:
                ps /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked for player {}, do workaround.".format(curPlayer))
                ps = ps + valids
                ps /= np.sum(ps)

            # check whether the position is forced
            for m in range(1,board.getStateRep()[1]+1):
                if (board.getStateRep()[2]-m)%(board.getStateRep()[1]+1)==0:
                    signal = False
                    break

            # in single pile Nim (normal), the winning position is n%k!=0
            # an action is correct if it make the next player not in winning position
            if signal is not None:
                signal = (board.getStateRep()[2]-action)%(board.getStateRep()[1]+1)==0

            if verbose:
                assert(self.display)
                self.display(board, curPlayer)
                # print(board.f2str(expand=True))
                # print(f'{np.argmax(pi)}:{np.max(pi)}',f'{np.argmax(ps)}:{np.max(ps)}',v1,v2)
                # print(v1,v2)
                # print(pi)
                # print(ps)
                if curPlayer==1:
                    print("NN1:",concept1,"Target:",0 if (board.getStateRep()[2])%(board.getStateRep()[1]+1)==0 else 1)
                else:
                    print("NN2:",concept2,"Target:",0 if (board.getStateRep()[2])%(board.getStateRep()[1]+1)==0 else 1)
                print("Turn ", str(it), "Player ", str(curPlayer), "Role", str(curRole), "Action", action, "C" if signal==True else "I" if signal==False else "F")
               


            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer),1)

            if valids[action]==0:
                print(action)
                print(board.f2str())
                assert valids[action] >0
            board, curPlayer, curRole = self.game.getNextState(board, curPlayer, curRole, action)
        if verbose:
            assert(self.display)
            # print(board.f2str(expand=True))
            print("Game over: Turn ", str(it+1), "Result ", str(self.game.getGameEnded(board, curPlayer, curRole)), "Player ", str(curPlayer), "Role", str(curRole))
            self.display(board, curPlayer)
            vec_cur = [1, 3, 100, -1, -1, -1, 1]
            _, _, concept1 = self.mcts.nnet.predict(np.array([vec_cur]))
            _, _, concept2 = self.mcts.nnet2.predict(np.array([vec_cur]))
            print("NN1:",concept1,"NN2:",concept2,"Target:",0)
            vec_cur = [1, 3, 99, -1, -1, -1, 1]
            _, _, concept1 = self.mcts.nnet.predict(np.array([vec_cur]))
            _, _, concept2 = self.mcts.nnet2.predict(np.array([vec_cur]))
            print("NN1:",concept1,"NN2:",concept2,"Target:",1)
        r = self.game.getGameEnded(board, curPlayer, curRole)
        return  r if curPlayer==self.main_player else -r

    def playGames(self, num, verbose=False, greedy=True):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        # bar = Bar('Arena playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        num = int(num/2)
        mainWon = 0
        fixWon = 0
        for _ in range(num):
            gameResult = self.playGame(verbose=verbose, greedy=greedy)
            if gameResult==1:
                mainWon+=1
            else:
                fixWon+=1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            # bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:} | P: {p1:} OP: {p2:}'.format(eps=eps, maxeps=maxeps, et=eps_time.avg,
                                                                                                       # total=bar.elapsed_td, eta=bar.eta_td, p1=oneWon, p2=twoWon)
            # logging.warning('({eps}/{maxeps}) Eps Time: {et:.3f}s | P_1: {p1:} OP_2: {p2:}'.format(eps=eps, maxeps=maxeps, et=eps_time.avg, p1=p1, p2=p2))
            print('({eps}/{maxeps}) Eps Time: {et:.3f}s | Main: {p1:} Fix: {p2:}'.format(eps=eps, maxeps=maxeps, et=eps_time.avg, p1=mainWon, p2=fixWon))
            # bar.next()

        return mainWon, fixWon

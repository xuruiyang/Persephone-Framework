import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time
import pdb
import logging
from persephone.PersephoneMu import predicate_register
from collections import deque, defaultdict
from utils import fair
import conf
import copy

logging.basicConfig(filename='experiment-mod-7-7-128.log',format='%(asctime)s %(message)s')

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, display=None):
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

        self.correct_count={1:0,-1:0}
        self.step_count={1:0,-1:0}
        self.preFaults={1:False,-1:False}
        self.switch_count={1:0,-1:0}
        
        self.prop_correct = 0
        self.prop_n = 0
        self.oppo_correct = 0
        self.oppo_n = 0

        # a player makes a fault iff he is in a winning position but take an incorrect move
        self.preFault = False

        # a switch happens when the previous player makes a fault and current player captures it.
        # a capture means the current player makes a correct move,
        # due to the winning position from previous one's fault
        self.prop_switch = 0
        self.oppo_switch = 0
        
        self.ground_truth = np.zeros((25,25))
        for i in range(25):
            self.ground_truth[i][0] = 1
            self.ground_truth[i][i] = 2**i
            if i>0:
                for j in range(1,i):
                    self.ground_truth[i][j] = self.ground_truth[i-1][j-1] + self.ground_truth[i-1][j]

    def get_ground_truth(self,k,q):
        # shift by 1
        return self.ground_truth[q][k] if k<=q else self.ground_truth[q][q]
    
    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        game = copy.deepcopy(self.game)

        board,player1,player2 = game.getInitBoard()
        if board.getActions() is None:
            print("Formula trivially true, terminate")
            exit()
        curRole = board.getRole()
        curPlayer = player1.name if player1.getRoleName() == curRole else player2.name
        prevPlayer = -curPlayer

        it = 0
        # print()
        self.preFault = False
        self.preFaults={1:False,-1:False}
        correct_move = 1
        path=deque([])
        trace=deque([])
        on_path=defaultdict(lambda: False)
        fp_count=defaultdict(lambda: 0)
        GAMMA = conf.GAMMA - 1
        BOUNDED = conf.BOUNDED
        RANDOM_SEARCH = False
        time0 = time.time()

        changes=[1]*conf.NUM_P
        mcts_dt=0
        auto_mode=False

        while game.getGameEnded(board, curPlayer, curRole, changes)==0:
        # while game.getGameEnded(board, curPlayer, curRole)==-1:
            it+=1
            game = copy.deepcopy(self.game)
            t0=time.time()

            if board._f_type=='a'or board._f_type=='e':
                if len(trace)==0:
                    trace.append(0)
                trace.append(board.getStateRep()[1:conf.NUM_P+1])

            if BOUNDED and board._f_type=='pred' and board.fp_type is not None:
                # s = game.stringRepresentation(board, curPlayer)
                ss = board.getStateRep()[1:conf.NUM_P+1]
                s = str([ss[0]]+(ss[1:conf.NUM_P]))
                # s = str(ss[0])
                if not on_path[s]:
                    path.append(s)
                    on_path[s] = True
                else:
                    while path[-1]!=s:
                        on_path[path[-1]] = False
                        fp_count[path.pop()]=0
                fp_count[s] += 1
                if fp_count[s] > GAMMA:
                    # print("LOOP DETECTED!")
                    isfair = fair(trace, num_p=conf.NUM_P,verbose=False)

                    if board.fp_type==0:
                        # least fixpoint in cycle, OP win
                        if isfair:
                            print("End at LFP")
                            r = 1 if curRole==-1 else -1
                            if verbose:
                                assert(self.display)
                                print("Game over: Turn ", str(it), "Result ", str(r), "Player ", str(curPlayer), "Role", str(curRole))
                                self.display(board, curPlayer)
                                # breakpoint()
                            return r
                    elif board.fp_type==1:
                        # greatest fixpoint in cycle, P win
                        print("End at GFP")
                        r = 1 if curRole==1 else -1
                        # breakpoint()
                        # r = r if isfair else -r
                        if verbose:
                            assert(self.display)
                            print("Game over: Turn ", str(it), "Result ", str(r), "Player ", str(curPlayer), "Role", str(curRole))
                            self.display(board, curPlayer)
                            # breakpoint()
                        return r
                    else:
                        assert False

            t1=time.time()
            if board.getStateRep()[0]==2 and board._f_type=='or1':
                if board.getStateRep()[1]==7:
                    action, pi, v = 0, [0.5,0.5], -1
                else:
                    action, pi, v = 1, [0.5,0.5], -1
            else:
                # reset=mcts_dt>0.5
                reset=False
                action, pi, v = players[curPlayer+1](game.getCanonicalForm(board, curPlayer), curPlayer, curRole, path, on_path, fp_count, trace, changes, reset) 
                # action = np.random.choice(len(board.getActions()))
                # action0 = action
                if board.getStateRep()[0]==2 and board._f_type=='a' and changes[action]-np.min(changes)==conf.MAX_D:
                    if np.min(changes) < np.min(changes[1:]):
                        action = np.argmin(changes)
                    else:
                        action = np.argmin(changes[1:])+1
                if action==0 and (np.argmin(changes)!=0 or np.max(changes[1:])-np.min(changes[1:])<=conf.MAX_D) and board.getStateRep()[0]==2 and board._f_type=='a' and (board.getStateRep()[1]==3 or board.getStateRep()[1]==4):
                    tb, _, _ = game.getNextState(board, curPlayer, curRole, action)
                    if tb.getStateRep()[1]==7:
                        action = np.argmin(changes[1:])+1
                if len(path)>5000 and board.getStateRep()[0]==2 and board._f_type=='a':
                    print("path construction:")
                    action = int(input())
                    changes = [0]*conf.NUM_P

            mcts_dt=time.time()-t1
            valids = game.getValidMoves(game.getCanonicalForm(board, curPlayer),1)

            if valids[action]==0:
                print(action)
                print(board.f2str())
                assert valids[action] >0

            if verbose:
                if board._f_type=='a'or board._f_type=='e':
                    tdt=time.time()-time0
                    print(f"{board.getStateRep()[0]},LT={len(trace)//2},LP={len(path)},DT={mcts_dt:.2f},{board.getStateRep()[1:conf.NUM_P+1]},{changes},{np.mean(changes)}")

            if board._f_type=='a'or board._f_type=='e':
                trace.append(int(action))
                changes[int(action)]+=1

            prevPlayer = curPlayer
            board, curPlayer, curRole = game.getNextState(board, curPlayer, curRole, action)
            detect_dt=time.time()-t0

            time0 = time.time()

        if verbose:
            assert(self.display)
            print("Game over: Turn ", str(it), "Result ", str(game.getGameEnded(board, curPlayer, curRole, changes)), "Player ", str(curPlayer), "Role", str(curRole))
            self.display(board, curPlayer)
        return game.getGameEnded(board, curPlayer, curRole, changes)

    def playGames(self, num, verbose=False):
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

        stop = False

        num = int(num/2)
        oneWon = 0
        twoWon = 0
        p1 = 0
        p2 = 0
        oneWon0 = 0
        twoWon0 = 0
        draws = 0
        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult==1:
                oneWon+=1
            elif gameResult==-1:
                twoWon+=1
            else:
                draws+=1
            p1 = oneWon
            p2 = twoWon
            eps += 1
            eps_time.update(time.time() - end)                                                                       total=bar.elapsed_td, eta=bar.eta_td, p1=p1, p2=p2)
            logging.warning('({eps}/{maxeps}) Eps Time: {et:.3f}s | P_1: {p1:} OP_2: {p2:}'.format(eps=eps, maxeps=maxeps, et=eps_time.avg, p1=p1, p2=p2))

        print(f"P_1_A_Faults {self.switch_count[1]} : P_2_B_Faults {self.switch_count[-1]}")
        print(f"P_1_A_Corr {self.correct_count[1]} : P_2_B_Corr {self.correct_count[-1]}")

        # if self.prop_switch==0 and self.oppo_switch==0:
        #     stop = True
        if self.switch_count[1]==0 and self.switch_count[-1]==0:
            stop = True


        self.player1, self.player2 = self.player2, self.player1
        p1=0
        p2=0

        self.prop_correct = 0
        self.prop_n = 0
        self.oppo_correct = 0
        self.oppo_n = 0

        self.preFault = False
        self.prop_switch = 0
        self.oppo_switch = 0

        self.correct_count={1:0,-1:0}
        self.step_count={1:0,-1:0}
        self.preFaults={1:False,-1:False}
        self.switch_count={1:0,-1:0}

        good_oppo = False
        if twoWon > oneWon:
            good_oppo = True
        oneWon0, twoWon0 = oneWon, twoWon
        
        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult==-1:
                oneWon+=1
                p1+=1                
            elif gameResult==1:
                twoWon+=1
                p2+=1
            else:
                draws+=1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            logging.warning('({eps}/{maxeps}) Eps Time: {et:.3f}s | P_2: {p2:} OP_1: {p1:}'.format(eps=eps, maxeps=maxeps, et=eps_time.avg, p1=p1, p2=p2))
        
        print(f"P_1_B_Faults {self.switch_count[1]} : P_2_A_Faults {self.switch_count[-1]}")
        print(f"P_1_A_Corr {self.correct_count[1]} : P_2_B_Corr {self.correct_count[-1]}")
        if self.switch_count[1]==0 and self.switch_count[-1]==0 and stop:
            stop = True
        else:
            stop = False

        return oneWon, twoWon, draws, stop

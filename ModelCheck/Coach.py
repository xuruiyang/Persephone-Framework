from collections import deque, defaultdict
import copy
from Arena import Arena
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
from utils import fair
import conf

import matplotlib.pyplot as plt
import logging



class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(copy.deepcopy(game))  # the competitor network
        self.args = args
        self.mcts = MCTS(copy.deepcopy(game), self.nnet, self.args, {}, {}, {})
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = conf.skipFirstSelfPlay    # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        game = copy.deepcopy(self.game)
        trainExamples = []
        board,player1,player2 = game.getInitBoard()
        if board.getActions() is None:
            print("Formula trivially true, terminate")
            exit()
        self.curRole = board.getRole() if board.getRole() is not None else 1
        self.curPlayer = player1.name if player1.getRoleName() == self.curRole else player2.name
        episodeStep = 0

        path=deque([])
        trace=deque([])
        fp_count=defaultdict(lambda: 0)
        on_path=defaultdict(lambda: False)
        changes=[1]*conf.NUM_P
        GAMMA=conf.GAMMA
        BOUNDED=conf.BOUNDED

        while True:
            episodeStep += 1
            game = copy.deepcopy(self.game)
            canonicalBoard = game.getCanonicalForm(board,self.curPlayer)
            if canonicalBoard._f_type=='a'or canonicalBoard._f_type=='e':
                if len(trace)==0:
                    trace.append(0)
                trace.append(canonicalBoard.getStateRep()[1:conf.NUM_P+1])
            if BOUNDED and board._f_type=='pred' and board.fp_type is not None:
                ss = canonicalBoard.getStateRep()[1:conf.NUM_P+1]
                s = str([ss[0]]+(ss[1:conf.NUM_P]))
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
                    isfair = fair(trace, num_p=conf.NUM_P, verbose=False)
                    # print("FAIR=",isfair)
                    
                    if board.fp_type==0:
                        # least fixpoint in cycle, OP win
                        if isfair:
                            r = 1 if self.curRole==-1 else -1
                            return [(x[0],x[3],x[2],x[4],r*((-1)**(x[1]!=self.curPlayer))) for x in trainExamples], 1 if r==self.curPlayer else 0
                    elif board.fp_type==1:
                        # greatest fixpoint in cycle, P win
                        r = 1 if self.curRole==1 else -1
                        return [(x[0],x[3],x[2],x[4],r*((-1)**(x[1]!=self.curPlayer))) for x in trainExamples], 1 if r==self.curPlayer else 0
                    else:
                        assert False


            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard,self.curPlayer,self.curRole, 
                temp=temp, 
                path=path, 
                on_path=on_path, 
                fp_count=fp_count, 
                trace=trace,
                changes=changes)

            action = np.random.choice(len(pi), p=pi)


            if canonicalBoard._f_type=='a'or canonicalBoard._f_type=='e':
                trace.append(action)
                changes[int(action)]+=1

            def one_hot(action):
                probs = [0]*game.getActionSize()
                probs[action]=1
                return probs

            sym = game.getSymmetries(canonicalBoard, pi)

            valids = game.getValidMoves(canonicalBoard, self.curPlayer)
            if np.sum(valids)>1:
            # if True:
                for b,p in sym:
                    # b[0].append(self.curPlayer)
                    b[0].append(self.curRole)
                    x = np.array(changes)
                    # x = x/np.sum(x)
                    x = (x-np.min(x))/10
                    b[0] = b[0]+x.tolist()

                    # # use advantage
                    # vec_cur = canonicalBoard.getStateRep()
                    # # vec_cur.append(self.curPlayer)
                    # vec_cur.append(self.curRole)
                    # _, v = self.nnet.predict(np.array([vec_cur]))

                    # s_tmp,p_tmp,r_tmp = game.getNextState(canonicalBoard, self.curPlayer, self.curRole, action)
                    # vec_tmp = s_tmp.getStateRep()
                    # # vec_tmp.append(p_tmp)
                    # vec_tmp.append(r_tmp)
                    # _, v_tmp = self.nnet.predict(np.array([vec_tmp]))
                    # r = game.getGameEnded(s_tmp, p_tmp, r_tmp)
                    # if r==1 or r==-1:
                    #     if p_tmp!=self.curPlayer:
                    #         adv = -v-r
                    #     else:
                    #         adv = -v+r
                    # else:
                    #     if p_tmp!=self.curPlayer:
                    #         adv = -v_tmp-v
                    #     else:
                    #         adv = v_tmp-v

                    # if self.curPlayer==1 and canonicalBoard.getStateRep()[1]==7 and canonicalBoard.getStateRep()[2]==7 and canonicalBoard.getStateRep()[3]==128:
                    #     self.p0 = pi
                    trainExamples.append([b, self.curPlayer, pi, 0, p])

            board, self.curPlayer, self.curRole = game.getNextState(board, self.curPlayer, self.curRole, action)
            assert self.curRole is not None
            r = game.getGameEnded(board, self.curPlayer, self.curRole, changes)
            # print(f"ADV={adv}")

            # if r!=-1:
            if r!=0:
                return [(x[0],x[3],x[2],x[4],r*((-1)**(x[1]!=self.curPlayer))) for x in trainExamples], 1 if r==self.curPlayer else 0

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        time0 = time.time()

        for i in range(1, self.args.numIters+1):
            # bookkeeping
            # print('------ITER ' + str(i) + '------')
            print(f'------ITER {i} ------ {time.time()-time0}')
            logging.warning(f"neural MCTS learning iteration {i}")
            # examples of the iteration
            game = copy.deepcopy(self.game)
            if not self.skipFirstSelfPlay or i>1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
    
                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()
                
                qtrace = {}
                vtrace = {}
                utrace = {}
                # self.mcts = MCTS(game, self.nnet, self.args, qtrace, vtrace, utrace)   # reset search tree
                logging.warning(f"Start Self-play")
                win_count=0
                for eps in range(self.args.numEps):
                    # Warm Start to speed up learning! Very useful!
                    self.mcts = MCTS(game, self.nnet, self.args, qtrace, vtrace, utrace)   # reset search tree
                    t,r = self.executeEpisode()
                    iterationTrainExamples += t
                    win_count += 1 if r==1 else 0
    
                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s Wins: {wc}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg, wc=win_count)
                    #
                    # logging.warning('({eps}/{maxeps}) Eps Time: {et:.3f}s'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg))
                    bar.next()
                print("wins=",win_count)
                bar.finish()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)
                
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                # self.trainExamplesHistory.pop(0)
                self.trainExamplesHistory = []
                self.trainExamplesHistory.append(iterationTrainExamples)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            # self.saveTrainExamples(i-1)
            
            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(game, self.pnet, self.args, {}, {}, {})
            logging.warning(f"Start Training")
            self.nnet.train(trainExamples)
            nmcts = MCTS(game, self.nnet, self.args, {}, {}, {})

            print('PITTING AGAINST PREVIOUS VERSION')
            logging.warning(f"Start Evaluation")
            def act_pi(mcts,x,p,r,path,on_path,fp_count,trace,changes, reset=False):
                pi = mcts.getActionProb(x,p,r, temp=1, path=path, on_path=on_path, fp_count=fp_count, trace=trace, changes=changes, reset=reset)
                curRole = x.getRole() if x.getRole() is not None else r
                stateVec = x.getStateRep()
                stateVec.append(curRole)

                return (np.argmax(pi),pi,0)

            arena = Arena(lambda x,p,r,path,on_path,fp_count,trace,changes, reset: act_pi(pmcts,x,p,r,path,on_path,fp_count,trace,changes,reset),
                          lambda x,p,r,path,on_path,fp_count,trace,changes, reset: act_pi(nmcts,x,p,r,path,on_path,fp_count,trace,changes,reset), 
                          game, 
                          lambda b,p:print(p,b.getStateRep(),b.f2str(expand=False)))

            pwins, nwins, draws, stop = arena.playGames(self.args.arenaCompare, verbose=True)

            print('ACCEPTING NEW MODEL')
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

        return self.args.numIters              

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile+".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True

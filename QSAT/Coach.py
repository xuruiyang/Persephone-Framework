from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
import sys
import matplotlib.pyplot as plt
import networkx as nx

from qsat.QSATGame import QSATGame as Game


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, nnet, nnet2, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.nnet2 = nnet2
        self.pnet2 = self.nnet2.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.nnet2, self.args)
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.trainExamplesHistory2= []
        self.skipFirstSelfPlay = False # can be overriden in loadTrainExamples()
        np.set_printoptions(threshold=sys.maxsize)

        self.sacc = 0

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
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        trainExamples2 = []
        board = self.game.getInitBoard()
        # the first player must be existantial player!
        self.curPlayer = 1
        episodeStep = 0
        win_count = 0

        print()

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board,self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            variable = board.variables[sorted(board.variables.keys())[0]][0]

            pi = self.mcts.getActionProb(canonicalBoard, self.curPlayer, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)

            for b,p in sym:
                if self.curPlayer==1:
                    trainExamples.append([b[:-2], self.curPlayer, [p], None])
                else:
                    trainExamples2.append([b[:-2], self.curPlayer, [p], None])
            action = np.random.choice(len(pi),p=pi)

            if self.curPlayer==1:
                print('var:{},val:{},pred_pi_v:{} '.format(variable,action,self.mcts.nnet.predict(canonicalBoard)[0:2]),end="")
            else:
                print('var:{},val:{},pred_pi_v:{} '.format(variable,action,self.mcts.nnet2.predict(canonicalBoard)[0:2]),end="")

            print(pi)

            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(self.game.getCanonicalForm(board,self.curPlayer), self.curPlayer)

            if r!=0:
                self.sacc = len(self.mcts.state)
                self.mcts.state = set()
                print('result:{}'.format(r))
                if r==1:    # P1 win
                    win_count += 1
                    return [(x[0],x[2],r*((-1)**(x[1]!=1))) for x in trainExamples], win_count, [(x[0],x[2],r*((-1)**(x[1]!=1))) for x in trainExamples2]
                else:       # P2 win
                    return [(x[0],x[2],r*((-1)**(x[1]!=1))) for x in trainExamples], win_count, [(x[0],x[2],r*((-1)**(x[1]!=1))) for x in trainExamples2]


    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters+1):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i>1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                iterationTrainExamples2 = deque([], maxlen=self.args.maxlenOfQueue)
    
                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()
                win_count = 0

                self.mcts = MCTS(self.game, self.nnet, self.nnet2, self.args)   # reset search tree
    
                for eps in range(self.args.numEps):
                    self.sacc = 0
                    examples,_win,examples2 = self.executeEpisode()
                    iterationTrainExamples += examples
                    iterationTrainExamples2 += examples2
                    win_count += _win
    
                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:} | StatesAcc: {sacc:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,
                                                                                                               total=bar.elapsed_td, eta=bar.eta_td, sacc=self.sacc)
                    bar.next()
                bar.finish()
                print('win_count = {}'.format(win_count))

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)
                self.trainExamplesHistory2.append(iterationTrainExamples2)
                
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            
            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            print(len(trainExamples))

            if len(self.trainExamplesHistory2) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory2), " => remove the oldest trainExamples")
                self.trainExamplesHistory2.pop(0)
            # SAVE NOT WORKING FOR DUAL
            # self.saveTrainExamples(i-1)
            
            # shuffle examples before training
            trainExamples2 = []
            for e in self.trainExamplesHistory2:
                trainExamples2.extend(e)
            shuffle(trainExamples2)

            print(len(trainExamples2))
            
            self.nnet.train(trainExamples)
            self.nnet2.train(trainExamples2)

            test_g = Game('qsat/qdimacs_test3.txt')

            nmcts = MCTS(test_g, self.nnet, self.nnet2, self.args)

            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best1.pth.tar')
            self.nnet2.save_checkpoint(folder=self.args.checkpoint, filename='best2.pth.tar')

            print('PITTING AGAINST PREVIOUS VERSION')
            # continue

            arena = Arena(lambda x,player: np.argmax(nmcts.getActionProb(x, player, temp=0)),
                          lambda x,player: np.argmax(nmcts.getActionProb(x, player, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare, False)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))             

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.' + '.tar'

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
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = False

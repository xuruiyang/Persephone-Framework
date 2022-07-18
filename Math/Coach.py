from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle

import matplotlib.pyplot as plt
import logging

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
        self.pnet2 = self.nnet2.__class__(self.game, normal_size=False)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.nnet2, self.args, {}, {}, {})
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.trainExamplesHistory2 = []
        self.skipFirstSelfPlay = False    # can be overriden in loadTrainExamples()

        self.flipper = {}
        self.flipper_record = {}

    def executeEpisode(self, main_player):
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
        trainExamples = []
        trainExamples2 = []
        board,player1,player2 = self.game.getInitBoard()
        if board.getActions() is None:
            print("Formula trivially true, terminate")
            exit()
        self.curRole = board.getRole()
        self.curPlayer = player1.name if player1.getRoleName() == self.curRole else player2.name
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board,self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)
            pi = self.mcts.getActionProb(canonicalBoard,self.curPlayer,self.curRole, temp=1, main_player=main_player)

            action = np.random.choice(len(pi), p=pi)

            def one_hot(action):
                probs = [0]*self.game.getActionSize()
                probs[action]=1
                return probs

            sym = self.game.getSymmetries(canonicalBoard, pi)

            for b,p in sym:
                b[0].append(self.curPlayer)

                # use advantage
                vec_cur = canonicalBoard.getStateRep()
                vec_cur.append(self.curPlayer)
                if self.curPlayer==1:
                    _, v, _ = self.nnet.predict(np.array([vec_cur]))
                else:
                    ps, v, _ = self.nnet2.predict(np.array([vec_cur]))
                    valids = self.game.getValidMoves(canonicalBoard, self.curPlayer)
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

                valids = self.game.getValidMoves(canonicalBoard, self.curPlayer)
                s_tmp,p_tmp,r_tmp = self.game.getNextState(canonicalBoard, self.curPlayer, self.curRole, action)
                vec_tmp = s_tmp.getStateRep()
                vec_tmp.append(p_tmp)
                # if p_tmp==1:
                if main_player==1:
                    _, v_tmp, _ = self.nnet.predict(np.array([vec_tmp]))
                else:
                    _, v_tmp, _ = self.nnet2.predict(np.array([vec_tmp]))
                r = self.game.getGameEnded(s_tmp, p_tmp, r_tmp)
                # print(f"v={v},v'={v_tmp},r={r}")
                if r==1 or r==-1:
                    if p_tmp!=self.curPlayer:
                        adv = -v-r
                    else:
                        adv = -v+r
                else:
                    if p_tmp!=self.curPlayer:
                        adv = -v_tmp-v
                    else:
                        adv = v_tmp-v

                if self.curPlayer==1:
                    trainExamples.append([b, self.curPlayer, pi, adv, p])
                    trainExamples2.append([b, self.curPlayer, pi, adv, p])
                else:
                    trainExamples.append([b, self.curPlayer, pi, adv, p])
                    trainExamples2.append([b, self.curPlayer, pi, adv, p])

            old_board = board.getStateRep()
            old_player = self.curPlayer
            old_role = self.curRole

            board, self.curPlayer, self.curRole = self.game.getNextState(board, self.curPlayer, self.curRole, action)

            r = self.game.getGameEnded(board, self.curPlayer, self.curRole)

            if r==1 or r==-1:
                return self.curPlayer, r if self.curPlayer==main_player else -r,[(x[0],x[3],x[2],x[4],r*((-1)**(x[1]!=self.curPlayer)),0 if int(x[0][0][2])%int(x[0][0][1]+1)==0 else 1) for x in trainExamples], [(x[0],x[3],x[2],x[4],r*((-1)**(x[1]!=self.curPlayer)),0 if int(x[0][0][2])%int(x[0][0][1]+1)==0 else 1) for x in trainExamples2]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        main_player=1
        p1_win_count=0
        p2_win_count=0
        ite=0
        time0 = time.time()
        for i in range(0, self.args.numIters):
            # bookkeeping
            print(f'------ITER {i}------{time.time()-time0}')
            # print(self.flipper_record)
            ite+=1
            print(p1_win_count,p2_win_count,ite)
            logging.warning(f"neural MCTS learning iteration {i}")
            # examples of the iteration
            if not self.skipFirstSelfPlay or i>1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                iterationTrainExamples2 = deque([], maxlen=self.args.maxlenOfQueue)
    
                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()
                
                qtrace = {}
                vtrace = {}
                utrace = {}

                logging.warning(f"Start Self-play")
                if p1_win_count<=0 and main_player==1:
                    main_player=1 
                elif p1_win_count>0 and main_player==1:
                    main_player=-1
                    p1_win_count=0
                    print('reset search tree')
                    ite=0
                    self.mcts = MCTS(self.game, self.nnet, self.nnet2, self.args, qtrace, vtrace, utrace)   # reset search tree
                    self.trainExamplesHistory=[]
                    self.trainExamplesHistory2=[]
                elif p2_win_count<=0 and main_player==-1:
                    main_player=-1
                elif p2_win_count>0 and main_player==-1:
                    main_player=1
                    p2_win_count=0
                    print('reset search tree')
                    ite=0
                    self.mcts = MCTS(self.game, self.nnet, self.nnet2, self.args, qtrace, vtrace, utrace)   # reset search tree
                    self.trainExamplesHistory=[]
                    self.trainExamplesHistory2=[]
                else:
                    assert False
                print(f'main_player:{main_player}')
                win_count = 0
                for eps in range(self.args.numEps):
                    # Warm Start to speed up learning! Very useful!
                    # self.mcts = MCTS(self.game, self.nnet, self.nnet2, self.args, qtrace, vtrace, utrace)   # reset search tree
                    last_player,r,t1,t2 = self.executeEpisode(main_player)
                    win_count += 1 if r==1 else 0

                    iterationTrainExamples += t1
                    iterationTrainExamples2 += t2

                    qtrace = self.mcts.Qtrace
                    vtrace = self.mcts.Vtrace
                    utrace = self.mcts.Utrace
                    ntrace = self.mcts.nQsa
    
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s Wins: {win_c}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg, win_c=win_count)

                    bar.next()
                bar.finish()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)
                self.trainExamplesHistory2.append(iterationTrainExamples2)
                
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory=[self.trainExamplesHistory.pop(-1)]
                self.mcts = MCTS(self.game, self.nnet, self.nnet2, self.args, qtrace, vtrace, utrace)   # reset search tree
            if len(self.trainExamplesHistory2) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory2) =", len(self.trainExamplesHistory2), " => remove the oldest trainExamples")
                self.trainExamplesHistory2=[self.trainExamplesHistory2.pop(-1)]
                self.mcts = MCTS(self.game, self.nnet, self.nnet2, self.args, qtrace, vtrace, utrace)   # reset search tree
            
            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            trainExamples2 = []
            for e in self.trainExamplesHistory2:
                trainExamples2.extend(e)
            shuffle(trainExamples2)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.nnet2.save_checkpoint(folder=self.args.checkpoint, filename='temp2.pth.tar')
            self.pnet2.load_checkpoint(folder=self.args.checkpoint, filename='temp2.pth.tar')

            pmcts = MCTS(self.game, self.pnet, self.pnet2, self.args, {}, {}, {})

            if main_player==1:
            # if True:
                logging.warning(f"Start Training NN1")
                self.nnet.train(trainExamples)
            else:
                logging.warning(f"Start Training NN2")
                self.nnet2.train(trainExamples2)

            nmcts = MCTS(self.game, self.nnet, self.nnet2, self.args, {}, {}, {})

            print('PITTING AGAINST PREVIOUS VERSION')
            logging.warning(f"Start Evaluation")
            arena = Arena(lambda x,p,r: nmcts.getActionProb(x,p,r, temp=1, main_player=main_player),
                          lambda x,p,r: nmcts.getActionProb(x,p,r, temp=1, main_player=main_player), self.game, lambda b,p:print(b.getStateRep()+[p]),nmcts,main_player)
            mainWin,fixWin = arena.playGames(self.args.arenaCompare, verbose=True)

            nmcts = MCTS(self.game, self.nnet, self.nnet2, self.args, {}, {}, {})
            if main_player==1:
                if mainWin==1:
                    p1_win_count += 1
                else:
                    p1_win_count = 0
            else:
                if mainWin==1:
                    p2_win_count += 1
                else:
                    p2_win_count = 0              

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

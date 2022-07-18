import math
import numpy as np
import copy
from collections import defaultdict
from utils import fair
import conf
EPS = 1e-8

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args, Qtrace, Vtrace, Utrace):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, player, role, temp=1, path=[], on_path={}, fp_count={}, trace=[], changes=[], reset=False):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        if reset:
            self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
            self.Nsa = {}       # stores #times edge s,a was visited
            self.Ns = {}        # stores #times board s was visited
            self.Ps = {}        # stores initial policy (returned by neural net)

            self.Es = {}        # stores game.getGameEnded ended for board s
            self.Vs = {}        # stores game.getValidMoves for board s

        self.primaryPlayer = player
        game = copy.deepcopy(self.game)
        player1,player2 = game.alignPlayers(player,role)
        curRole = canonicalBoard.getRole() if canonicalBoard.getRole() is not None else role
        curPlayer = player1.name if player1.getRoleName() == curRole else player2.name

        valids = game.getValidMoves(canonicalBoard, curPlayer)
        depths=[]
        if np.sum(valids)>1:
        # if True:
            for i in range(self.args.numMCTSSims):
                # t = copy.deepcopy(trace)
                t = []
                if len(t)>0 and not isinstance(t[-1],int):
                    t.pop()
                depth=[0]
                self.search(canonicalBoard, curPlayer ,curRole, 
                    isRootNode=False, 
                    path=copy.deepcopy(path), 
                    on_path=copy.deepcopy(on_path), 
                    fp_count=copy.deepcopy(fp_count), 
                    trace=t, 
                    depth=depth,
                    changes=copy.deepcopy(changes))
                depths.append(depth[0])

            s = game.stringRepresentation(canonicalBoard, curPlayer)
            ss = canonicalBoard.getStateRep()[1:conf.NUM_P+1]
            ss = str([ss[0]]+(ss[1:conf.NUM_P]))
            s_count = fp_count[ss]
            counts = [self.Nsa[((s,s_count),a)] if ((s,s_count),a) in self.Nsa else 0 for a in range(game.getActionSize())]
        else:
            counts = [0 for _ in range(game.getActionSize())]

        counts = counts*valids
        counts = counts+valids
        if np.sum(counts)==0:
            counts = counts + valids

        def one_hot(counts,valids):
            counts = counts*valids
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs

        if temp==0:
            return one_hot(counts,valids)

        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            return np.exp(x) / np.sum(np.exp(x), axis=0)

        counts = [x**(1./temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x/counts_sum for x in counts]
        return probs


    def search(self, canonicalBoard, curPlayer, curRole, isRootNode=False, path=[], on_path={}, fp_count={}, trace=[], depth=[0], changes=[]):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        depth[0]+=1
        game = copy.deepcopy(self.game)

        s = game.stringRepresentation(canonicalBoard, curPlayer)
        ss = canonicalBoard.getStateRep()[1:conf.NUM_P+1]
        ss = str([ss[0]]+(ss[1:conf.NUM_P]))
        # ss = str(ss[0])
        if canonicalBoard._f_type=='a'or canonicalBoard._f_type=='e':
            if len(trace)==0:
                trace.append(0)
            if not isinstance(trace[-1],int):
                trace.pop()
            trace.append(canonicalBoard.getStateRep()[1:conf.NUM_P+1])
        GAMMA=conf.GAMMA
        BOUNDED = conf.BOUNDED

        if BOUNDED and canonicalBoard._f_type=='pred' and canonicalBoard.fp_type is not None:
            if not on_path[ss]:
                path.append(ss)
                on_path[ss] = True
            else:
                while path[-1]!=ss:
                    on_path[path[-1]] = False
                    fp_count[path.pop()]=0
            fp_count[ss] += 1
            if fp_count[ss] > GAMMA:
                # print("LOOP DETECTED!")
                isfair = fair(trace, num_p=conf.NUM_P, verbose=False)

                if canonicalBoard.fp_type==0:
                    # least fixpoint in cycle, OP win
                    r = -(1 if curRole==-1 else -1)
                    r = r if isfair else -r
                elif canonicalBoard.fp_type==1:
                    # greatest fixpoint in cycle, P win
                    r = -(1 if curRole==1 else -1)
                else:
                    assert False
                return r


        s_count = fp_count[ss]
        if (s,s_count) not in self.Es:
            self.Es[(s,s_count)] = game.getGameEnded(canonicalBoard, curPlayer, curRole, changes)
        if self.Es[(s,s_count)]!=0:
            return -self.Es[(s,s_count)]
        
        if canonicalBoard.getActions() is None:
            print(self.Es[(s,s_count)])
        assert(canonicalBoard.getActions() is not None)

        valids = game.getValidMoves(canonicalBoard, curPlayer)
        if np.sum(valids)<2:
        # if False:
            assert np.sum(valids)!=0
            a = np.argmax(valids)
            if canonicalBoard._f_type=='a'or canonicalBoard._f_type=='e':
                assert len(trace)>0
                trace.append(a)
            next_s, next_player, next_role = game.getNextState(canonicalBoard, curPlayer, curRole, a)
            v = self.search(next_s, next_player, next_role, 
                path=copy.deepcopy(path), 
                on_path=copy.deepcopy(on_path), 
                fp_count=copy.deepcopy(fp_count), 
                trace=copy.deepcopy(trace), 
                depth=depth,
                changes=copy.deepcopy(changes))
            v = v if next_player!=curPlayer else -v
            return -v
        else:
            if (s,s_count) not in self.Ps:
                # leaf node
                stateVec = canonicalBoard.getStateRep()
                stateVec.append(curRole)
                x = np.array(changes)
                # x = x/np.sum(x)
                x = (x-np.min(x))/10
                stateVec = stateVec + x.tolist()
                self.Ps[(s,s_count)], v = self.nnet.predict(np.array([stateVec]))

                valids = game.getValidMoves(canonicalBoard, curPlayer)

                self.Ps[(s,s_count)] = self.Ps[(s,s_count)]*valids      # masking invalid moves
                sum_Ps_s = np.sum(self.Ps[(s,s_count)])
                if sum_Ps_s > 0:
                    self.Ps[(s,s_count)] /= sum_Ps_s    # renormalize
                else:
                    # if all valid moves were masked make all valid moves equally probable
                    
                    # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                    # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                    print("All valid moves were masked for player {}, do workaround.".format(curPlayer))
                    self.Ps[(s,s_count)] = self.Ps[(s,s_count)] + valids
                    self.Ps[(s,s_count)] /= np.sum(self.Ps[(s,s_count)])

                self.Vs[(s,s_count)] = valids
                self.Ns[(s,s_count)] = 0
                return -v

            valids = self.Vs[(s,s_count)]
            cur_best = -float('inf')
            best_act = -1

            if isRootNode:
                filter_valids = list(filter(lambda x: x>0,valids))
                l = len(filter_valids)
                noise = np.random.dirichlet([1] * l)
            vidx = 0

            # pick the action with the highest upper confidence bound'
            for a in range(game.getActionSize()):
                if valids[a] and isRootNode:
                    prior = 0.75*self.Ps[(s,s_count)][a] + 0.25*noise[vidx]
                    vidx += 1
                elif valids[a]:
                    prior = self.Ps[(s,s_count)][a]
                if valids[a]:
                    if ((s,s_count),a) in self.Qsa:
                        u = self.Qsa[((s,s_count),a)] + self.args.cpuct*prior*math.sqrt(self.Ns[(s,s_count)])/(1+self.Nsa[((s,s_count),a)])
                    else:
                        u = self.args.cpuct*prior*math.sqrt(self.Ns[(s,s_count)] + EPS) 

                    if u > cur_best:
                        cur_best = u
                        best_act = a

            a = best_act
            if np.sum(valids)<2:
                print("action=",a)
                assert a==0
            next_s, next_player, next_role = game.getNextState(canonicalBoard, curPlayer, curRole, a)
            if canonicalBoard._f_type=='a'or canonicalBoard._f_type=='e':
                assert len(trace)>0
                trace.append(a)
                changes[int(a)] += 1

            v = self.search(next_s, next_player, next_role, 
                path=copy.deepcopy(path), 
                on_path=copy.deepcopy(on_path), 
                fp_count=copy.deepcopy(fp_count), 
                trace=copy.deepcopy(trace), 
                depth=depth,
                changes=copy.deepcopy(changes))
            v = v if next_player!=curPlayer else -v 

            if ((s,s_count),a) in self.Qsa:
                self.Qsa[((s,s_count),a)] = (self.Nsa[((s,s_count),a)]*self.Qsa[((s,s_count),a)] + v)/(self.Nsa[((s,s_count),a)]+1)
                self.Nsa[((s,s_count),a)] += 1
            else:
                self.Qsa[((s,s_count),a)] = v
                self.Nsa[((s,s_count),a)] = 1

            self.Ns[(s,s_count)] += 1
            return -v

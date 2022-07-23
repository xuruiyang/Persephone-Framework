import math
import numpy as np
EPS = 1e-8

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, nnet2, args, Qtrace, Vtrace, Utrace):
        self.game = game
        self.nnet = nnet
        self.nnet2 = nnet2
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

        self.Qtrace = Qtrace
        self.Vtrace = Vtrace
        self.Utrace = Utrace

        self.nQsa = {}
        self.v0 = None

    def getRandActionProb(self, canonicalBoard, player, role, temp=1, main_player=1):
        player1,player2 = self.game.alignPlayers(player,role)
        curRole = canonicalBoard.getRole()
        curPlayer = player1.name if player1.getRoleName() == curRole else player2.name
        valids = self.game.getValidMoves(canonicalBoard, curPlayer)
        counts = [1 for a in range(self.game.getActionSize())]
        counts = counts*valids
        counts_sum = float(sum(counts))
        probs = [x/counts_sum for x in counts]
        return probs


    def getActionProb(self, canonicalBoard, player, role, temp=1, main_player=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        def one_hot(counts,valids):
            counts = counts*valids
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            # bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs

        self.primaryPlayer = player 
        player1,player2 = self.game.alignPlayers(player,role)
        curRole = canonicalBoard.getRole()
        curPlayer = player1.name if player1.getRoleName() == curRole else player2.name
        if curPlayer!=main_player:
            stateVec = canonicalBoard.getStateRep()
            stateVec.append(curPlayer)
            if curPlayer==1:
                ps, v, _ = self.nnet.predict(np.array([stateVec]))
            else:
                ps, v, _ = self.nnet2.predict(np.array([stateVec]))
            valids = self.game.getValidMoves(canonicalBoard, curPlayer)
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
            return ps
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard, curPlayer ,curRole, main_player)

        s = self.game.stringRepresentation(canonicalBoard, curPlayer)
        valids = self.game.getValidMoves(canonicalBoard, curPlayer)
        counts = [self.Nsa[(s,a)]+1 if (s,a) in self.Nsa else 1 for a in range(self.game.getActionSize())]

        counts = counts*valids

        if temp==0:
            return one_hot(counts,valids)

        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            return np.exp(x) / np.sum(np.exp(x), axis=0)
        counts = [x**(1./temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x/counts_sum for x in counts]
        return probs


    def search(self, canonicalBoard, curPlayer, curRole, main_player):
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

        s = self.game.stringRepresentation(canonicalBoard, curPlayer)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, curPlayer, curRole)
        if self.Es[s]==1 or self.Es[s]==-1:
            # terminal node
            return -self.Es[s]
        
        if canonicalBoard.getActions() is None:
            print(self.Es[s])
        assert(canonicalBoard.getActions() is not None)

        if s not in self.Ps:
            # leaf node
            stateVec = canonicalBoard.getStateRep()
            stateVec.append(curPlayer)

            # IDEA: use opponent's neural policy but main_player's value estimate
            if curPlayer==1:
                self.Ps[s], _, _ = self.nnet.predict(np.array([stateVec]))
            else:
                self.Ps[s], _, _ = self.nnet2.predict(np.array([stateVec]))
            if main_player==1:
                _, v, _ = self.nnet.predict(np.array([stateVec]))
            else:
                _, v, _ = self.nnet2.predict(np.array([stateVec]))

            valids = self.game.getValidMoves(canonicalBoard, curPlayer)
            # use pre generated Q values? similar to SAVE!
            for a in range(self.game.getActionSize()):
                if valids[a]:
                    s_tmp,p_tmp,r_tmp = self.game.getNextState(canonicalBoard, curPlayer, curRole, a)
                    v_tmp = s_tmp.getStateRep()
                    v_tmp.append(p_tmp)
                    if p_tmp==1:
                        _, q, _ = self.nnet.predict(np.array([v_tmp]))
                    else:
                        _, q, _ = self.nnet2.predict(np.array([v_tmp]))
                    r = self.game.getGameEnded(s_tmp, p_tmp, r_tmp)
                    if r==0:
                        self.Qsa[(s,a)] = -q if curPlayer!=p_tmp else q
                    else:
                        self.Qsa[(s,a)] = -r if curPlayer!=p_tmp else r
                    self.Nsa[(s,a)] = 1

            self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                
                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked for player {}, do workaround.".format(curPlayer))
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound'
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if curPlayer==main_player:
                # if True:
                    if (s,a) in self.Qsa:
                        u = self.Qsa[(s,a)] + self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                    else:
                        u = self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)
                else:
                    # force to follow fixed policy
                    if a==np.argmax(self.Ps[s]):
                        u = float('inf')
                    else:
                        u = -float('inf')

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act

        next_s, next_player, next_role = self.game.getNextState(canonicalBoard, curPlayer, curRole, a)

        v = self.search(next_s, next_player, next_role, main_player)

        v = v if next_player!=curPlayer else -v 

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1
        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return -v

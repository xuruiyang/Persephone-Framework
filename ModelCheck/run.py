import persephone.PersephoneMu as Persephone
import sys
from antlr4 import *
from antlr4 import TokenStreamRewriter
from parser_gen.persephoneLexer import persephoneLexer
from parser_gen.persephoneParser import persephoneParser
from parser_gen.persephoneListener import persephoneListener


from Coach import Coach
import copy
from persephone.PersephoneGame import PersephoneGame as Game
from persephone.keras.NNet import NNetWrapper as nn
from utils import *

from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf

from ts.TransitionSystem import HSRTranSys, DinePhiTranSys

args = dotdict({
    'numIters': 1000000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 100000,        #
    'updateThreshold': 0.0,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 5,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 2,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp','best.pth.tar'),
    # 'load_folder_file': ('C://Users//ruiya//alpha-zero-general//temp','8_0.pth.tar'),
    'numItersForTrainExamplesHistory': 10,

})

class WalkListener(persephoneListener):

    def __init__(self, rewriter):
        self.rewriter = rewriter
        super().__init__()

    def exitPredicate_definition(self, ctx:persephoneParser.Predicate_definitionContext):
        print("GENERATING PREDICATE:")
        name = ctx.PRED_CONST().getText()
        print("NAME",name)
        params = [v.getText() for v in ctx.VAR()]
        print("PARAMS",params)
        # body = ctx.formula().getText()
        sub_ctx = ctx.formula()
        start_idx = sub_ctx.start.tokenIndex
        stop_idx = sub_ctx.stop.tokenIndex
        body = self.rewriter.getText("default",start_idx,stop_idx)
        print("BODY",body)
        print()
        Persephone.generatePredicate(name,params,body)

    def exitGfp_definition(self, ctx:persephoneParser.Predicate_definitionContext):
        print("GENERATING GFP:")
        name = ctx.PRED_CONST().getText()
        print("NAME",name)
        params = [v.getText() for v in ctx.VAR()]
        print("PARAMS",params)
        # body = ctx.formula().getText()
        sub_ctx = ctx.formula()
        start_idx = sub_ctx.start.tokenIndex
        stop_idx = sub_ctx.stop.tokenIndex
        body = self.rewriter.getText("default",start_idx,stop_idx)
        print("BODY",body)
        print()
        Persephone.generateGFP(name,params,body)

    def exitLfp_definition(self, ctx:persephoneParser.Predicate_definitionContext):
        print("GENERATING LFP:")
        name = ctx.PRED_CONST().getText()
        print("NAME",name)
        params = [v.getText() for v in ctx.VAR()]
        print("PARAMS",params)
        # body = ctx.formula().getText()
        sub_ctx = ctx.formula()
        start_idx = sub_ctx.start.tokenIndex
        stop_idx = sub_ctx.stop.tokenIndex
        body = self.rewriter.getText("default",start_idx,stop_idx)
        print("BODY",body)
        print()
        Persephone.generateLFP(name,params,body)

    def enterGame_play(self, ctx:persephoneParser.Game_playContext):
        print("STATING THE GAME:")
        name = "Persephone."+ctx.PRED_CONST().getText()
        params = [v.getText() for v in ctx.term()]
        p_str = "("
        for v in params:
            p_str+=v+","
        p_str = p_str[:-1]+")"
        f = None
        f=eval(name+p_str)
        # game = SemanticGame(g,Player(Role_P(),'P1'),Player(Role_OP(),'P2'))
        # game.play()

        g = Game(6)
        g.init_state = f

        with tf.device("cpu:0"):
            nnet = nn(g)
            if args.load_model:
                nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
            g = Game(6)
            f=eval(name+p_str)
            g.init_state = f
            c = Coach(g, nnet, args)
            i = c.learn()


    def enterAlg_bin_op(self, ctx:persephoneParser.Alg_bin_opContext):
        token = ctx.start
        if ctx.getText()=="And":
            self.rewriter.replaceSingleToken(token,"Alg_And")
        elif ctx.getText()=="Or":
            self.rewriter.replaceSingleToken(token,"Alg_Or")
        else:
            print("ERROR:",token.getText())
            assert False

    def enterFormula(self, ctx:persephoneParser.FormulaContext):
        token = ctx.start
        if ctx.getText()=="TRUE":
            self.rewriter.replaceSingleToken(token,"TrueConst()")
        elif ctx.getText()=="FALSE":
            self.rewriter.replaceSingleToken(token,"FalseConst()")
        
def main(argv):
    input_stream = FileStream(argv[1])
    lexer = persephoneLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = persephoneParser(stream)
    rewriter = TokenStreamRewriter.TokenStreamRewriter(stream)
    tree = parser.main()

    walkListener = WalkListener(rewriter)
    walker = ParseTreeWalker()
    walker.walk(walkListener, tree)
 
if __name__ == '__main__':
    main(sys.argv)
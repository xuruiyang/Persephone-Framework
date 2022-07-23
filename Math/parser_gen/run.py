from Persephone import *
import sys
from antlr4 import *
from python_experiment.persephoneLexer import persephoneLexer
from python_experiment.persephoneParser import persephoneParser
from python_experiment.persephoneListener import persephoneListener

class TestListener(persephoneListener):
    def enterPredicate_definition(self, ctx:persephoneParser.Predicate_definitionContext):
        print("PREDICATE")
        name = ctx.PRED_CONST().getText()
        print("NAME",name)
        params = [v.getText() for v in ctx.VAR()]
        print("PARAMS",params)
        body = ctx.formula().getText()
        print("BODY",body)
        generatePredicate(name,params,body)

    def enterGame_play(self, ctx:persephoneParser.Game_playContext):
        exec(ctx.getText())

def main(argv):
    input_stream = FileStream(argv[1])
    lexer = persephoneLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = persephoneParser(stream)
    tree = parser.main()

    testListener = TestListener()
    walker = ParseTreeWalker()
    walker.walk(testListener, tree)
 
if __name__ == '__main__':
    main(sys.argv)
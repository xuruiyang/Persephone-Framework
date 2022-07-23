import sys
from antlr4 import *
from persephoneLexer import persephoneLexer
from persephoneParser import persephoneParser
from persephoneListener import persephoneListener

class TestListener(persephoneListener):
    def enterPredicate_definition(self, ctx:persephoneParser.Predicate_definitionContext):
        print("PREDICATE")
        name = ctx.PRED_CONST().getText()
        print("NAME",name)
        params = str([v.getText() for v in ctx.VAR()])
        print("PARAMS",params)
        body = ctx.formula().getText()
        print("BODY",body)
        print("generatePredicate('{}',{},'{}')".format(name,params,body))
 
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
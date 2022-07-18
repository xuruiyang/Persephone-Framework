from Coach import Coach
from qsat.QSATGame import QSATGame as Game
from qsat.tensorflow.NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 10,
    'numEps': 100,
    'tempThreshold': 1000,
    'updateThreshold': 0.1,
    'maxlenOfQueue': 1000,
    'numMCTSSims': 20,
    'arenaCompare': 2,
    'cpuct': 1.2,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'load_folder_file2': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 5,

})

if __name__=="__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    g = Game('qsat/generated/false_examples/f_out_9.qdimacs')
    nnet = nn(g)
    nnet2 = nn(g,identity=2)

    if args.load_model:
        nnet.load_checkpoint('./trained_f_out_0/', 'best1.pth.tar')
        nnet2.load_checkpoint('./trained_f_out_0/', 'best2.pth.tar')

    c = Coach(g, nnet, nnet2, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()

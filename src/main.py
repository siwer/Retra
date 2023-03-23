'''
Script to re-run the experiments from the paper "RETRA: Recurrent Transformers for Learning Temporally Contextualized Knowledge Graph Embeddings"
Warning
!!! Not every combination is possible !!!
Settings from the paper are defined in the experiments*.sh files
'''
import models
import trainer
import argparse

def get_parameter():
    parser = argparse.ArgumentParser()
    #determine the network settings
    parser.add_argument('-gpuNr', default=0, type=int, help='Which GPU to use')
    parser.add_argument('-nrEntities', default=7128, type=int, help='Nr. of entities (ICEWS = 7128, SUMO = 149')
    parser.add_argument('-nrRelations', default=230, type=int, help='Nr. of relations')
    parser.add_argument('-embeddingDim', default=50, type=int, help='Dimensionality of embeddings')
    parser.add_argument('-nrHeads', default=1, type=int, help='Nr. of attention heads')
    parser.add_argument('-nrLayers', default=1, type=int, help='Nr. of encoder layers')
    parser.add_argument('-forwardDim', default=50, type=int, help='Dimensionality of feed-forward network in transformer')
    parser.add_argument('-dropout', default=0.2, type=float, help='Dropout in transformer')
    parser.add_argument('-context', default='False', type=str, help='Use context in ICEWS (True or False)')
    parser.add_argument('-train', default='', type=str, help='Path to training set')
    parser.add_argument('-test', default='', type=str, help='Path to test set')
    parser.add_argument('-valid', default='', type=str, help='Path to validation set')
    parser.add_argument('-nrEpochs', default=10, type=int, help='Nr. of epochs')
    parser.add_argument('-learningRate', default=0.005, type=float, help='Learning rate')
    parser.add_argument('-batchSize', default=256, type=int, help='Batch size in non-recurrent setting')
    #determine which experiments to run
    parser.add_argument('-data', default='ICEWS', type=str, help='ICEWS or SUMO')
    parser.add_argument('-recurrency', default='True', type=str, help='True or False')
    parser.add_argument('-scoring', default='transE', type=str, help='transE, holE, simplE')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_parameter()
    context = bool
    if args.context == 'True':
        context = True
    else:
        context = False

    if args.data == 'ICEWS':
        if args.scoring == 'transE':
            model = models.RetraTransE(args.gpuNr,args.nrEntities,args.nrRelations,args.embeddingDim,args.nrHeads,args.nrLayers,args.forwardDim,args.dropout,context)
            trainingInstance = trainer.NonRecurrentICEWS(args.train,args.valid,args.test,args.nrEpochs,args.learningRate,args.batchSize)
            trainingInstance.trainModel(model)
            trainingInstance.testModel(model)
        elif args.scoring == 'holE':
            model = models.RetraHolE(args.gpuNr,args.nrEntities,args.nrRelations,args.embeddingDim,args.nrHeads,args.nrLayers,args.forwardDim,args.dropout,context)
            trainingInstance = trainer.NonRecurrentICEWS(args.train,args.valid,args.test,args.nrEpochs,args.learningRate,args.batchSize)
            trainingInstance.trainModel(model)
            trainingInstance.testModel(model)
        elif args.scoring == 'simplE':
            model = models.RetraSimplE(args.gpuNr,args.nrEntities,args.nrRelations,args.embeddingDim,args.nrHeads,args.nrLayers,args.forwardDim,args.dropout,context)
            trainingInstance = trainer.NonRecurrentICEWS(args.train,args.valid,args.test,args.nrEpochs,args.learningRate,args.batchSize)
            trainingInstance.trainModel(model)
            trainingInstance.testModel(model)
    
    elif args.data == 'SUMO':
        if args.recurrency == 'True':
            model = models.RecurrentSUMO(args.gpuNr,args.nrEntities,args.embeddingDim,args.nrHeads,args.nrLayers,args.forwardDim,args.dropout)
            trainingInstance = trainer.RecurrentSUMO(args.train,args.valid,args.test,args.nrEpochs,args.learningRate,'adam')
            trainingInstance.trainModel(model)
            trainingInstance.testModel(model)
        elif args.recurrency == 'False':
            model = models.RetraSUMO(args.gpuNr,args.nrEntities,args.embeddingDim,args.nrHeads,args.nrLayers,args.forwardDim,args.dropout)
            trainingInstance = trainer.NonRecurrentSUMO(args.train,args.valid,args.test,args.nrEpochs,args.learningRate,args.batchSize)
            trainingInstance.trainModel(model)
            trainingInstance.testModel(model)
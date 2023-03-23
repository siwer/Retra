#!/bin/bash

python3 main.py -nrEntities 149 -nrHeads 5 -nrLayers 5 -forwardDim 10 -learningRate 0.00005 -nrEpochs 100 -context False -train data/sumoNon/sumoLargeTrain.data -test data/sumoNon/sumoLargeTest.data -valid data/sumoNon/sumoLargeValid.data -recurrency False -data SUMO -batchSize 2048

python3 main.py -nrEntities 149 -nrHeads 5 -nrLayers 5 -forwardDim 10 -learningRate 0.00005 -nrEpochs 100 -context False -train data/sumoRec/sumoTrain5_2048.data -test data/sumoRec/sumoTest5_2048.data -valid data/sumoRec/sumoValid5_2048.data -recurrency True -data SUMO
python3 main.py -nrEntities 149 -nrHeads 5 -nrLayers 5 -forwardDim 10 -learningRate 0.00005 -nrEpochs 100 -context False -train data/sumoRec/sumoTrain10_2048.data -test data/sumoRec/sumoTest5_2048.data -valid data/sumoRec/sumoValid10_2048.data -recurrency True -data SUMO
python3 main.py -nrEntities 149 -nrHeads 5 -nrLayers 5 -forwardDim 10 -learningRate 0.00005 -nrEpochs 100 -context False -train data/sumoRec/sumoTrain15_2048.data -test data/sumoRec/sumoTest15_2048.data -valid data/sumoRec/sumoValid15_2048.data -recurrency True -data SUMO
python3 main.py -nrEntities 149 -nrHeads 5 -nrLayers 5 -forwardDim 10 -learningRate 0.00005 -nrEpochs 100 -context False -train data/sumoRec/sumoTrain20_2048.data -test data/sumoRec/sumoTest20_2048.data -valid data/sumoRec/sumoValid20_2048.data -recurrency True -data SUMO

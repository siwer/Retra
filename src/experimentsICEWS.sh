#!/bin/bash

python3 main.py -learningRate 0.008 -nrEpochs 15 -batchSize 256 -gpuNr 0 -scoring transE -context False -train ../data/icewsNon/icewsTrain.data -test ../data/icewsNon/icewsTest.data -valid ../data/icewsNon/icewsValid.data -recurrency False
python3 main.py -learningRate 0.008 -nrEpochs 15 -batchSize 256 -gpuNr 0 -scoring simplE -context False -train ../data/icewsNon/icewsTrain.data -test ../data/icewsNon/icewsTest.data -valid ../data/icewsNon/icewsValid.data -recurrency False
python3 main.py -learningRate 0.008 -nrEpochs 15 -batchSize 256 -gpuNr 0 -scoring holE -context False -train ../data/icewsNon/icewsTrain.data -test ../data/icewsNon/icewsTest.data -valid ../data/icewsNon/icewsValid.data -recurrency False

python3 main.py -learningRate 0.008 -nrEpochs 15 -batchSize 256 -gpuNr 0 -scoring transE -context True -train ../data/icewsNon/icewsTrain.data -test ../data/icewsNon/icewsTest.data -valid ../data/icewsNon/icewsValid.data -recurrency False
python3 main.py -learningRate 0.008 -nrEpochs 15 -batchSize 256 -gpuNr 0 -scoring simplE -context True -train ../data/icewsNon/icewsTrain.data -test ../data/icewsNon/icewsTest.data -valid ../data/icewsNon/icewsValid.data -recurrency False
python3 main.py -learningRate 0.008 -nrEpochs 15 -batchSize 256 -gpuNr 0 -scoring holE -context True -train ../data/icewsNon/icewsTrain.data -test ../data/icewsNon/icewsTest.data -valid ../data/icewsNon/icewsValid.data -recurrency False
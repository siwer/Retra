# Retra - Driving Scene Dataset
This repository contains the SUMO driving scene data that is used for the Driving Scene Classification task in the ESWC 2021 paper: *"RETRA: Recurrent Transformers for Learning Temporally Contextualized Knowledge Graph Embeddings"* (https://openreview.net/pdf?id=l7fvWxQ3RG). Upon internal approval, we will add the code for the experimental section of the RETRA paper.

As of now, this repo contains the unprocessed driving scene Knowledge Graph described in the ESWC 2021 paper *"Towards a Knowledge Graph-based Approach for Situation Comprehension in Driving Scenarios"* (https://openreview.net/pdf?id=XBWVwf4lab8) and the processed data that was used in the RETRA paper.

The raw data consists of three .nt files (test, rain and validation data) that contain ~ 4 Million triples. Based on this data, the processed folder contains three .csv files (tab separated) with the same data in different form, with each row representing a conflict (with a unique conflictID). Note that for the second dataset, all floating point numbers were rounded to the next Integer.
Due to restricted maximum file size on Github, the raw data is contained in a compressed archive.

# Model training parameters
As mentioned in our Paper (Section 6), we present the parameters that were used to achieve the results in the respective tasks. The presented durations include both training and testing. 
 - Location Recommendation
    - Adagrad optimizer, learning rate = 0.09, time needed ~ 12 - 14 hours (due to batchsize)
    - Batchsize = 1, trained for 10 epochs, embedding dimension = 130, forward dimension = 130
    - 1 encoder layer, 1 attention head
 - Driving Scene Classification
    - Adam optimizer, learning rate = 0.00005, time needed ~ 5 to 10 minutes
    - Batchsize = 2048, trained for 100 epochs, embedding dimension = 50, forward dimension = 10
    - 5 encoder layers, 5 attention heads
 - Event Prediction
    - Adam optimizer, learning rate = 0.008, time needed ~ 10 to 15 minutes
    - Batchsize = 256, trained for 15 epochs, embedding dimension = 50, forward dimension = 50
    - 1 encoder layer, 1 attention head

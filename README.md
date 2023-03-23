# Retra - Driving Scene Dataset
This repository contains the SUMO driving scene data that is used for the Driving Scene Classification task in the ESWC 2021 paper: *"RETRA: Recurrent Transformers for Learning Temporally Contextualized Knowledge Graph Embeddings"* (https://openreview.net/pdf?id=l7fvWxQ3RG).

This repo contains the unprocessed driving scene Knowledge Graph described in the ESWC 2021 paper *"Towards a Knowledge Graph-based Approach for Situation Comprehension in Driving Scenarios"* (https://openreview.net/pdf?id=XBWVwf4lab8) and the processed data that was used in the RETRA paper.

The raw data consists of three .nt files (test, rain and validation data) that contain ~ 4 Million triples. Based on this data, the processed folder contains three .csv files (tab separated) with the same data in different form, with each row representing a conflict (with a unique conflictID). Note that for the second dataset, all floating point numbers were rounded to the next Integer.
Due to restricted maximum file size on Github, the raw data is contained in a compressed archive.

# Retra - Code for the experiments

The spec-file.txt can be used to recreate the conda environment under which all experiments were conducted

All data is provided in preprocessed from:
- put into the needed data structure
- for sumo, all numbers were rounded to integers
- in both datasets, all values were replaced by indices to save space

The experimental settings are saved in the .sh files, which contain all the reported experiments on ICEWS and SUMO datasets

To log the data, the console output can be piped to a logfile by adding "|& tee -a logFile.txt" to the end of the lines (without quotation marks)

To save the models, the saveModel Parameter must be set to true for the alle settings in the inits in trainer.py
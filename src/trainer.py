'''
Provides the training, testing and evaluation protocoll for the different models
Note that for the recurrent case, the batchsize is provided withing the dataset
'''
import torch
import torch.nn as nn
import torch.optim as optim
import dataset as ds
from torch.utils.data import DataLoader

class TrainerInit:
    def __init__(self,trainData,validationData,testData,nrEpochs,learningRate,saveModel):
        #data is provided as String containing the path to the file
        self.trainData = trainData
        self.validationData = validationData
        self.testData = testData
        self.nrEpochs = nrEpochs
        self.learningRate = learningRate
        self.saveModel = saveModel

class RecurrentICEWS(TrainerInit):

    def __init__(self,trainData,validationData,testData,nrEpochs,learningRate,scheduler):
        super(RecurrentICEWS,self).__init__(trainData,validationData,testData,nrEpochs,learningRate,False)
        self.scheduler = scheduler

    def trainModel(self,model):
        print ('Parameters:\n' + 'Epochs: ' + str(self.nrEpochs) + '\tLearningRate: ' + str(self.learningRate) + '\n' + 'Training Data: ' + self.trainData + '\tTest Data: ' + self.testData + '\tValidation Data: ' + self.validationData)
        lossFunction = nn.CrossEntropyLoss()
        trainData = ds.retraDataset(self.trainData)
        dataloader = DataLoader(trainData,batch_size=1,num_workers=0)
        print ('~~~TRAINING~~~')
        if self.scheduler == 'adam':
            optimizer = optim.Adam(model.parameters(),lr = self.learningRate)
        elif self.scheduler == 'adadelta':
            optimizer = optim.Adadelta(model.parameters(),lr = self.learningRate)
        elif self.scheduler == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr = self.learningRate)
        epochCount = 1
        model.train()
        for x in range(self.nrEpochs):
            model.zero_grad()
            absLoss = 0
            checkinNr = 0
            for sequence in enumerate(dataloader): 
                #init counter for feedback
                #taking the actucal data of one user
                checkins = sequence[1][0][0].to(model.device)
                targets = sequence[1][1][0].to(model.device)
                #none variable for new user
                contextualizedSubject = torch.zeros(checkins.shape[2],model.embeddingDimension).to(model.device)
                #print (checkins.shape)

                if (len(checkins) == len(targets)):
                    for i in range(len(checkins)-1):
                        checkin = checkins[i].clone()
                        nextCheckin = checkins[i+1].clone()
                        #forward step
                        scores, userC = model(checkin,nextCheckin,contextualizedSubject)
                        contextualizedSubject = userC
                        #writing the variable to use as input in next step
                        loss = lossFunction(scores,targets[i+1])
                        loss.backward(retain_graph=True)
                        optimizer.step()
                        #cast to float to prevent loss from keeping all gradients
                        absLoss += float(loss)
                        checkinNr += 1
                        #print(float(loss))
            if epochCount % (self.nrEpochs/10) == 0:
                print ('Loss in Epoch ' + str(x) + ': ' + str(absLoss/checkinNr))
                self.validateModel(model)
            epochCount += 1
        if self.saveModel:
            torch.save(model,str(model.gpuNr) + '_retra.model')

    def validateModel(self,model):
        lossFunction = nn.CrossEntropyLoss()
        validata = ds.retraDataset(self.validationData)
        dataloader = DataLoader(validata,batch_size=1,num_workers=0)
        model.eval()
        absLoss = 0
        model.zero_grad()
        with torch.no_grad():
            checkinNr = 0
            for sequence in enumerate(dataloader): 
                checkins = sequence[1][0][0].to(model.device)
                targets = sequence[1][1][0].to(model.device)
                contextualizedSubject = torch.zeros(checkins.shape[2],model.embeddingDimension).to(model.device)

                if (len(checkins) == len(targets)):
                    for i in range(len(checkins)-1):
                        checkin = checkins[i].clone()
                        nextCheckin = checkins[i+1].clone()
                        scores, userC = model(checkin,nextCheckin,contextualizedSubject)
                        contextualizedSubject = userC
                        loss = lossFunction(scores,targets[i+1])
                        absLoss += float(loss)
                        checkinNr += 1
            print ('\tValidation Loss: ' +str(absLoss/checkinNr))

    def testModel(self,model):
        testData = ds.retraDataset(self.testData)
        print ('~~~EVALUATION~~~')
        dataloader = DataLoader(testData,batch_size = 1, num_workers=1)
        model.eval()
        hits1 = 0
        hits3 = 0
        hits10 = 0
        hits100 = 0
        ranks = 0
        rRanks = 0
        checkinCounter = 0
        for sequence in enumerate(dataloader):
            with torch.no_grad():
                checkins = sequence[1][0][0].to(model.device)
                targets = sequence[1][1][0].to(model.device)
                contextualizedUser = torch.zeros(checkins.shape[2],model.embeddingDimension).to(model.device)
                if (len(checkins) == len(targets)):
                    for i in range(len(checkins)-1):
                        checkin = checkins[i].clone()
                        nextCheckin = checkins[i+1].clone()
                        scores, userC = model(checkin,nextCheckin,contextualizedUser)
                        contextualizedUser = userC
                        for j in range(len(scores)):
                            checkinCounter +=1
                            rank = (int((torch.topk(scores[j],len(scores[j])).indices == targets[i+1][j]).nonzero().flatten())) + 1
                            ranks += rank
                            rRanks += 1/rank
                            if targets[i+1][j] in torch.topk(scores[j],1).indices:
                                hits1 += 1
                            if targets[i+1][j] in torch.topk(scores[j],3).indices:
                                hits3 += 1
                            if targets[i+1][j] in torch.topk(scores[j],10).indices:
                                hits10 += 1
                            if targets[i+1][j] in torch.topk(scores[j],100).indices:
                                hits100 += 1
        print ('Checkins:\t' + str(checkinCounter))
        print ('Hits@1:\t' + str(hits1/checkinCounter) + '\tAbsolut: ' + str(hits1))
        print ('Hits@3:\t' + str(hits3/checkinCounter) + '\tAbsolut: ' + str(hits3))
        print ('Hits@10:\t' + str(hits10/checkinCounter) + '\tAbsolut: ' + str(hits10))
        print ('Hits@100:\t' + str(hits100/checkinCounter) + '\tAbsolut: ' + str(hits100))
        print ('MR:\t' + str(ranks/checkinCounter))
        print ('MRR:\t' + str(rRanks/checkinCounter))

class NonRecurrentICEWS(TrainerInit):

    def __init__(self,trainData,validationData,testData,nrEpochs,learningRate,batchSize):
        super(NonRecurrentICEWS,self).__init__(trainData,validationData,testData,nrEpochs,learningRate,False)
        self.batchSize = batchSize

    def trainModel(self,model):
        print ('Parameters:\n' + 'Epochs: ' + str(self.nrEpochs) + '\tLearningRate: ' + str(self.learningRate) + '\n' + 'Training Data: ' + self.trainData + '\tTest Data: ' + self.testData + '\tValidation Data: ' + self.validationData)
        print ("~~~TRAINING~~~")
        optimizer = optim.Adagrad(model.parameters(), lr = self.learningRate)
        lossFunction = nn.CrossEntropyLoss()
        data = torch.load(self.trainData)
        bestLoss = 100
        bestEpoch = 0
        model.train()
        for j in range(1,self.nrEpochs+1):
            optimizer.zero_grad()
            nrBatches = 0
            lossAbs = 0
            for i in range(0,len(data),self.batchSize):
                #select batch of data
                tmp = torch.stack(data[i:i+self.batchSize]).transpose(0,1).to(model.device)
                target = tmp[2]
                out = model(tmp)
                loss = lossFunction(out,target)
                nrBatches += 1
                lossAbs += float(loss)
                loss.backward()
                optimizer.step()
            if j % (self.nrEpochs/10) == 0:
                print("Loss in Epoch #"+str(j)+": "+str(lossAbs/nrBatches))
                valLoss = self.validateModel(model)
                print('Validation Loss:\t\t'+str(valLoss))
                if bestLoss > valLoss:
                    bestLoss = valLoss
                    bestEpoch = j
        print ("Best loss: "+str(bestLoss)+" in Epoch: "+str(bestEpoch))

    def validateModel(self,model):
        lossFunction = nn.CrossEntropyLoss()
        data = torch.load(self.validationData)
        model.eval()
        nrBatches = 0
        lossAbs = 0
        with torch.no_grad():
            for i in range(0,len(data),self.batchSize):
                tmp = torch.stack(data[i:i+self.batchSize]).transpose(0,1).to(model.device)
                target = tmp[2]
                out = model(tmp)
                loss = lossFunction(out,target)
                nrBatches += 1
                lossAbs += float(loss)
            return lossAbs/nrBatches

    def testModel(self,model):
        print("~~~EVALUATION~~~")
        data = torch.load(self.testData)
        model.eval()
        hits1 = 0
        hits3 = 0
        hits10 = 0
        checkinCounter = 0
        hits100 = 0
        ranks = 0
        rRanks = 0
        with torch.no_grad():
            for i in range(len(data)):
                checkinCounter +=1
                tmp = data[i].unsqueeze(0).transpose(0,1).to(model.device)
                target = data[i][2].to(model.device)
                out = model(tmp)
                rank = int((torch.topk(out.squeeze(),len(out.squeeze())).indices == target).nonzero().flatten()) + 1
                ranks += rank
                rRanks += 1/rank
                if target in torch.topk(out.squeeze(),1).indices:
                    hits1 += 1
                if target in torch.topk(out.squeeze(),3).indices:
                    hits3 += 1
                if target in torch.topk(out.squeeze(),10).indices:
                    hits10 += 1
                if target in torch.topk(out.squeeze(),100).indices:
                    hits100 += 1
        print ("Hits@1: " + str(hits1/checkinCounter))
        print ("Hits@3: " + str(hits3/checkinCounter))
        print ("Hits@10: " + str(hits10/checkinCounter))
        print ("Hits@100: " + str(hits100/checkinCounter))
        print ("MR: " + str(ranks/checkinCounter))
        print ("MRR: " + str(rRanks/checkinCounter) + '\n')

class RecurrentSUMO(TrainerInit):

    def __init__(self,trainData,validationData,testData,nrEpochs,learningRate,scheduler):
        super(RecurrentSUMO,self).__init__(trainData,validationData,testData,nrEpochs,learningRate,False)
        self.scheduler = scheduler

    def trainModel(self,model):
        print ('Parameters:\n' + 'Epochs: ' + str(self.nrEpochs) + '\tLearningRate: ' + str(self.learningRate) + '\n' + 'Training Data: ' + self.trainData + '\tTest Data: ' + self.testData + '\tValidation Data: ' + self.validationData)
        print ("~~~TRAINING~~~")
        lossFunction = nn.CrossEntropyLoss()
        trainData = ds.retraDataset(self.trainData)
        dataloader = DataLoader(trainData,1,num_workers=1)
        if self.scheduler == 'adam':
            optimizer = optim.Adam(model.parameters(),lr = self.learningRate)
        elif self.scheduler == 'adadelta':
            optimizer = optim.Adadelta(model.parameters(),lr = self.learningRate)
        elif self.scheduler == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr = self.learningRate)
        model.train()
        for x in range(1,self.nrEpochs+1):
            model.zero_grad()
            absLoss = 0
            checkinNr = 0
            for sequence in enumerate(dataloader): 
                #taking the actucal data of one user
                checkins = sequence[1][0][0].to(model.device)
                targets = sequence[1][1][0].to(model.device)
                oldSituation = torch.zeros(checkins.shape[2],model.embeddingDimension).to(model.device)

                if (len(checkins) == len(targets)):
                    for i in range(len(checkins)-1):
                        checkin = checkins[i].clone()
                        #forward step
                        scores, newSituation = model(checkin,oldSituation)
                        oldSituation = newSituation
                        loss = lossFunction(scores,targets[i])
                        loss.backward(retain_graph=True)
                        optimizer.step()
                        #cast to float to prevent loss from keeping all gradients
                        absLoss += float(loss)
                        checkinNr += 1
            if x % (self.nrEpochs/10) == 0:
                print ('Loss in Epoch ' + str(x) + ': ' + str(absLoss/checkinNr))
                self.validateModel(model)
        if self.saveModel:
            torch.save(model,str(model.gpuNr) + '_retra.model')

    def validateModel(self,model):
        lossFunction = nn.CrossEntropyLoss()
        validata = ds.retraDataset(self.validationData)
        dataloader = DataLoader(validata,batch_size=1,num_workers=0)
        model.eval()
        absLoss = 0
        model.zero_grad()
        with torch.no_grad():
            checkinNr = 0
            for sequence in enumerate(dataloader): 
                checkins = sequence[1][0][0].to(model.device)
                oldSituation = torch.ones(checkins.shape[2],model.embeddingDimension).to(model.device)
                targets = sequence[1][1][0].to(model.device)

                if (len(checkins) == len(targets)):
                    for i in range(len(checkins)-1):
                        checkin = checkins[i].clone()
                        scores, newSituation = model(checkin,oldSituation)
                        oldSituation = newSituation
                        loss = lossFunction(scores,targets[i])
                        absLoss += float(loss)
                        checkinNr += 1
            print ('\tValidation Loss: ' +str(absLoss/checkinNr))

    def testModel(self,model):
        print ('~~~EVALUATION~~~')
        testData = ds.retraDataset(self.testData)
        dataloader = DataLoader(testData,batch_size = 1, num_workers=1)
        model.eval()
        hits1 = 0
        hits3 = 0
        hits5 = 0
        ranks = 0
        rRanks = 0
        checkinCounter = 0
        for sequence in enumerate(dataloader):
            with torch.no_grad():
                checkins = sequence[1][0][0].to(model.device)
                oldSituation = torch.ones(checkins.shape[2],model.embeddingDimension).to(model.device)
                #shape [sequence length, context, batch size, dimension]
                targets = sequence[1][1][0].to(model.device)
                #shape [sequence length,batch size]
                if (len(checkins) == len(targets)):
                    for i in range(len(checkins)-1):
                        checkin = checkins[i].clone()
                        scores, newSituation = model(checkin,oldSituation)
                        oldSituation = newSituation
                        #scores = scores.transpose(1,0)
                        for j in range(len(scores)):
                            checkinCounter +=1
                            rank = (int((torch.topk(scores[j],len(scores[j])).indices == targets[i][j]).nonzero().flatten())) + 1
                            ranks += rank
                            rRanks += 1/rank
                            if targets[i][j] in torch.topk(scores[j],1).indices:
                                hits1 += 1
                            if targets[i][j] in torch.topk(scores[j],3).indices:
                                hits3 += 1
                            if targets[i][j] in torch.topk(scores[j],5).indices:
                                hits5 += 1
        print ("Hits@1: " + str(hits1/checkinCounter))
        print ("Hits@3: " + str(hits3/checkinCounter))
        print ("Hits@5: " + str(hits5/checkinCounter))
        print ("MR: " + str(ranks/checkinCounter))
        print ("MRR: " + str(rRanks/checkinCounter) + '\n')

class NonRecurrentSUMO(TrainerInit):

    def __init__(self,trainData,validationData,testData,nrEpochs,learningRate,batchSize):
        super(NonRecurrentSUMO,self).__init__(trainData,validationData,testData,nrEpochs,learningRate,False)
        self.batchSize = batchSize

    def trainModel(self,model):
        print ("~~~TRAINING~~~")
        optimizer = optim.Adam(model.parameters(),self.learningRate)
        lossFunction = nn.CrossEntropyLoss()
        data = torch.load(self.trainData)
        bestLoss = 100
        bestEpoch = 0
        model.train()
        for j in range(1,self.nrEpochs+1):
            optimizer.zero_grad()
            nrBatches = 0
            lossAbs = 0
            for i in range(0,len(data),self.batchSize):
                tmp = torch.stack(data[i:i+self.batchSize]).transpose(0,1).to(model.device)
                inputTensor = torch.LongTensor(8,len(tmp[1])).to(model.device)
                inputTensor[0] = tmp[1]
                inputTensor[1] = tmp[2]
                inputTensor[2] = tmp[3]
                inputTensor[3] = tmp[4]
                inputTensor[4] = tmp[5]
                inputTensor[5] = tmp[6]
                inputTensor[6] = tmp[7]
                inputTensor[7] = tmp[8]
                
                target = tmp[0]
                out = model(inputTensor)
                loss = lossFunction(out,target)
                nrBatches += 1
                lossAbs += float(loss)
                loss.backward()
                optimizer.step()
            if j % (self.nrEpochs/10) == 0:
                print("Loss in Epoch #"+str(j)+": "+str(lossAbs/nrBatches))
                valLoss = self.validateModel(model)
                print('Validation Loss:\t\t'+str(valLoss))
                if bestLoss > valLoss:
                    bestLoss = valLoss
                    bestEpoch = j
        print ("Best loss: "+str(bestLoss)+" in Epoch: "+str(bestEpoch))

    def validateModel(self,model):
        lossFunction = nn.CrossEntropyLoss()
        data = torch.load(self.validationData)
        model.eval()
        nrBatches = 0
        lossAbs = 0
        for i in range(0,len(data),self.batchSize):
            tmp = torch.stack(data[i:i+self.batchSize]).transpose(0,1).to(model.device)
            inputTensor = torch.LongTensor(8,len(tmp[1])).to(model.device)
            inputTensor[0] = tmp[1]
            inputTensor[1] = tmp[2]
            inputTensor[2] = tmp[3]
            inputTensor[3] = tmp[4]
            inputTensor[4] = tmp[5]
            inputTensor[5] = tmp[6]
            inputTensor[6] = tmp[7]
            inputTensor[7] = tmp[8]
                
            target = tmp[0]
            out = model(inputTensor)
            loss = lossFunction(out,target)
            nrBatches += 1
            lossAbs += float(loss)
        return lossAbs/nrBatches

    def testModel(self,model):
        print("~~~EVALUATION~~~")
        data = torch.load(self.testData)
        model.eval()
        hits1 = 0
        hits3 = 0
        hits5 = 0
        checkinCounter = 0
        ranks = 0
        rRanks = 0
        with torch.no_grad():
            for i in range(len(data)):
                checkinCounter +=1
                inputTensor = torch.LongTensor(8,1).to(model.device)
                inputTensor[0] = data[i][1]
                inputTensor[1] = data[i][2]
                inputTensor[2] = data[i][3]
                inputTensor[3] = data[i][4]
                inputTensor[4] = data[i][5]
                inputTensor[5] = data[i][6]
                inputTensor[6] = data[i][7]
                inputTensor[7] = data[i][8]
                
                target = data[i][0].to(model.device)
                out = model(inputTensor)
                rank = int((torch.topk(out.squeeze(),len(out.squeeze())).indices == target).nonzero().flatten()) + 1
                ranks += rank
                rRanks += 1/rank
                if target in torch.topk(out.squeeze(),1).indices:
                    hits1 += 1
                if target in torch.topk(out.squeeze(),3).indices:
                    hits3 += 1
                if target in torch.topk(out.squeeze(),5).indices:
                    hits5 += 1
        print ("Hits@1: " + str(hits1/checkinCounter))
        print ("Hits@3: " + str(hits3/checkinCounter))
        print ("Hits@5: " + str(hits5/checkinCounter))
        print ("MR: " + str(ranks/checkinCounter))
        print ("MRR: " + str(rRanks/checkinCounter) + '\n')
'''
Provides the model definitions, i.e. their forward pass and initialisation
The indices prodived in the forward functions are exactly set to the respective dataset which they are intended to use with
To use different datasets, both the forward pass and the training/test/eval from trainer.py need to be adapted
'''
import torch
import torch.nn as nn
import torch.optim as optim
import holescore as hs

#super class for initializing models for ICEWS and SUMO experiments
class RetraInit(nn.Module):

    def __init__(self,gpuNr,nrEntities,nrRelations,embeddingDimension,nrAttentionHeads,nrEncoderLayers,forwardDimension,dropout,context):
        super(RetraInit,self).__init__()
        #class variables
        self.gpuNr = gpuNr
        self.nrEntities = nrEntities
        self.nrRelations = nrRelations
        self.embeddingDimension = embeddingDimension
        self.nrAttentionHeads = nrAttentionHeads
        self.nrEncoderLayers = nrEncoderLayers
        self.forwardDimension = forwardDimension
        self.dropout = dropout
        self.ctx = context

        self.device = torch.device('cuda:'+str(self.gpuNr) if torch.cuda.is_available() else 'cpu')
        #embedding layers
        self.entityEmbedding = nn.Embedding(self.nrEntities,self.embeddingDimension).to(self.device)
        torch.nn.init.xavier_uniform_(self.entityEmbedding.weight)

        self.relationEmbedding = nn.Embedding(self.nrRelations,self.embeddingDimension).to(self.device)
        torch.nn.init.xavier_uniform_(self.relationEmbedding.weight)

        #only needed for ICEWS, therefore initialized with a fixed number (nr. of potential context entities)
        if self.ctx:
            self.contextEmbedding = nn.Embedding(2443,self.embeddingDimension).to(self.device)
            torch.nn.init.xavier_uniform_(self.contextEmbedding.weight)
        
        #transformer layers
        self.encoderLayer = nn.TransformerEncoderLayer(self.embeddingDimension,self.nrAttentionHeads,self.forwardDimension,dropout=self.dropout).to(self.device)
        self.transformerEnc = nn.TransformerEncoder(self.encoderLayer,self.nrEncoderLayers).to(self.device)

'''
THE RECURRENT RUNS ON ICEWS WERE NOT REPORTED IN THE PAPER
'''
class RecurrentHolE(RetraInit):

    def __init__(self,gpuNr,nrEntities,nrRelations,embeddingDimension,nrAttentionHeads,nrEncoderLayers,forwardDimension,dropout,context):
        super(RecurrentHolE,self).__init__(gpuNr,nrEntities,nrRelations,embeddingDimension,nrAttentionHeads,nrEncoderLayers,forwardDimension,dropout,context)
    
    def forward(self, input, nextCheckin, contextualisedSubject):
        #get the vector representations
        head = self.entityEmbedding(input[0])
        tail = self.entityEmbedding(input[2])
        relation = self.relationEmbedding(input[1])

        nextRelation = self.relationEmbedding(nextCheckin[1])
        #use the contextual information if needed
        if self.ctx:
            context = self.contextEmbedding(input[3:])
            contextualized = self.transformerEnc(torch.cat([head.unsqueeze(0),relation.unsqueeze(0),tail.unsqueeze(0),context,contextualisedSubject.unsqueeze(0)],0))
        else:
            contextualized = self.transformerEnc(torch.cat([head.unsqueeze(0),relation.unsqueeze(0),tail.unsqueeze(0),contextualisedSubject.unsqueeze(0)],0))
        #represents the contextualised subject
        subject = contextualized[0]
        #this variable will be passed on to the next step
        history = contextualized[-1]
        '''
        Calculates the scores according to the HolE scoring function for all possible tails (entityEmbedding.weight)
        This operation is very memory intensive; to save memory, it could be replaced by a for-loop over all entities
        The implementation of the scoring function is adapted from https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/module/model/HolE.py
        '''
        subject = torch.unsqueeze(subject,1).repeat(1,self.nrEntities,1)
        nextRelation = torch.unsqueeze(nextRelation,1).repeat(1,self.nrEntities,1)
        scores = hs.holeScore(subject,nextRelation,self.entityEmbedding.weight)
        #returns the recurrent variable and a distribution of scores over all entities
        return scores, history

class RecurrentTransE(RetraInit):

    def __init__(self,gpuNr,nrEntities,nrRelations,embeddingDimension,nrAttentionHeads,nrEncoderLayers,forwardDimension,dropout,context):
        super(RecurrentTransE,self).__init__(gpuNr,nrEntities,nrRelations,embeddingDimension,nrAttentionHeads,nrEncoderLayers,forwardDimension,dropout,context)
    
    def forward(self, input, nextCheckin, contextualisedSubject):
        head = self.entityEmbedding(input[0])
        tail = self.entityEmbedding(input[2])
        relation = self.relationEmbedding(input[1])

        nextRelation = self.relationEmbedding(nextCheckin[1])
        if self.ctx:
            context = self.contextEmbedding(input[3:])
            contextualized = self.transformerEnc(torch.cat([head.unsqueeze(0),relation.unsqueeze(0),tail.unsqueeze(0),context,contextualisedSubject.unsqueeze(0)],0))
        else:
            contextualized = self.transformerEnc(torch.cat([head.unsqueeze(0),relation.unsqueeze(0),tail.unsqueeze(0),contextualisedSubject.unsqueeze(0)],0))
        subject = contextualized[0]
        history = contextualized[-1]
        #TransE scoring function, adapted from https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/module/model/TransE.py
        subject = torch.unsqueeze(subject,1).repeat(1,self.nrEntities,1)
        nextRelation = torch.unsqueeze(nextRelation,1).repeat(1,self.nrEntities,1)
        scores = torch.norm(subject + (nextRelation - self.entityEmbedding.weight),1,-1)

        return scores, history

class RecurrentSimplE(RetraInit):

    def __init__(self,gpuNr,nrEntities,nrRelations,embeddingDimension,nrAttentionHeads,nrEncoderLayers,forwardDimension,dropout,context):
        super(RecurrentSimplE,self).__init__(gpuNr,nrEntities,nrRelations,embeddingDimension,nrAttentionHeads,nrEncoderLayers,forwardDimension,dropout,context)
        '''
        SimplE needs additional embedding layers, because it needs embeddings for the entity head, the entity tail, the relation and the inverse relation
        The layers from the super class are therefore used as the entity head and the relation embedding
        '''
        self.entTailEmbedding = nn.Embedding(self.nrEntities,self.embeddingDimension).to(self.device)
        torch.nn.init.xavier_uniform_(self.entTailEmbedding.weight)

        self.relInvEmbedding = nn.Embedding(self.nrRelations,self.embeddingDimension).to(self.device)
        torch.nn.init.xavier_uniform_(self.relInvEmbedding.weight)

    def forward(self, input, nextCheckin, contextualisedSubject):
        headhead = self.entityEmbedding(input[0])
        headtail = self.entTailEmbedding(input[0])
        tailhead = self.entityEmbedding(input[2])
        tailtail = self.entTailEmbedding(input[2])
        relation = self.relationEmbedding(input[1])
        relInv = self.relInvEmbedding(input[1])

        nextRelation = self.relationEmbedding(nextCheckin[1])
        nextInvRel = self.relInvEmbedding(nextCheckin[1])
        if self.ctx:
            context = self.contextEmbedding(input[3:])
            contextualized = self.transformerEnc(torch.cat([headhead.unsqueeze(0),headtail.unsqueeze(0),relation.unsqueeze(0),relInv.unsqueeze(0),tailhead.unsqueeze(0),tailtail.unsqueeze(0),context,contextualisedSubject.unsqueeze(0)],0))
        else:
            contextualized = self.transformerEnc(torch.cat([headhead.unsqueeze(0),headtail.unsqueeze(0),relation.unsqueeze(0),relInv.unsqueeze(0),tailhead.unsqueeze(0),tailtail.unsqueeze(0),contextualisedSubject.unsqueeze(0)],0))
        subjectHead = contextualized[0]
        subjectTail = contextualized[1]
        history = contextualized[-1]
        #This needs even more memory because of the additional tensors that are involved
        subjectHead = torch.unsqueeze(subjectHead,1).repeat(1,self.nrEntities,1)
        subjectTail = torch.unsqueeze(subjectTail,1).repeat(1,self.nrEntities,1)
        nextRelation = torch.unsqueeze(nextRelation,1).repeat(1,self.nrEntities,1)
        nextInvRel = torch.unsqueeze(nextInvRel,1).repeat(1,self.nrEntities,1)
        #Implementation adapted from https://github.com/baharefatemi/SimplE/blob/master/SimplE.py
        scores1 = torch.sum(subjectHead * nextRelation * self.entityEmbedding.weight,dim=2)
        scores2 = torch.sum(subjectTail * nextInvRel * self.entTailEmbedding.weight,dim=2)
        scores = torch.clamp((scores1+scores2)/2,-20,20)

        return scores, history

'''
THE FOLLOWING SETTINGS WERE USED TO PRODUCE THE NON-RECURRENT ICEWS RESULTS FROM THE PAPER
'''
class RetraHolE(RetraInit):

    def __init__(self,gpuNr,nrEntities,nrRelations,embeddingDimension,nrAttentionHeads,nrEncoderLayers,forwardDimension,dropout,context):
        super(RetraHolE,self).__init__(gpuNr,nrEntities,nrRelations,embeddingDimension,nrAttentionHeads,nrEncoderLayers,forwardDimension,dropout,context)
    
    def forward(self,input):
        head = self.entityEmbedding(input[0])
        relation = self.relationEmbedding(input[1])

        if self.ctx:
            context = self.contextEmbedding(input[3:4])
            context2 = self.contextEmbedding(input[6:])
            contextualized = self.transformerEnc(torch.cat([head.unsqueeze(0),relation.unsqueeze(0),context,context2],0))
        else:
            contextualized = self.transformerEnc(torch.cat([head.unsqueeze(0),relation.unsqueeze(0)],0))
        subject = contextualized[0]
        
        subject = torch.unsqueeze(subject,1).repeat(1,self.nrEntities,1)
        relation = torch.unsqueeze(relation,1).repeat(1,self.nrEntities,1)
        scores = hs.holeScore(subject,relation,self.entityEmbedding.weight)
        
        return scores

class RetraTransE(RetraInit):

    def __init__(self,gpuNr,nrEntities,nrRelations,embeddingDimension,nrAttentionHeads,nrEncoderLayers,forwardDimension,dropout,context):
        super(RetraTransE,self).__init__(gpuNr,nrEntities,nrRelations,embeddingDimension,nrAttentionHeads,nrEncoderLayers,forwardDimension,dropout,context)

    def forward(self, input):
        head = self.entityEmbedding(input[0])
        relation = self.relationEmbedding(input[1])
        if self.ctx:
            context = self.contextEmbedding(input[3:4])
            context2 = self.contextEmbedding(input[6:])
            contextualized = self.transformerEnc(torch.cat([head.unsqueeze(0),relation.unsqueeze(0),context,context2],0))
        else:
            contextualized = self.transformerEnc(torch.cat([head.unsqueeze(0),relation.unsqueeze(0)],0))
        subject = contextualized[0]
        
        subject = torch.unsqueeze(subject,1).repeat(1,self.nrEntities,1)
        relation = torch.unsqueeze(relation,1).repeat(1,self.nrEntities,1)
        scores = torch.norm(subject + (relation - self.entityEmbedding.weight),1,-1)
        return scores

class RetraSimplE(RetraInit):

    def __init__(self,gpuNr,nrEntities,nrRelations,embeddingDimension,nrAttentionHeads,nrEncoderLayers,forwardDimension,dropout,context):
        super(RetraSimplE,self).__init__(gpuNr,nrEntities,nrRelations,embeddingDimension,nrAttentionHeads,nrEncoderLayers,forwardDimension,dropout,context)

        self.entTailEmbedding = nn.Embedding(self.nrEntities,self.embeddingDimension).to(self.device)
        torch.nn.init.xavier_uniform_(self.entTailEmbedding.weight)

        self.relInvEmbedding = nn.Embedding(self.nrRelations,self.embeddingDimension).to(self.device)
        torch.nn.init.xavier_uniform_(self.relInvEmbedding.weight)
    
    def forward(self,input):
        headhead = self.entityEmbedding(input[0])
        headtail = self.entTailEmbedding(input[0])
        relation = self.relationEmbedding(input[1])
        relInv = self.relInvEmbedding(input[1])

        if self.ctx:
            context = self.contextEmbedding(input[3:4])
            context2 = self.contextEmbedding(input[6:])
            contextualized = self.transformerEnc(torch.cat([headhead.unsqueeze(0),headtail.unsqueeze(0),relation.unsqueeze(0),relInv.unsqueeze(0),context,context2],0))
        else:
            contextualized = self.transformerEnc(torch.cat([headhead.unsqueeze(0),headtail.unsqueeze(0),relation.unsqueeze(0),relInv.unsqueeze(0)],0))
        subjectHead = contextualized[0]
        subjectTail = contextualized[1]
        
        subjectHead = torch.unsqueeze(subjectHead,1).repeat(1,self.nrEntities,1)
        subjectTail = torch.unsqueeze(subjectTail,1).repeat(1,self.nrEntities,1)
        relation = torch.unsqueeze(relation,1).repeat(1,self.nrEntities,1)
        invRel = torch.unsqueeze(relInv,1).repeat(1,self.nrEntities,1)
        scores1 = torch.sum(subjectHead * relation * self.entityEmbedding.weight,dim=2)
        scores2 = torch.sum(subjectTail * invRel * self.entTailEmbedding.weight,dim=2)
        scores = torch.clamp((scores1+scores2)/2,-20,20)

        return scores

'''
THE FOLLOWING SETTINGS WERE USED TO PRODUCE THE SUMO RESULTS FROM THE PAPER
'''
class RetraInitSUMO(nn.Module):

    def __init__(self,gpuNr,nrEntities,embeddingDimension,nrAttentionHeads,nrEncoderLayers,forwardDimension,dropout):
        super(RetraInitSUMO,self).__init__()
        #class variables
        self.gpuNr = gpuNr
        self.nrEntities = nrEntities
        self.embeddingDimension = embeddingDimension
        self.nrAttentionHeads = nrAttentionHeads
        self.nrEncoderLayers = nrEncoderLayers
        self.forwardDimension = forwardDimension
        self.dropout = dropout

        self.device = torch.device('cuda:'+str(self.gpuNr) if torch.cuda.is_available() else 'cpu')

        self.entityEmbedding = nn.Embedding(self.nrEntities,self.embeddingDimension).to(self.device)
        torch.nn.init.xavier_uniform_(self.entityEmbedding.weight)

        self.encoderLayer = nn.TransformerEncoderLayer(self.embeddingDimension,self.nrAttentionHeads,self.forwardDimension,dropout=self.dropout).to(self.device)
        self.transformerEnc = nn.TransformerEncoder(self.encoderLayer,self.nrEncoderLayers).to(self.device)

        #Output is fixed at 5 because there are 5 possible outcome scenarios (=situations) in the dataset
        self.outLayer = torch.nn.Linear(self.embeddingDimension,5).to(self.device)
        self.relu = torch.nn.ReLU().to(self.device)
        self.drop = nn.Dropout(self.dropout).to(self.device)

class RecurrentSUMO(RetraInitSUMO):
    
    def __init__(self,gpuNr,nrEntities,embeddingDimension,nrAttentionHeads,nrEncoderLayers,forwardDimension,dropout):
        super(RecurrentSUMO,self).__init__(gpuNr,nrEntities,embeddingDimension,nrAttentionHeads,nrEncoderLayers,forwardDimension,dropout)
    
    #the implicit scene embedding is the recurrent variable
    def forward(self, input, oldScene):
        vecs = self.entityEmbedding(input)
        vecs = vecs.squeeze()
        vecs = torch.cat([vecs,oldScene.unsqueeze(0)])
        contextualized = self.transformerEnc(vecs)
        
        newScene = contextualized[-1]
        scores = self.outLayer(newScene)
        scores = self.relu(scores)
        scores = self.drop(scores)
        
        return scores, newScene

class RetraSUMO(RetraInitSUMO):

    def __init__(self,gpuNr,nrEntities,embeddingDimension,nrAttentionHeads,nrEncoderLayers,forwardDimension,dropout):
        super(RetraSUMO,self).__init__(gpuNr,nrEntities,embeddingDimension,nrAttentionHeads,nrEncoderLayers,forwardDimension,dropout)

    def forward(self, input):
        vecs = self.entityEmbedding(input)
        #represents the implicit scene embedding
        scene = torch.zeros(1,vecs.shape[1],vecs.shape[2]).to(self.device)
        vecs = torch.cat([vecs,scene],0)
        outcome = self.transformerEnc(vecs)

        scores = self.outLayer(outcome[-1])
        scores = self.relu(scores)
        scores = self.drop(scores)
        return scores

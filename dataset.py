import os
import shutil
import random
from natsort import natsorted
import pandas as pd
import statistics

import torch
from torch.utils.data import Dataset

class Walk():

    def __init__(self, data, startRow, endRow, label, filePath):

        self.data = data
        self.startRow = startRow
        self.endRow = endRow
        self.label = label
        self.originalLabel = label
        self.inferenceLabel = 'x'
        self.filePath = filePath
        return
    
    def sample(self, winLength):
        
        s = random.randint(self.startRow, self.endRow-winLength)
        e = s + winLength
        return self.data[s:e]
    
    def __str__(self):

        print("\tData shape: (%d,%d), Start-end: (%d,%d), Label: %d" %(
            self.data.shape[0], self.data.shape[1], self.startRow, self.endRow, self.label))
        return ""

class Condition():

    def __init__(self, SubjectId, folderPath, conditionName, featuresIdx, winLength):

        self.id = SubjectId
        self.folderPath = folderPath
        self.condition = conditionName
        self.featuresIdx = featuresIdx
        self.winLength = winLength

        # these attributes will be filled by the following functions
        self.rb1Data = None
        self.rb2Data = None
        self.labels1 = None
        self.labels2 = None
        self.paths = None

        # each condition has 12 walks objects
        self.walks = []

        # load the walks
        self._loadCSVs()
        self._setWalks()
        return
    
    def _setWalks(self):

        # find the labels of each walk during the condition
        int1 = self._findIntervals(self.labels1)
        int2 = self._findIntervals(self.labels2)

        for i,(s,e,l) in enumerate(int1):

            # we skip the first one (walking static over the same place)
            if i == 0: continue

            # we skip a walk if its len is less than winLength
            if e-s <= self.winLength: continue

            self.walks.append(Walk(data=self.rb1Data, startRow=s, endRow=e, label=l, 
                filePath = self.paths['labels1Path']))

        for i,(s,e,l) in enumerate(int2):
            if i == 0: continue
            if e-s <= self.winLength: continue
            self.walks.append(Walk(data=self.rb2Data, startRow=s, endRow=e, label=l,
                filePath = self.paths['labels2Path']))
        
        return
    
    def _findIntervals(self, tensor):

        intervals = []
        start = 0
        label = 0

        for i in range(len(tensor)):
            
            if tensor[i] != 0:  # Found a new nonzero value

                intervals.append((start, i, int(tensor[i].item())))
                start = i+1


        return intervals

    def _readCSV(self, path, labels=False):

        df = pd.read_csv(path, header=None, low_memory=False)
        df = df.iloc[1:, 1:]

        # this will only affect the test label files
        if labels:
            df.replace({'x': 10}, inplace=True)
            df.fillna(0, inplace=True)

        #data = torch.tensor(df.to_numpy(dtype=float), dtype=torch.float32)
        #return data
        return df

    
    def _loadCSVs(self):

        paths = self._createCSVPaths()

        rokoko1Data = self._readCSV(paths['rokoko1Path'])
        rokoko2Data = self._readCSV(paths['rokoko2Path'])

        self.labels1 = self._readCSV(paths['labels1Path'], labels=True)
        self.labels2 = self._readCSV(paths['labels2Path'], labels=True)

        self.labels1=torch.tensor(self.labels1.to_numpy(dtype=float), dtype=torch.float32)
        self.labels2=torch.tensor(self.labels2.to_numpy(dtype=float), dtype=torch.float32)

        bitalino1Data = self._readCSV(paths['bitalino1Path'])
        bitalino2Data = self._readCSV(paths['bitalino2Path'])

        rb1Data = pd.concat([rokoko1Data, bitalino1Data], axis=1)
        rb2Data = pd.concat([rokoko2Data, bitalino2Data], axis=1)
        rb1Data = torch.tensor(rb1Data.to_numpy(dtype=float), dtype=torch.float32)
        rb2Data = torch.tensor(rb2Data.to_numpy(dtype=float), dtype=torch.float32)

        #rb1Data = rokoko1Data
        #rb2Data = rokoko2Data

        self.rb1Data = rb1Data[:, self.featuresIdx]
        self.rb2Data = rb2Data[:, self.featuresIdx]

        if torch.any(torch.isnan(self.rb1Data)) or torch.any(torch.isnan(self.rb2Data)):
            print("NaNs found when reading the data")
            print(self.id, self.condition)

        assert(self.rb1Data.shape[0] == self.labels1.shape[0])
        assert(self.rb2Data.shape[0] == self.labels2.shape[0])
        self.paths = paths
        return
    
    def _createCSVPaths(self):

        # create rokoko file paths
        r1d = (self.id + '-' + self.condition + '-1_rokoko_data_unified.csv').lower()
        r1l = (self.id + '-' + self.condition + '-1_rokoko_label_unified.csv').lower()
        r2d = (self.id + '-' + self.condition + '-2_rokoko_data_unified.csv').lower()
        r2l = (self.id + '-' + self.condition + '-2_rokoko_label_unified.csv').lower()

        r1d_path = os.path.join(self.folderPath, self.id, 'Rokoko', 'CSV', r1d )
        r1l_path = os.path.join(self.folderPath, self.id, 'Rokoko', 'CSV',r1l )
        r2d_path = os.path.join(self.folderPath, self.id, 'Rokoko', 'CSV',r2d )
        r2l_path = os.path.join(self.folderPath, self.id, 'Rokoko', 'CSV',r2l )

        # create bitalino file paths
        b1d = (self.id + '-' + self.condition + '-1_data_unified.csv').lower()
        b1l = (self.id + '-' + self.condition + '-1_label_unified.csv').lower()
        b2d = (self.id + '-' + self.condition + '-2_data_unified.csv').lower()
        b2l = (self.id + '-' + self.condition + '-2_label_unified.csv').lower()

        b1d_path = os.path.join(self.folderPath, self.id, 'BITalino', 'CSV',b1d )
        b1l_path = os.path.join(self.folderPath, self.id, 'BITalino', 'CSV',b1l )
        b2d_path = os.path.join(self.folderPath, self.id, 'BITalino', 'CSV',b2d )
        b2l_path = os.path.join(self.folderPath, self.id, 'BITalino', 'CSV',b2l )
        
        paths = {'rokoko1Path': r1d_path, 'rokoko2Path': r2d_path,
                 'bitalino1Path': b1d_path, 'bitalino2Path': b2d_path,
                 'labels1Path': r1l_path, 'labels2Path': r2l_path}
                 
        
        return paths
    
    def __str__(self):

        print("Condition: %s" %(self.condition))
        for i,w in enumerate(self.walks):
            print("\tWalk %d" %(i))
            print(w)

        return ""
    
class Subject():

    def __init__(self, id, folderPath, featuresIdx, winLength):

        self.id = id
        self.folderPath = folderPath
        self.featuresIdx = featuresIdx

        self.controlCondition = Condition(id, folderPath, 'Control', featuresIdx, winLength)
        self.highCondition = Condition(id, folderPath, 'High', featuresIdx, winLength)
        self.lowCondition = Condition(id, folderPath, 'Low', featuresIdx, winLength)
        self.conditions = [self.controlCondition, self.highCondition, self.lowCondition]

        self._relabelConditions()

        return

    def _relabelConditions(self):

        # the original labels come in a likert scale from 1 to 7 (light to heavy)
        # for the classification competition we are interested in whether the user
        # feels lighter or heavier than with the control. Thus: we are going to 
        # re-lable those 1-7 likert scale indexes, to -1 (the user feels lighter
        # than with the control audio condition) 0 (is a control condition) and (1)
        # the subject claims to feel heavier than with the control condition).
        #
        # Since for the control walks, the subject can give different 1-7 labels, 
        # we get the median of the control labels to use it as a reference. Then,
        # we label each walk as -1 if the user claims a 1-7 index smaller than the 
        # control median, and +1 if the user claims a 1-7 index larger than the control 
        # median. 
        # 
        # Our models need to learn to predict given a sample of a condition (LF, HF or control)
        # and a sample of the control condition (X, Xc), if the user feels lighter, heavier or the same
        # as with the control condition. 


        # find the median of the controls
        controlLabels = [walk.label for walk in self.controlCondition.walks]
        median = statistics.median(controlLabels)

        # all control windows are given a label of 0
        for walk in self.controlCondition.walks: 
            walk.label = 0

        # relabel high freq 
        for walk in self.highCondition.walks: 
            
            if walk.label >  median: 
                walk.label = 1
            elif walk.label == median: 
                walk.label = 0
            elif walk.label <  median: 
                walk.label = -1
        
        # and low freq conditions
        for walk in self.lowCondition.walks: 
            
            if walk.label >  median: 
                walk.label = 1
            elif walk.label == median: 
                walk.label = 0
            elif walk.label <  median: 
                walk.label = -1
        return
    
    def __getitem__(self, index):
        return self.conditions[index]
    
    def __str__(self):

        print("SubjectId: %s" %(self.id))
        print("Folder: %s" %(self.folderPath))

        print(self[0])
        print(self[1])
        print(self[2])
        return ""

class SubjectDataset(Dataset):

    def __init__(self, folderPath, excludeIds, mode='train', 
                 winLength = 300, featuresIdx = list(range(0,312)),
                 splitPcts= {'train': 80, 'val': 20, 'test': 0}, seed=42):


        self.folderPath = folderPath
        self.excludeIds = excludeIds
        self.mode = mode
        self.winLength = winLength
        self.featuresIdx = featuresIdx
        self.splitPcts = splitPcts
        self.seed = seed
        self.subjectIds = []
        self.subjectObj = []
        self.numSamples = -1

        # 1 - Get the Subject ids
        self.subjectIds = self._getSubjectIds()

        # 2 - Load the subjects
        self.loadSubjects()

        # 3 - Compute the total samples avaliable (estimation)
        self.setNumSamples()

        return
    
    
    def __getitem__(self, idx):

        # pick a subject, condition (Low freq, high freq or control) and walk randomly
        p = random.randint(0,len(self.subjectObj)-1)
        c = random.randint(0,2)
        w = random.randint(0, len(self.subjectObj[p].conditions[c].walks)-1)
        walk = self.subjectObj[p].conditions[c].walks[w]

        # sample winLength samples randomly from the walk
        X = walk.sample(self.winLength)
        y = walk.label

        # let's do the same with the control condition now
        w = random.randint(0, len(self.subjectObj[p].conditions[0].walks)-1)
        walk = self.subjectObj[p].conditions[0].walks[w]
        Xc = walk.sample(self.winLength)

        #prepare data dims and type
        X = X.unsqueeze(0)
        Xc = Xc.unsqueeze(0)
        y = torch.tensor(y, dtype=torch.long)

        return X,Xc,y
    
    def __len__(self):
        return self.numSamples

    def setNumSamples(self):

        # we are going to sample randomly from the signals. 
        # There are len(self.subjectObj) subjects in this dataset, 
        # each subject with 3 conditions (control, high, low), 
        # each condition has 12 walks
        # we are going to subsample chunks of data from each walk
        # each chunk is of len self.winSpecs[0]. There are an average of
        # 5 independent windows per walk when using 300 samples per window
        numSamples = len(self.subjectIds)*3*12*5

        if self.mode == 'train':
            self.numSamples = numSamples
        else:
            self.numSamples = int(numSamples/10)
        return
    
    def loadSubjects(self):

        for i,id_ in enumerate(self.subjectIds):
            print("\rLoading subject (%d/%d)" %(i+1, len(self.subjectIds)), end="")
            self.subjectObj.append(
                Subject(id_, self.folderPath, self.featuresIdx, self.winLength)
            )
        print("")
        return
    
    def _getSubjectIds(self):

        # read all folder names from self.folderPath and sort them naturally
        folders = [f for f in os.listdir(self.folderPath) if os.path.isdir(os.path.join(self.folderPath, f))]
        subjectIds = natsorted([id_ for id_ in folders if id_ not in self.excludeIds])

        random.seed(self.seed)
        random.shuffle(subjectIds)

        trainSize = int( len(subjectIds)*self.splitPcts['train']/100 )

        if self.mode == 'train':
            ids = subjectIds[:trainSize]
        elif self.mode == 'val':
            ids = subjectIds[trainSize:]
        else:
            print("Mode should be train or val")

        return ids
    

    def _copyLabelFiles(self, source, destination):

        for root, dirs, files in os.walk(source):

            # Compute the destination path
            dest_path = os.path.join(destination, os.path.relpath(root, source))
            
            # Create directories in destination
            os.makedirs(dest_path, exist_ok=True)
            
            # Copy files except those containing "data"
            for file in files:
                if "data" not in file:
                    shutil.copy2(os.path.join(root, file), os.path.join(dest_path, file))
        return
    
    def generateTestSubmission(self, path='testSubmission'):

        # 1 - Copy all labels.csv files to path destination
        self._copyLabelFiles(self.folderPath, path)

        # 2 - Iterate over each subject, condition and walk and set the labels in the .csv files
        for subjectObj in self.subjectObj:  
            for condition in subjectObj.conditions:
                
                # there are two csv files we have to edit for each condition
                csv1Path = condition.paths['labels1Path']
                csv2Path = condition.paths['labels2Path']

                csv1Path = csv1Path[csv1Path.find("test"):].replace("test", path, 1)
                csv2Path = csv2Path[csv2Path.find("test"):].replace("test", path, 1)

                df1 = pd.read_csv(csv1Path)
                df2 = pd.read_csv(csv2Path)

                for walk in condition.walks:

                    df = df1 if walk.filePath == condition.paths['labels1Path'] else df2

                    # write the inference label for this walk in the pandas dataframe
                    df.iloc[walk.endRow, 1] = walk.inferenceLabel

                # save the modified CSV's
                df1.to_csv(csv1Path, index=False)
                df2.to_csv(csv2Path,  index=False)
        return
  
if __name__ == '__main__':

    # # "Dataset "
    # folderPath = '../dataset/MLSP_bit_dataset/train' 
    # excludeIds = ['p021', 'p028', 'p051', 'p288']

    # winLength = 300
    # featuresIdx = list(range(0,306))
    # splitPcts= {'train': 60, 'val': 40}
    # seed=42

    # datasetTr = SubjectDataset(folderPath, excludeIds, mode='train', 
    #              winLength = 300, featuresIdx = list(range(0,306)),
    #              splitPcts= splitPcts, seed=seed)
    
    # datasetVal = SubjectDataset(folderPath, excludeIds, mode='val', 
    #              winLength = 300, featuresIdx = list(range(0,306)),
    #              splitPcts= splitPcts, seed=seed)

    seed=42
    winLength = 300
    featidx = list(range(0,306))
    splitPcts= {'train': 5, 'val': 0}
    testDB = '../dataset/MLSP_bit_dataset/test'
    test_dataset = SubjectDataset(testDB, [], mode='train', 
                 winLength = 300, featuresIdx = featidx,
                 splitPcts= splitPcts, seed=seed)
    
    test_dataset.generateTestSubmission('testSubmission')

    
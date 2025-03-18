import torch
import random
import argparse
from models import Simple1DCNN
from dataset import SubjectDataset

def run(modelPath, testDB):

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # features
    featidx = list(range(0,306)) # use all features  

    # load the model
    model = Simple1DCNN(num_sensors=len(featidx), num_classes=3).to(device)
    checkpoint = torch.load(modelPath)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # load the database
    splits = {'train': 100, 'val': 0}
    seed = 42
    winLength = 125
    test_dataset = SubjectDataset(testDB, [], mode='train', 
                 winLength = winLength, featuresIdx = featidx,
                 splitPcts= splits, seed=seed)
    

    # go over each test walk and set the walk.inferenceLabel attribute
    # Note that we are only using winLength samples from the full walk to 
    # create the labels for the entire walk. More intelligent approaches
    # are recommended...

    for subjectObj in test_dataset.subjectObj:
        for condition in subjectObj.conditions:
            for walk in condition.walks:

                # optional: get a sample from a control walk if you model needs both X, and Xc
                w = random.randint(0, len(subjectObj.conditions[0].walks)-1)
                walkControl = subjectObj.conditions[0].walks[w]
                Xc = walkControl.sample(winLength)

                # get a sample for the current walk
                X = walk.sample(winLength)
                
                with torch.no_grad():
                    X = X.unsqueeze(0).unsqueeze(0)
                    Xc = Xc.unsqueeze(0).unsqueeze(0)
                    X, Xc = X.to(device), Xc.to(device)
                    output = model(X,Xc)
                    _,label = output.max(1)
                    label = label - 1 # Shift labels by -1, the model outputs 0,1,2 but labels are expected -1,0,1
                    walk.inferenceLabel = label.item()

    # Once we have the labels for each walk, we can create the testSubmission folder with:
    test_dataset.generateTestSubmission(path = 'baselineSubmission')

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train the baseline model")

    parser.add_argument('--test_path', type=str, default='../dataset/MLSP_bit_dataset/test', help='the path to the test dataset')
    parser.add_argument('--model', type=str, default='checkpoint_epoch_10.pth', help='the path to the model for inference')

    args = parser.parse_args()

    model = args.model
    testDB = args.test_path
    run(model, testDB)
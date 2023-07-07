 #!/usr/bin/env python
import argparse
import os
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import uproot
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

def getRocRate(rootFile):
   partialEvents=uproot.open(rootFile+":demo/partialDisappear/eventCount").to_numpy()
   hcalE=uproot.open(rootFile+":demo/partialDisappear_largeCSCDefl/ConeEnergy")
   passNum=partialEvents[0]
   binnum = 0
   totalNum=0
   for value in hcalE.values():
      if(hcalE.axis().edges()[binnum]<1):
         totalNum=totalNum+value
      binnum=binnum+1
   return totalNum/passNum
 
def testOffPeak(bdt,features):
   sameSignFile=uproot.open("/local/cms/user/revering/dphoton/slc7/CMSSW_10_6_17_patch1/src/DarkPhoton/MuAnalyzer/histograms/CRReReco_SameSign.root")
   sameSignEvents=sameSignFile["demo"]["partialDisappear"]["sigVariables"]
   sameSignInputs=sameSignEvents.arrays(features, library="pd")
   label=[]
   for entry in range(sameSignInputs.shape[0]):
      label.append(0)
   sameSignInputs['label']=label
   staEfilter = (sameSignInputs['standaloneDEoverE']<-0.6)&(sameSignInputs['cellEdgeDeta']>0.004)&(sameSignInputs['cellEdgeDphi']>0.016)
   filteredSameSign = sameSignInputs[staEfilter] 
   matrixSameSign = xgb.DMatrix(filteredSameSign.drop(['EventWeight','label','cellEdgeDeta','cellEdgeDphi'],axis=1))
   pred = bdt.predict(matrixSameSign)
   return pred

def binLogNet(x,y):
   train_input, test_input, train_lbl, test_lbl = train_test_split(x, y, test_size=1/7., random_state=0)
   dtrain = xgb.DMatrix(train_input,label=train_lbl)
   dtest = xgb.DMatrix(test_input,label=test_lbl)
   param = {'objective':'binary:logistic','max_depth':3,'eta':1}
   param['nthread'] = 8
   param['eval_metric']='auc'
   num_round = 10
   evallist = [(dtest,'eval'),(dtrain,'train')]
   bst = xgb.train(param, dtrain,num_round, evallist)
   return bst

def main():
   
   parser = argparse.ArgumentParser(description="Fit the background fraction as a function of isolation")
   parser.add_argument("--i", dest="inDir", help="Directory containing analyzer results", default=os.getcwd()+"/histograms/")
   arg = parser.parse_args()
   inDir = arg.inDir
   #Check for trailing slash on input dir and delete
   if arg.inDir.split("/")[-1] == "": inDir = arg.inDir[:-1]
   if not os.path.isdir(inDir):
        print("Input directory " + inDir + " does not exist!")
        quit()
  
#   events = uproot.open(str(inDir)+"/ZmmSim.root:demo/partialDisappear/sigVariables")
   features = ["pt","eta","phi","staDR","staPhi","staE","staChi","cscDR","HEDepth_0","HEDepth_1","HEDepth_2","HEDepth_3","HEDepth_4","HEDepth_5"]
   featuresNoHE = ["pt","eta","phi","staDR","staPhi","staE","staChi","cscDR"]

   mcFile = uproot.open(str(inDir)+"/ZmmSim.root")
   falsePos = getRocRate(str(inDir)+"/ZmmSim.root")
   events=mcFile["demo"]["partialDisappear"]["sigVariables"]
   mcInputs = events.arrays(features, library="pd")
   mcLabel = []
   for entry in range(mcInputs.shape[0]):
     #mcLabel.append('DY')
     mcLabel.append(0)

   mcInputs['label'] = mcLabel 

   sigFiles = []
   for filename in os.listdir(inDir):
      f = os.path.join(inDir, filename)
      # checking if it is a file
      if os.path.isfile(f):
          if "DBrem" in filename:
             apmass = filename.split("_")[-1].split(".")[0]
             if "0p2" in apmass:
               sigEvents=uproot.open(str(inDir)+"/"+filename+":demo/partialDisappear/sigVariables")
               sigFiles.append(sigEvents.arrays(features, library="pd"))
               truePos = getRocRate(str(inDir)+"/"+filename)
               massLabel=[]
               for entry in range(sigFiles[-1].shape[0]):
                #massLabel.append(apmass)
                  massLabel.append(1)
               sigFiles[-1]['label']=massLabel

   for sigInput in sigFiles:
     mcInputs = pd.concat([mcInputs,sigInput])

   print(truePos,falsePos)
   x = mcInputs.loc[:, features].values
   y = mcInputs.loc[:, ['label']].values

   xNoHE = mcInputs.loc[:, featuresNoHE].values

   bst = binLogFit(x,y)

   dtrainNoHE = xgb.DMatrix(train_noHE,label=train_noHElbl)
   dtestNoHE = xgb.DMatrix(test_noHE,label=test_noHElbl)
   bstNoHE = xgb.train(param, dtrainNoHE,num_round, evallistNoHE)

   y_preds = bst.predict(xgb.DMatrix(x,label=y))

   return
   #Remove the scale prior to PCA decomposition
   x = StandardScaler().fit_transform(x)
   #Do the PCA decomp
   #Do a fraction instead of n_components to request a precision (i.e. 0.95 for 95% variance)
   pca = PCA(0.99)
   principalComponents = pca.fit_transform(x)
   print(pca.explained_variance_ratio_)

if __name__ == "__main__":
   main() 
   

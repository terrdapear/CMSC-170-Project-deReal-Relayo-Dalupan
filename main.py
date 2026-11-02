from scripts.preprocess import load_and_bow_everyth,load_and_bow_list,prepare_data,make_class_bags
from scripts.count import total_words
from scripts.NB import NB_for_2
from scripts.KNN import KNN_for_2_colors
from scripts.perceptron import Perceptron
from scripts.metrics import evaluate
from time import perf_counter
import os

#.-._.-._.- Parameters .-._.-._.-#

#share of episodes and acts used for training
TRAIN_RATIO = 0.8

#same seed, same split and same shuffles
SEED = 170

#NB smoothing
LAPLACE = 1

#k in KNN
K = 7

#distance exponent in KNN
P = 2

#Perceptron learning rate
R = 1

#Perceptron starting bias
BIAS = 0

#Perceptron stops here if it still makes mistakes
MAX_EPOCHS = 20

#.-._.-._.- Parameters .-._.-._.-#


ALL_BCS = "Better_Call_Saul_Transcripts/All_Seasons"
ALL_RA = "RA/RA_split"

BCS_DIR = "Data/Training/BCS"
RA_DIR = "Data/Training/RA"

BCS_TEST = "Data/Testing/BCS"
RA_TEST = "Data/Testing/RA"

def test_NB(NB):
    print("\nTesting NB...\n")
    correct = 0
    total = 0
    
    #expecting 0s
    for root,dirs,files in os.walk(BCS_TEST):
        for file in files:
            file_path = os.path.join(root,file)
            with open(file_path,'r',encoding="utf-8") as f:
                result = NB.classify_paragraph(f.read())
                if result==0:
                    correct+=1
                print("%40s" % (f"{file} is __{result}__"))
                total+=1

    #expecting 1s
    for root,dirs,files in os.walk(RA_TEST):
        for file in files:
            file_path = os.path.join(root,file)
            with open(file_path,'r',encoding="utf-8") as f:
                result = NB.classify_paragraph(f.read())
                if result==1:
                    correct+=1
                print("%40s" % (f"{file} is __{result}__"))
                total+=1

    print(f"\nTotal tests: {total}\nCorrectly classified: {correct}")


def test_KNN(KNN):
    BCS = load_and_bow_list(BCS_TEST)
    RA = load_and_bow_list(RA_TEST)

    print("\nTesting KNN...\n")

    correct = 0
    total = 0

    # 00000
    for bag,name in BCS:
        result = KNN.classify(bag)
        print("%40s" % (f"{name} is __{result}__"))
        if result==0:
            correct+=1
        total+=1

    # 1111111
    for bag,name in RA:
        result = KNN.classify(bag)
        print("%40s" % (f"{name} is __{result}__"))
        if result==1:
            correct+=1
        total+=1

    print(f"\nTotal tests: {total}\nCorrectly classified: {correct}")


#out of curiousity
def test_all_NB(NB):
    correct = 0
    total = 0
    for root,dirs,files in os.walk(ALL_RA):
        for file in files:
            file_path = os.path.join(root,file)
            with open(file_path,'r',encoding="utf-8") as f:
                result = NB.classify_paragraph(f.read())
                if result==1:
                    correct+=1
                if result==0:
                    print("%40s" % (f"{file} is __{result}__"))
                total+=1

    print(f"Total tests: {total}\nCorrectly classified: {correct}")


#the classes are balanced, so a model has to beat this to be learning anything
class AlwaysRA:
    def classify_paragraph(s,p):
        return 1


def print_results(rows):
    print("\n---------- RESULTS ----------")
    print(f"{'':<12}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}{'Train (s)':>11}{'Classify (s)':>14}")
    for name,train_time,r in rows:
        print(f"{name:<12}{r['accuracy']:>10.4f}{r['precision']:>11.4f}{r['recall']:>9.4f}{r['f1']:>9.4f}{train_time:>11.3f}{r['runtime']:>14.3f}")

    print(f"\n{'':<12}{'TP':>7}{'FN':>7}{'FP':>7}{'TN':>7}")
    for name,train_time,r in rows:
        print(f"{name:<12}{r['tp']:>7}{r['fn']:>7}{r['fp']:>7}{r['tn']:>7}")


def main():

    print(f"---------- LOAD and PREPROCESS ----------")
    train,train_labels,test,test_labels = prepare_data(TRAIN_RATIO,SEED)
    print(f"---------- LOAD and PREPROCESS ----------")

    rows = [("Always RA",0,evaluate(AlwaysRA(),test,test_labels))]

    #bags are made here since they are part of training NB
    start = perf_counter()
    BCS,RA = make_class_bags(train,train_labels)
    NB = NB_for_2(BCS,RA,LAPLACE)
    rows.append(("Naive Bayes",perf_counter()-start,evaluate(NB,test,test_labels)))

    PT = Perceptron(R,BIAS,MAX_EPOCHS,SEED)
    start = perf_counter()
    PT.train(train,train_labels)
    rows.append(("Perceptron",perf_counter()-start,evaluate(PT,test,test_labels)))

    #KNN still runs on the old Data folders until it can handle the real split
    print_results(rows)



if __name__ == "__main__":
    main()
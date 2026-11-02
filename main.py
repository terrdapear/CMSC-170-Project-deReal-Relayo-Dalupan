from scripts.preprocess import load_and_bow_everyth,load_and_bow_list
from scripts.count import total_words
from scripts.NB import NB_for_2
from scripts.KNN import KNN_for_2_colors
import os

#.-._.-._.- Parameters .-._.-._.-#

#NB smoothing
LAPLACE = 1

#k in KNN
K = 7

#distance exponent in KNN
P = 2

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


def main():
    
    #Better call saul bag
    print(f"---------- LOAD and BOW ----------")
    BCS = load_and_bow_everyth(BCS_DIR)
    RA = load_and_bow_everyth(RA_DIR)

    BCS_list = load_and_bow_list(BCS_DIR)
    RA_list = load_and_bow_list(RA_DIR)
    print(f"---------- LOAD and BOW ----------")
    # print(total_words(BCS),total_words(RA))

    NB = NB_for_2(BCS,RA,1)
    KNN = KNN_for_2_colors([x[0] for x in BCS_list],[x[0] for x in RA_list],K,P)
    
    test_NB(NB)
    test_KNN(KNN)
    # test_all_NB(NB)
    



if __name__ == "__main__":
    main()
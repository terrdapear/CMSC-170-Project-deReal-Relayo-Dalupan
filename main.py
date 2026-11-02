from scripts.load import load_and_bow_everyth
from scripts.count import total_words
from scripts.NB import NB_for_2
import os

ALL_BCS = "Better_Call_Saul_Transcripts/All_Seasons"
ALL_RA = "RA/RA_split"

BCS_DIR = "Data/Training/BCS"
RA_DIR = "Data/Training/RA"

BCS_TEST = "Data/Testing/BCS"
RA_TEST = "Data/Testing/RA"

MAGIC_NUM = 3847
#average number of words in a Better Call Saul episode

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
    print(f"---------- LOAD and BOW ----------")
    # print(total_words(BCS),total_words(RA))

    NB = NB_for_2(BCS,RA,1)
    
    test_NB(NB)
    # test_all_NB(NB)
    



if __name__ == "__main__":
    main()
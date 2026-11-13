from scripts.preprocess import prepare_data_folders,make_class_bags,make_bag_lists
from scripts.NB import NB_for_2
from scripts.KNN import KNN_for_2_colors
from scripts.perceptron import Perceptron
from scripts.metrics import evaluate
from time import perf_counter

#.-._.-._.- Parameters .-._.-._.-#

#same seed, same shuffles in the Perceptron
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


BCS_DIR = "Better_Call_Saul_Transcripts/per10"
RA_DIR = "RA/per10"
test_ratio = 80
#int from 0 to 100

#the classes are balanced, so a model has to beat this to be learning anything
class AlwaysRA:
    def classify_paragraph(s,p):
        return 1


def print_results(rows):
    print("\n---------- RESULTS ----------")
    print(f"{'':<28}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}{'Train (s)':>11}{'Classify (s)':>14}")
    for name,train_time,r in rows:
        print(f"{name:<28}{r['accuracy']:>10.4f}{r['precision']:>11.4f}{r['recall']:>9.4f}{r['f1']:>9.4f}{train_time:>11.3f}{r['runtime']:>14.3f}")

    print(f"\n{'':<28}{'TP':>7}{'FN':>7}{'FP':>7}{'TN':>7}")
    for name,train_time,r in rows:
        print(f"{name:<28}{r['tp']:>7}{r['fn']:>7}{r['fp']:>7}{r['tn']:>7}")


def main():

    print(f"---------- LOAD and PREPROCESS ----------")
    train,train_labels,test,test_labels = prepare_data_folders(BCS_DIR,RA_DIR,test_ratio)
    print(f"---------- LOAD and PREPROCESS ----------")

    rows = [("Always RA",0,evaluate(AlwaysRA(),test,test_labels))]

    #bags are made here since they are part of training NB
    start = perf_counter()
    BCS,RA = make_class_bags(train,train_labels)
    NB = NB_for_2(BCS,RA,LAPLACE)
    print("Testing Naive Bayes...")
    rows.append(("Naive Bayes",perf_counter()-start,evaluate(NB,test,test_labels)))

    for K in [3,5,11]:
        for P in [1,2]:
            start = perf_counter()
            BCS_list,RA_list = make_bag_lists(train,train_labels)
            KNN = KNN_for_2_colors(BCS_list,RA_list,K,P)
            print(f"Testing KNN [K={K} P={P}]...")
            rows.append((f"KNN [K={K} P={P}]",perf_counter()-start,evaluate(KNN,test,test_labels)))

    PT = Perceptron(R,BIAS,MAX_EPOCHS,SEED)
    start = perf_counter()
    PT.train(train,train_labels)
    print("Testing Perceptron...")
    rows.append(("Perceptron",perf_counter()-start,evaluate(PT,test,test_labels)))

    print_results(rows)



if __name__ == "__main__":
    main()
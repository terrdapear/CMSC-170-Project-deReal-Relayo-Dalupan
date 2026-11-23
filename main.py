import os
import sys
from time import perf_counter
from scripts.preprocess import prepare_data_folders, make_class_bags, make_bag_lists
from scripts.NB import NB_for_2
from scripts.KNN import KNN_for_2_colors
from scripts.perceptron import Perceptron
from scripts.metrics import evaluate
from scripts.persistence import save_model, load_model

#.-._.-._.- Parameters .-._.-._.-#

# same seed, same shuffles in the Perceptron
SEED = 170

# NB smoothing, one run for each
LAPLACES = [1, 0.5, 0.1]

# k in KNN
KS = [3, 5,7,9,11,13,15]

# distance exponent in KNN
PS = [1, 2]

# Perceptron learning rate
RS = [0.1,0.5,0.01]

# Perceptron starting bias
BIASES = [0, 1, -1]

# Perceptron stops here if it still makes mistakes
MAX_EPOCHS = 100

#.-._.-._.- Parameters .-._.-._.-#

BCS_DIR = "Better_Call_Saul_Transcripts/per10"
RA_DIR = "RA/RA_single/per10"
MODELS_DIR = "models"
train_percent = 80  # int from 0 to 100

out = []

# the classes are balanced, so a model has to beat this to be learning anything
class AlwaysRA:
    def classify_paragraph(s, p):
        return 1

def print(*args):
    p = " ".join(map(str,args))
    sys.stdout.write(f"{p}\n")
    out.append(p)

    

def print_results(rows):
    print("\n---------- RESULTS ----------")
    print(f"{'':<28}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}{'Train (s)':>11}{'Classify (s)':>14}")
    for name, train_time, r in rows:
        print(f"{name:<28}{r['accuracy']:>10.4f}{r['precision']:>11.4f}{r['recall']:>9.4f}{r['f1']:>9.4f}{train_time:>11.3f}{r['runtime']:>14.3f}")

    print(f"\n{'':<28}{'TP':>7}{'FN':>7}{'FP':>7}{'TN':>7}")
    for name, train_time, r in rows:
        print(f"{name:<28}{r['tp']:>7}{r['fn']:>7}{r['fp']:>7}{r['tn']:>7}")


def train_models():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\n---------- LOAD and PREPROCESS ----------")
    train, train_labels, test, test_labels = prepare_data_folders(BCS_DIR, RA_DIR, train_percent)
    save_model((test, test_labels), "test_data", MODELS_DIR)
    print("---------- LOAD and PREPROCESS ----------")

    print("\n---------- TRAINING & SAVING MODELS ----------")

    for LAPLACE in LAPLACES:
        start = perf_counter()
        BCS, RA = make_class_bags(train, train_labels)
        NB = NB_for_2(BCS, RA, LAPLACE)
        elapsed = perf_counter() - start

        name = f"nb_k{LAPLACE}"
        ok = save_model(NB, name, MODELS_DIR)
        print(f"[{'OK' if ok else 'FAIL'}] {name}.pkl ({elapsed:.3f}s)")

    for K in KS:
        for P in PS:
            start = perf_counter()
            BCS_list, RA_list = make_bag_lists(train, train_labels)
            KNN = KNN_for_2_colors(BCS_list, RA_list, K, P)
            elapsed = perf_counter() - start

            name = f"knn_k{K}_p{P}"
            ok = save_model(KNN, name, MODELS_DIR)
            print(f"[{'OK' if ok else 'FAIL'}] {name}.pkl ({elapsed:.3f}s)")

    for R in RS:
        for BIAS in BIASES:
            PT = Perceptron(R, BIAS, MAX_EPOCHS, SEED)
            start = perf_counter()
            PT.train(train, train_labels)
            elapsed = perf_counter() - start

            name = f"pt_r{R}_b{BIAS}"
            ok = save_model(PT, name, MODELS_DIR)
            print(f"[{'OK' if ok else 'FAIL'}] {name}.pkl ({elapsed:.3f}s)")

    return test, test_labels


def run_models(test=None, test_labels=None):
    if test is None or test_labels is None:
        saved_test = load_model(f"{MODELS_DIR}/test_data.pkl")
        if saved_test is None:
            print("No saved test data found - run [1] train models first.")
            return
        test, test_labels = saved_test

    print("\n---------- RUNNING SAVED MODELS ----------")
    rows = [("Always RA", 0, evaluate(AlwaysRA(), test, test_labels))]

    for LAPLACE in LAPLACES:
        path = f"{MODELS_DIR}/nb_k{LAPLACE}.pkl"
        NB = load_model(path)
        if NB is None:
            print(f"Missing {path} - skipping (train models first).")
            continue
        print(f"Testing Naive Bayes [k={LAPLACE}]...")
        rows.append((f"Naive Bayes [k={LAPLACE}]", 0, evaluate(NB, test, test_labels)))

    for K in KS:
        for P in PS:
            path = f"{MODELS_DIR}/knn_k{K}_p{P}.pkl"
            KNN = load_model(path)
            if KNN is None:
                print(f"Missing {path} - skipping (train models first).")
                continue
            print(f"Testing KNN [K={K} P={P}]...")
            rows.append((f"KNN [K={K} P={P}]", 0, evaluate(KNN, test, test_labels)))

    for R in RS:
        for BIAS in BIASES:
            path = f"{MODELS_DIR}/pt_r{R}_b{BIAS}.pkl"
            PT = load_model(path)
            if PT is None:
                print(f"Missing {path} - skipping (train models first).")
                continue
            print(f"Testing Perceptron [r={R} b={BIAS}]...")
            rows.append((f"Perceptron [r={R} b={BIAS}]", 0, evaluate(PT, test, test_labels)))

    print_results(rows)


def delete_models():
    print("\n---------- DELETING SAVED MODELS ----------")
    if not os.path.exists(MODELS_DIR):
        print("No models directory found.")
        return

    removed = 0
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith(".pkl"):
            os.remove(os.path.join(MODELS_DIR, filename))
            print(f"Deleted {filename}")
            removed += 1

    if removed == 0:
        print("No saved models to delete.")
    else:
        print(f"Removed {removed} file(s).")


def main():
    test, test_labels = None, None

    while True:
        print("\n[1] train models")
        print("[2] run models")
        print("[3] delete all models")
        print("[4] exit")
        choice = input("> ").strip()

        if choice == "1":
            test, test_labels = train_models()
        elif choice == "2":
            run_models(test, test_labels)
        elif choice == "3":
            delete_models()
            test, test_labels = None, None
        elif choice == "4":
            break
        else:
            print("Invalid option, enter 1, 2, 3, or 4.")
    
    with open("out.txt","w") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
import os
from collections import Counter
from scripts.bow import clean_update


def list_folder(path):
    # sorted so every run sees the same order
    return sorted(os.listdir(path))


def load_folder(path, files):
    texts = []
    for file in files:
        with open(os.path.join(path, file), 'r', encoding="utf-8") as f:
            texts.append(f.read())
    return texts


def prepare_data_folders(bcs,ra,test_ratio): # test_ratio out of 100
    '''
    returns
    train paragraphs, train labels, test paragraphs, test labels

    Each file in the folders is one paragraph. Both folders are cut to the
    same number of files, and the first test_ratio% of each goes to training.

    labels
    0 : Better Call Saul
    1 : Republic Act
    '''
    bcs_files = list_folder(bcs)
    ra_files = list_folder(ra)

    m,n = len(bcs_files),len(ra_files)

    #only the first men files of each are used, so only those are read
    men = min(m,n)
    bcs = load_folder(bcs,bcs_files[:men])
    ra = load_folder(ra,ra_files[:men])

    bcs_train = bcs[:men*test_ratio//100]
    ra_train = ra[:men*test_ratio//100]
    train_paragraphs = bcs_train + ra_train
    train_labels = [0] * len(bcs_train) + [1] * len(ra_train)

    bcs_test = bcs[men*test_ratio//100:]
    ra_test = ra[men*test_ratio//100:]
    test_paragraphs = bcs_test + ra_test
    test_labels = [0] * len(bcs_test) + [1] * len(ra_test)

    print(f"Train: {len(train_paragraphs)} files, Test: {len(test_paragraphs)} files")
    return train_paragraphs, train_labels, test_paragraphs, test_labels


def make_bag_lists(paragraphs, labels):
    '''
    returns
    list of bags of Better Call Saul, list of bags of Republic Acts
    '''
    bcs = []
    ra = []
    for i in range(len(paragraphs)):
        bag = Counter()
        clean_update(bag, paragraphs[i])
        if labels[i] == 0:
            bcs.append(bag)
        else:
            ra.append(bag)
    return bcs, ra


def make_class_bags(paragraphs, labels):
    '''
    returns
    bag of words of Better Call Saul, bag of words of Republic Acts
    '''
    bcs = Counter()
    ra = Counter()
    for i in range(len(paragraphs)):
        if labels[i] == 0:
            clean_update(bcs, paragraphs[i])
        else:
            clean_update(ra, paragraphs[i])
    return bcs, ra

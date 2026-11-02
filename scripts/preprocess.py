import csv
import os
from collections import Counter
from random import Random
from scripts.bow import clean, clean_update
from scripts.count import total_words

BCS_DIR = "Better_Call_Saul_Transcripts/All_Seasons"
RA_DIR = "RA/RA_split"

BCS_HEADER_END = "=" * 40
BCS_FOOTER = ("Previous Episode", "Next Episode", "Have any questions?")
BCS_CREDIT = "Synced and corrected by"

WINDOW_SIZE = 10
MAX_WINDOWS_PER_ACT = 50

TRAIN_RATIO = 0.8
SEED = 170


def strip_bcs_extras(text):
    lines = text.split("\n")

    # title, source, then a line of =
    if BCS_HEADER_END in lines[:5]:
        lines = lines[lines.index(BCS_HEADER_END) + 1:]

    # links to other episodes and the contact line
    while lines and (lines[-1].strip() == "" or lines[-1].startswith(BCS_FOOTER)):
        lines.pop()

    kept = []
    for line in lines:
        if not line.startswith(BCS_CREDIT):
            kept.append(line)
    return "\n".join(kept)


def load_bcs(path=BCS_DIR):
    '''
    returns
    list of (episode name, episode text)
    '''
    episodes = []

    # sorted so every run sees the same order
    for file in sorted(os.listdir(path)):
        file_path = os.path.join(path, file)
        with open(file_path, 'r', encoding="utf-8") as f:
            episodes.append((file, strip_bcs_extras(f.read())))

    print(f"{path} loaded: {len(episodes)} episodes")
    return episodes


def read_lines(path):
    # the RA_split files are one table cut into pieces, read them back as one
    for file in sorted(os.listdir(path)):
        file_path = os.path.join(path, file)
        with open(file_path, 'r', encoding="utf-8", newline='') as f:
            for line in f:
                yield line


def load_ra(path=RA_DIR):
    '''
    returns
    list of (act name, act text)

    Only the content column is kept. The other columns (id, path, tags, ...)
    are only in the RA data, so a model could use them to cheat.
    '''
    # some acts are longer than the default limit
    csv.field_size_limit(2**31 - 1)

    acts = []
    for row in csv.DictReader(read_lines(path), delimiter='\t'):
        acts.append((row["basename"], row["content"]))

    print(f"{path} loaded: {len(acts)} acts")
    return acts


def remove_duplicates(documents):
    # the same text in training and testing would not be a fair test
    seen = set()
    kept = []
    for name, text in documents:
        if text not in seen:
            seen.add(text)
            kept.append((name, text))
    return kept


def tokenize(text):
    tokens = []
    for word in text.split():
        cleaned = clean(word)
        if cleaned != None:
            tokens.append(cleaned)
    return tokens


def make_windows(text, size=WINDOW_SIZE, max_windows=None):
    '''
    returns
    list of paragraphs, each with exactly size words

    Leftover words at the end are dropped so every paragraph is the same length.
    If there are more than max_windows, evenly spaced ones are kept so a very
    long document cannot take over its class.
    '''
    tokens = tokenize(text)

    windows = []
    for i in range(0, len(tokens) - size + 1, size):
        windows.append(" ".join(tokens[i:i + size]))

    if max_windows != None and len(windows) > max_windows:
        spaced = []
        for i in range(max_windows):
            spaced.append(windows[i * len(windows) // max_windows])
        return spaced

    return windows


def split_documents(documents, train_ratio, random):
    # split before making windows so a whole episode or act stays on one side
    shuffled = list(documents)
    random.shuffle(shuffled)
    cut = int(len(shuffled) * train_ratio)
    return shuffled[:cut], shuffled[cut:]


def documents_to_windows(documents, max_windows=None):
    windows = []
    for name, text in documents:
        for window in make_windows(text, max_windows=max_windows):
            windows.append(window)
    return windows


def remove_repeats(windows, seen):
    # standard wording repeats across acts, and seen stops it from being in both train and test
    kept = []
    for window in windows:
        if window not in seen:
            seen.add(window)
            kept.append(window)
    return kept


def balance(bcs, ra, random):
    # RA has far more windows, so always guessing RA would look accurate otherwise
    size = min(len(bcs), len(ra))
    return random.sample(bcs, size), random.sample(ra, size)


def add_labels(bcs, ra, random):
    labeled = []
    for window in bcs:
        labeled.append((window, 0))
    for window in ra:
        labeled.append((window, 1))
    random.shuffle(labeled)

    paragraphs = []
    labels = []
    for window, label in labeled:
        paragraphs.append(window)
        labels.append(label)
    return paragraphs, labels


def prepare_data(train_ratio=TRAIN_RATIO, seed=SEED):
    '''
    returns
    train paragraphs, train labels, test paragraphs, test labels

    labels
    0 : Better Call Saul
    1 : Republic Act
    '''
    random = Random(seed)

    train_bcs, test_bcs = split_documents(remove_duplicates(load_bcs()), train_ratio, random)
    train_ra, test_ra = split_documents(remove_duplicates(load_ra()), train_ratio, random)

    # train goes first so repeated windows are kept in train and removed from test
    seen = set()
    train_bcs = remove_repeats(documents_to_windows(train_bcs), seen)
    train_ra = remove_repeats(documents_to_windows(train_ra, MAX_WINDOWS_PER_ACT), seen)
    test_bcs = remove_repeats(documents_to_windows(test_bcs), seen)
    test_ra = remove_repeats(documents_to_windows(test_ra, MAX_WINDOWS_PER_ACT), seen)

    train_bcs, train_ra = balance(train_bcs, train_ra, random)
    test_bcs, test_ra = balance(test_bcs, test_ra, random)

    train_paragraphs, train_labels = add_labels(train_bcs, train_ra, random)
    test_paragraphs, test_labels = add_labels(test_bcs, test_ra, random)

    print(f"Train: {len(train_paragraphs)} paragraphs, Test: {len(test_paragraphs)} paragraphs")
    return train_paragraphs, train_labels, test_paragraphs, test_labels


def load_and_bow_everyth(path): #called from main
    bag = Counter()
    print(f"Loading files from {path}\n")

    for root,dirs,files in os.walk(path):
        print(f"Loading {len(files)} files...\n")
        for file in files:
            # print(f"loading {file}")
            file_path = os.path.join(root,file)

            with open(file_path,'r',encoding="utf-8") as f:
                clean_update(bag,f.read())
                f.close()
    print(f"{path} loaded\n")
    print(f"{path}'s bag of words contains {total_words(bag)} total words")

    return bag


def load_and_bow_list(path):

    bags = []

    print(f"Loading files from {path}\n")
    for root,dirs,files in os.walk(path):
        print(f"Loading {len(files)} files...\n")
        for file in files:
            # print(f"loading {file}")
            file_path = os.path.join(root,file)

            with open(file_path,'r',encoding="utf-8") as f:
                bags.append([Counter(),file])
                clean_update(bags[-1][0],f.read())
                f.close()

    print(f"{path} loaded\n")
    print(f"Created {len(bags)} bags, totalling {sum([total_words(bag[0]) for bag in bags])} words. ")
    return bags

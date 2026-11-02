import csv
import os
from scripts.bow import clean

BCS_DIR = "Better_Call_Saul_Transcripts/All_Seasons"
RA_DIR = "RA/RA_split"

BCS_HEADER_END = "=" * 40
BCS_FOOTER = ("Previous Episode", "Next Episode", "Have any questions?")
BCS_CREDIT = "Synced and corrected by"

WINDOW_SIZE = 10
MAX_WINDOWS_PER_ACT = 50


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

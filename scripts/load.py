import os
from scripts.bow import clean_update
from collections import Counter
from scripts.count import total_words

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




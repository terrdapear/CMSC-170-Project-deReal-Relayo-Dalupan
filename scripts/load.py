import os
from scripts.bow import clean_update
from collections import Counter

def load_and_bow_everyth(path): #called from main
    bag = Counter()

    for root,dirs,files in os.walk(path):
        for file in files:
            # print(f"loading {file}")
            file_path = os.path.join(root,file)

            with open(file_path,'r',encoding="utf-8") as f:
                clean_update(bag,f.read())
                f.close()
    return bag




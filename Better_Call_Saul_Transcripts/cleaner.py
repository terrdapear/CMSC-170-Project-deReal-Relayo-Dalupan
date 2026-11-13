import os

def cleaner():
    path = "All_Seasons"
    for file in sorted(os.listdir(path)):
        text = []
        path = "All_Seasons"
        with open(os.path.join(path, file), 'r', encoding="utf-8") as f:
            text = f.read().splitlines()[4:-3]
        path = "cleaned"
        with open(os.path.join(path, file), 'w', encoding="utf-8") as f:
            f.write("\n".join(text))
    
from collections import Counter

VALID_CHAR = set([x for x in 'abcdefghijklmnopqrstuvwxyz1234567890'])

def clean(s: str):
    s_new = "".join([c for c in s.lower() if c in VALID_CHAR])
    return s_new if s_new!='' else None

def clean_update(c, s: str):
    c.update([x for x in map(clean,s.split()) if x!=None])

def main():
    path = "cleaned"
    cap = 50
    path2=f"per{cap}"
    
    count = 0
    for file in sorted(os.listdir(path)):
        with open(os.path.join(path, file), 'r', encoding="utf-8") as f:
            text = f.read().split()

            temp = 0
            words = []

            for word in text:
                word = clean(word)
                if word==None:
                    continue
                temp+=1
                words.append(word)
                if temp==cap:
                    with open(os.path.join(path2,f"{count:05d}.txt"),'w', encoding="utf-8") as g:
                        g.write(" ".join(words))
                    words.clear()
                    temp = 0
                    count+=1
            with open(os.path.join(path2,f"{count:05d}.txt"),'w', encoding="utf-8") as g:
                g.write(" ".join(words))
            words.clear()
            temp = 0
            count+=1
                


    
    

main()


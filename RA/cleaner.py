
from collections import Counter

VALID_CHAR = set([x for x in 'abcdefghijklmnopqrstuvwxyz1234567890'])

def clean(s: str):
    s_new = "".join([c for c in s.lower() if c in VALID_CHAR])
    return s_new if s_new!='' else None

def clean_update(c, s: str):
    c.update([x for x in map(clean,s.split()) if x!=None])


def main():
    print("blep")
    cnt = 0
    cap = 10
    path = f"per{cap}"
    with open("repacts_with_summary.txt",'r') as f:
        words = []
        for line in f:
            for word in line.split():
                word = clean(word)
                if word==None:
                    continue
                words.append(word)
                if len(words)==cap:
                    with open(f"per{cap}/{cnt:08d}.txt",'w') as g:
                        g.write(" ".join(words))
                    words.clear()
                    cnt+=1
        with open(f"per{cap}/{cnt:08d}.txt",'w') as g:
            g.write(" ".join(words))

main()
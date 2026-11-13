
import os

def main():
    # sorted so every run sees the same order
    path = "RA_split"
    texts = []
    for file in sorted(os.listdir(path)):
        with open(os.path.join(path, file), 'r', encoding="utf-8") as f:
            texts.append(f.read())
    with open("RA_single/repacts_with_summary.txt",'w') as f:
        f.write("\n".join(texts))
main()
import os

def main():
    path = "All_Seasons"
    for file in sorted(os.listdir(path)):
        text = []
        path = "All_Seasons"
        with open(os.path.join(path, file), 'r', encoding="utf-8") as f:
            text = f.read().splitlines()[4:-3]
        path = "cleaned"
        with open(os.path.join(path, file), 'w', encoding="utf-8") as f:
            f.write("\n".join(text))
    
main()


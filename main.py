from scripts.load import load_and_bow_everyth
from scripts.count import total_words

BCS_DIR = "Better_Call_Saul_Transcripts/All_Seasons"
RA_DIR = "RA/RA_split"

MAGIC_NUM = 3847
#average number of words in a Better Call Saul episode

def main():
    
    #Better call saul bag
    BCS = load_and_bow_everyth(BCS_DIR)
    RA = load_and_bow_everyth(RA_DIR)
    print(total_words(BCS),total_words(RA))



if __name__ == "__main__":
    main()
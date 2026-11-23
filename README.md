# CMSC-170-Project-deReal-Relayo-Dalupan

CATgpt (CMSC 170 EF-6L). We classify 10-token data as Philippine Republic Acts (RA) or Better Call Saul (BCS) transcripts, using Naive Bayes, KNN and a Perceptron written from scratch.

## How to run

You only need Python 3. No extra libraries are needed.

First, make the RA paragraphs (only needed once, since they are too many files to put on GitHub):

```
cd RA
python3 backbruh.py
mv RA_single/repacts_with_summary.txt .
python3 cleaner.py
cd ..
```

Then run the program:

```
python3 main.py
```

It runs every method and prints the results at the end. It takes about 15 minutes, mostly because of KNN.

## Settings

The settings are the lists at the top of main.py. The program runs once for every value in a list, so remove values to make it faster.

- LAPLACES: Naive Bayes smoothing
- KS: k in KNN
- PS: KNN distance (1 or 2)
- RS: Perceptron learning rate
- BIASES: Perceptron starting bias

## Other scripts

- Better_Call_Saul_Transcripts/cleaner.py remakes the BCS data (per10, per20, per50). These are already in the repo.
- helper_scripts/parq_to_txt.py turns the original Parquet file into repacts_with_summary.txt. Running RA/split.py on that file remakes RA/RA_split, which is already in the repo. This is the only script that uses pandas.

## Notes

On the RA text files, it was originally one large text file that consisted of 450 thousand lines and 16 million tokens. We decided to split the file into txt files approximately equal in size as an episode in Better Call Saul. Our Better Call Saul data have 242,388 tokens from 63 different episodes. Thus, on average, an episode has about 3,847 tokens. 

We also must say that pandas was not used in any script for this project EXCEPT for helper_scripts/parq_to_txt.py. This script was created to convert the parquet file to txt file, completely irrelevant to any of the project specifications. 

Another important detail is that the actual repacts_with_summary.txt is git ignored because it is larger than 100mb.

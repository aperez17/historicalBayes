# Historical Text Classification
Written by: Alex Perez, Yakov Neshcheretnyy
Date made: 11/8/2016

In this project we have the following folders:

| Folder/Files | Description |
| -------| ---------- |
| ClmetAnalytics | A project which takes the malformed html's created by CLMET and analyzes them and copies them to their correct folder |
| scripts | A few scripts which helped automate most of the process to pre process the corpus using ClmetAnalytics, updating the jar created, and splitting the data into the original 50/50 comparisons|
| *PreProcessed | Describes a PreProcessed corpus by which we parsed and output the text to the correct folder based on its date |
| *-train | Just for sanity purposes we put all the PreProcessed data into separate train folders to make it easier to pass in values |
| clmet | Our training corpus (not included due to volume)|
| nb.py | The original approach to NaiveBayes classification |
| multinomial_np.py | The final approach for NaiveBayes Classification |
| lib | Where we store the ClmetAnalytics jar |

### Getting the Dataset
Unfortunately there is no direct link to download the [Corpus](https://perswww.kuleuven.be/~u0044428/clmet.htm), this is because this was provided to us as a gift by the owner of the corpus.

### Steps to run classifier
1. Update jar by running `./scripts/update_jar.sh`
2. Run a pre processing based on how many periods you wish to split the data into with the script with the argument of the plain text CLMET corpus, for example `./scripts/pre_process_three.sh clmet/corpus/txt/plain`
3. Move the output to a directory with the format {folderName}/train/{year output of preprocess script}
4. Run the classifier with the argument of the training dataset and the number of periods by running `python multinomial_np.py five-train 5`

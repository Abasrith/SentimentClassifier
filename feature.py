import csv
import numpy as np
import argparse

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map

def generateGloveEmbeddings(inputTuple, gloveDictionary, outputFile):

    outputFileContents = []
    for inputTupleIndex in range(len(inputTuple)):
        wordsFoundCount = 0
        label, review = inputTuple[inputTupleIndex]
        # print(f"label:{label}, review:{review}")
        reviewWords = review.split()
        # print(f"reviewWordsLength:{len(reviewWords)}")
        gloveSum = [0] * len(gloveDictionary[next(iter(gloveDictionary))])
        # print(f"length of glove sum: {len(gloveSum)} and {gloveSum}")
        
        for word in reviewWords:
            if word in gloveDict:
                #print(f"Word found:{word}")
                gloveValue = gloveDictionary[word]
                # if word == 'i':
                #     print(f"glove value for i:{gloveValue}")
                #print(f"glove value:{gloveValue}")
                gloveSum  = [accuSum + partialSum for accuSum, partialSum in zip(gloveSum, gloveValue)]
                wordsFoundCount += 1
        #print(f"words found:{wordsFoundCount}")
        if wordsFoundCount > 0:
            avgGloveSum = [(gloveElement / wordsFoundCount) for gloveElement in gloveSum]
            #print(f"glove embedding avg:{avgGloveSum}")
        else:
            avgGloveSum = gloveSum
        
        formattedGloveSum = ["{:.6f}".format(val) for val in avgGloveSum]
        formattedGloveStr = '\t'.join(formattedGloveSum)
        label = "{:.6f}".format(label)
        lineFormat = f"{label}\t{formattedGloveStr}\n"
        outputFileContents.append(lineFormat)
    
    outputFileContents = "".join(outputFileContents)   
    with open(outputFile, "w") as outputFd:
        outputFd.write(outputFileContents)

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, 
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()
    
    gloveInputFile = args.feature_dictionary_in
    
    trainingInputFile = args.train_input
    testingInputFile = args.test_input
    validationInputFile = args.validation_input
    
    trainingOutputFile = args.train_out
    testingOutputFile = args.test_out
    validationOutputFile = args.validation_out

    # gloveInputFile = './glove_embeddings.txt'
    
    # trainingInputFile = './smalldata/train_small.tsv'
    # testingInputFile = './smalldata/test_small.tsv'
    # validationInputFile = './smalldata/val_small.tsv'
    
    # trainingOutputFile = './formatted_train_small_output.tsv'
    # testingOutputFile = './formatted_test_small_output.tsv'
    # validationOutputFile = './formatted_val_small_output.tsv'
        
    # trainingInputFile = './largedata/train_large.tsv'
    # testingInputFile = './largedata/test_large.tsv'
    # validationInputFile = './largedata/val_large.tsv'
     
    # trainingOutputFile = './formatted_train_large_output.tsv'
    # testingOutputFile = './formatted_test_large_output.tsv'
    # validationOutputFile = './formatted_val_large_output.tsv'

    gloveDict = load_feature_dictionary(gloveInputFile)
    trainingInputTuple = load_tsv_dataset(trainingInputFile)
    testInputTuple = load_tsv_dataset(testingInputFile)
    valInputTuple = load_tsv_dataset(validationInputFile)
    
    generateGloveEmbeddings(trainingInputTuple, gloveDict, trainingOutputFile)
    generateGloveEmbeddings(valInputTuple, gloveDict, validationOutputFile)
    generateGloveEmbeddings(testInputTuple, gloveDict, testingOutputFile)
    

    
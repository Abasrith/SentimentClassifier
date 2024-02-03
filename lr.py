import numpy as np
import argparse


def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def train(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    num_epoch : int, 
    learning_rate : float
) -> None:
    # print(f"X shape:{X.shape}")
    # print(f"y shape:{y.shape}")
    for epoch in range(num_epoch):
        for i in range(X.shape[0]):
            x_sample = X[i].reshape(-1,1)
            y_sample = y[i]
            # print(f"thetha shape:{theta.shape}")
            # print(f"x_sample shape:{x_sample.shape}")
            # print(f"y_sample shape:{y_sample.shape}")
            z = np.dot(x_sample.T, theta)
            # print(f"z shape:{z.shape}")
            prediction = sigmoid(z)
            
            gradientTheta = (prediction - y_sample)*x_sample
            # print(f"gradientTheta shape:{gradientTheta.shape}")
                        
            theta -= learning_rate * gradientTheta
    # print(f"thetha in the function:{theta}")

def predict(
    theta : np.ndarray,
    X : np.ndarray
) -> np.ndarray:
    #print(f"theta shape:{theta.shape}")
    #print(f"X shape:{X.shape}")
    z = np.dot(X, theta)
    #print(f"z:{z}")
    prob = sigmoid(z)
    #print(f"prob of predictions:{prob}")
    #print(prob)
    
    predictions = (prob>=0.5).astype(int)
    
    return predictions


def compute_error(
    y_pred : np.ndarray, 
    y : np.ndarray
) -> float:
    correctCount = 0
    inCorrectCount = 0
    
    compare = (y_pred == y)
    correctCount += np.sum(compare)
    inCorrectCount += np.sum(~compare)
    # print(f"correct count:{correctCount}")
    # print(f"incorrect count:{inCorrectCount}")
    
    error = inCorrectCount/(correctCount+inCorrectCount)
    return error


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, 
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()
    
    numEpochs = args.num_epoch
    learningRate = args.learning_rate
    trainingInput = np.loadtxt(args.train_input, dtype=float)
    valInput = np.loadtxt(args.validation_input, dtype=float)
    testingInput = np.loadtxt(args.test_input, dtype=float)
    
    metrics_output = args.metrics_out
    training_labels = args.train_out
    testing_labels = args.test_out
    
    # numEpochs = 500
    # learningRate = 0.1
    
    # train_input = './smalloutput/formatted_train_small.tsv'
    # test_input = './smalloutput/formatted_test_small.tsv'
    # validation_input = './smalloutput/formatted_val_small.tsv'
    
    # train_input = './formatted_train_small_output.tsv'
    # test_input = './formatted_test_small_output.tsv'
    # validation_input = './formatted_val_small_output.tsv'

    # train_input = './formatted_train_large_output.tsv'
    # test_input = './formatted_test_large_output.tsv'
    # validation_input = './formatted_val_large_output.tsv'
        
    # metrics_output = './formatted_metrics.txt'
    # training_labels = './formatted_training_labels.txt'
    # testing_labels = './formatted_testing_labels.txt'
    #val_labels = './formatted_val_labels.txt'

    # trainingInput = np.loadtxt(train_input, dtype=float)
    # testingInput = np.loadtxt(test_input, dtype=float)
    # valInput = np.loadtxt(validation_input, dtype=float)
    
    trainingFeature = trainingInput[:,1:]
    trainingLabel =  trainingInput[:,:1]
    
    trainingFeature = np.column_stack((np.ones(trainingFeature.shape[0]), trainingFeature))
    featureCount = trainingFeature.shape[1]
    # print(f"training features:{trainingFeature}")
    # print(f"training label:{trainingLabel}")

    theta = np.zeros((featureCount,1))
    #print(theta.shape)
    # print(f"thetha before: {theta}")
    train(theta, trainingFeature, trainingLabel, numEpochs, learningRate)
    # print(f"thetha after:{theta}")
    
    predictTrainOut = predict(theta, trainingFeature)
    # print(predictTrainOut)
    trainingError = compute_error(predictTrainOut, trainingLabel)
    trainingError = "{:.6f}".format(trainingError)
    # print(trainingError)
    
    
    # print train label and metrics
    with open(metrics_output, "w") as metricsOutput:
        metricsOutput.write("error(train): "+ trainingError+"\n")
    with open(training_labels, "w") as trainingLabelOut:
        for i in range(len(predictTrainOut)):
            trainingLabelOut.write(str(int(predictTrainOut[i]))+"\n")   
    
    # ValFeature = valInput[:,1:]
    # ValLabel =  valInput[:,1:2]
    # ValFeature = np.column_stack((np.ones(ValFeature.shape[0]), ValFeature))
    # predictValOut = predict(theta, ValFeature)
    # valError = compute_error(predictValOut, ValLabel)

    testFeature = testingInput[:,1:]
    testLabel =  testingInput[:,:1]
    testFeature = np.column_stack((np.ones(testFeature.shape[0]), testFeature))
    predictTestOut = predict(theta, testFeature)
    testingError = compute_error(predictTestOut, testLabel)
    testingError = "{:.6f}".format(testingError)
    
    # #print test label and metrics
    with open(metrics_output, "a") as metricsOutput:
        metricsOutput.write("error(test): "+str(testingError))
    with open(testing_labels, "w") as testingLabelOut:
        for i in range(len(predictTestOut)):
            testingLabelOut.write(str(int(predictTestOut[i]))+"\n")  
import argparse
from numpy import ndarray
from typing import Any, Union, List, Dict, Tuple
from utils.gradcheck import gradcheck_naive, grad_tests_softmax, grad_tests_negsamp
from utils.utils import normalizeRows, softmax


NumericType = Union[float, List[float], ndarray]
TokenType = Dict[str, int]
DataSetType = Any


def sigmoid(x: NumericType) -> NumericType:
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """
    from numpy import exp

    # YOUR CODE HERE (~1 Line)
    ans = exp(-1 * x)
    ans = 1 / (1 + ans)
    # END YOUR CODE
    return ans


def naiveSoftmaxLossAndGradient(centerWordVec: ndarray, outsideWordIdx: int, outsideVectors: ndarray,
                                dataset: DataSetType) -> Tuple[float, ndarray, ndarray]:
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models. For those unfamiliar with numpy notation, note 
    that a numpy ndarray with a shape of (x, ) is a one-dimensional array, which
    you can effectively treat as a vector with length x.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    in shape (word vector length, )
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors is
                    in shape (num words in vocab, word vector length) 
                    for all words in vocab (tranpose of U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """
    from numpy import matmul, zeros, log

    # YOUR CODE HERE (~6-8 Lines)
    num_voc, dim_vec = outsideVectors.shape
    assert dim_vec == len(centerWordVec)
    prod = matmul(outsideVectors, centerWordVec)
    y_hat = softmax(prod)
    y = zeros(num_voc, dtype=y_hat.dtype)
    y[outsideWordIdx] = 1
    # package ans
    loss = -1 * log(y_hat[outsideWordIdx])
    gradCenterVec = matmul(outsideVectors.T, y_hat - y)
    gradOutsideVecs = y_hat - y
    gradOutsideVecs = matmul(gradOutsideVecs.reshape([-1, 1]), centerWordVec.reshape([1, -1]))
    # END YOUR CODE
    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx: int, dataset: DataSetType, K: int) -> List[int]:
    """
        Samples K indexes which are not the outsideWordIdx
    """
    negSampleWordIndices = [0] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(centerWordVec: ndarray, outsideWordIdx: int, outsideVectors: ndarray,
                               dataset: DataSetType, K: int=10) -> Tuple[float, ndarray, ndarray]:
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """
    from numpy import matmul, log, take, zeros

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    # YOUR CODE HERE (~10 Lines)
    num_voc, dim_vec = outsideVectors.shape
    assert dim_vec == len(centerWordVec)
    # compute loss function
    sub_outside = take(outsideVectors, indices, axis=0)
    sub_outside[0, :] *= -1
    coef = sigmoid(x=-matmul(sub_outside, centerWordVec))
    loss = -1 * log(coef).sum()
    # \partial J \partial v_c
    gradCenterVec = matmul(sub_outside.T, 1 - coef)
    # \partial J \partial U
    gradOutsideVecs = zeros(shape=outsideVectors.shape)
    gradOutsideVecs[outsideWordIdx, :] = -1 * (1 - coef[0]) * centerWordVec
    for idx in range(K):
        gradOutsideVecs[negSampleWordIndices[idx], :] += (1 - coef[idx + 1]) * centerWordVec
    # END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (V in pdf handout)
    outsideVectors -- outside vectors is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (transpose of U in the pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """
    from numpy import zeros
    loss = 0.0
    gradCenterVecs = zeros(centerWordVectors.shape)
    gradOutsideVectors = zeros(outsideVectors.shape)

    # YOUR CODE HERE (~8 Lines)
    centerWordIdx = word2Ind[currentCenterWord]
    centerWordVec = centerWordVectors[centerWordIdx]
    for outsideWord in outsideWords:
        outsideWordIdx = word2Ind[outsideWord]
        outsidePart = word2vecLossAndGradient(centerWordVec=centerWordVec, outsideWordIdx=outsideWordIdx,
                                              outsideVectors=outsideVectors, dataset=dataset)
        loss += outsidePart[0]
        gradCenterVecs[centerWordIdx, :] += outsidePart[1]
        gradOutsideVectors += outsidePart[2]
    # END YOUR CODE
    
    return loss, gradCenterVecs, gradOutsideVectors


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset,
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    from numpy import zeros
    from random import randint
    batchsize = 50
    loss = 0.0
    grad = zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N/2),:]
    outsideVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        windowSize1 = randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        loss += c / batchsize
        grad[:int(N/2), :] += gin / batchsize
        grad[int(N/2):, :] += gout / batchsize

    return loss, grad


def test_sigmoid() -> None:
    """
        Test sigmoid function
    """
    from numpy import allclose, array
    print("=== Sanity check for sigmoid ===")
    assert sigmoid(0) == 0.5
    assert allclose(sigmoid(array([0])), array([0.5]))
    assert allclose(sigmoid(array([1, 2, 3])), array([0.73105858, 0.88079708, 0.95257413]))
    print("Tests for sigmoid passed!")


def getDummyObjects() -> Tuple[DataSetType, ndarray, TokenType]:
    """
        Helper method for naiveSoftmaxLossAndGradient and negSamplingLossAndGradient tests
    """
    from random import seed, randint
    from numpy.random import seed as np_seed, randn
    seed(31415)
    np_seed(9265)

    def dummySampleTokenIdx(): return randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[randint(0, 4)], [tokens[randint(0, 4)] for i in range(2*C)]

    dataset = type('dummy', (), {})()
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    dummy_vectors = normalizeRows(randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])
    return dataset, dummy_vectors, dummy_tokens


def test_naiveSoftmaxLossAndGradient() -> None:
    """
        Test naiveSoftmaxLossAndGradient
    """
    from numpy.random import randn

    dataset, dummy_vectors, dummy_tokens = getDummyObjects()
    print("==== Gradient check for naiveSoftmaxLossAndGradient ====")

    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = naiveSoftmaxLossAndGradient(vec, 1, dummy_vectors, dataset)
        return loss, gradCenterVec
    gradcheck_naive(temp, randn(3), "naiveSoftmaxLossAndGradient gradCenterVec")
    centerVec = randn(3)

    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = naiveSoftmaxLossAndGradient(centerVec, 1, vec, dataset)
        return loss, gradOutsideVecs
    gradcheck_naive(temp, dummy_vectors, "naiveSoftmaxLossAndGradient gradOutsideVecs")


def test_negSamplingLossAndGradient() -> None:
    """
        Test negSamplingLossAndGradient
    """
    from numpy.random import randn

    dataset, dummy_vectors, dummy_tokens = getDummyObjects()
    print("==== Gradient check for negSamplingLossAndGradient ====")

    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = negSamplingLossAndGradient(vec, 1, dummy_vectors, dataset)
        return loss, gradCenterVec
    gradcheck_naive(temp, randn(3), "negSamplingLossAndGradient gradCenterVec")
    centerVec = randn(3)

    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = negSamplingLossAndGradient(centerVec, 1, vec, dataset)
        return loss, gradOutsideVecs
    gradcheck_naive(temp, dummy_vectors, "negSamplingLossAndGradient gradOutsideVecs")


def test_skipgram() -> None:
    """
        Test skip-gram with naiveSoftmaxLossAndGradient
    """
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naiveSoftmaxLossAndGradient),
        dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")
    grad_tests_softmax(skipgram, dummy_tokens, dummy_vectors, dataset)

    print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient),
        dummy_vectors, "negSamplingLossAndGradient Gradient")
    grad_tests_negsamp(skipgram, dummy_tokens, dummy_vectors, dataset, negSamplingLossAndGradient)


def test_word2vec() -> None:
    """
        Test the two word2vec implementations, before running on Stanford Sentiment Treebank
    """
    test_sigmoid()
    test_naiveSoftmaxLossAndGradient()
    test_negSamplingLossAndGradient()
    test_skipgram()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test your implementations.')
    parser.add_argument('function', nargs='?', type=str, default='all',
                        help='Name of the function you would like to test.')

    args = parser.parse_args()
    if args.function == 'sigmoid':
        test_sigmoid()
    elif args.function == 'naiveSoftmaxLossAndGradient':
        test_naiveSoftmaxLossAndGradient()
    elif args.function == 'negSamplingLossAndGradient':
        test_negSamplingLossAndGradient()
    elif args.function == 'skipgram':
        test_skipgram()
    elif args.function == 'all':
        test_word2vec()

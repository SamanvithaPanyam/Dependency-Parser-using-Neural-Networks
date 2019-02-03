import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util

"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon (2017)
Modified by: Jun S. Kang (2018 Mar)
"""


class DependencyParserModel(object):

    def __init__(self, graph, embedding_array, Config):

        self.build_graph(graph, embedding_array, Config)

    def build_graph(self, graph, embedding_array, Config):
        """

        :param graph:
        :param embedding_array:
        :param Config:
        :return:
        """

        with graph.as_default():
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)

            """
            ===================================================================

            Define the computational graph with necessary variables.

            1) You may need placeholders of:
                - Many parameters are defined at Config: batch_size, n_Tokens, etc
                - # of transitions can be get by calling parsing_system.numTransitions()

            self.train_inputs = 
            self.train_labels = 
            self.test_inputs =
            ...


            2) Call forward_pass and get predictions

            ...
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)


            3) Implement the loss function described in the paper
             - lambda is defined at Config.lam

            ...
            self.loss =

            ===================================================================
            """

            self.train_inputs = tf.placeholder(tf.int32, shape=[Config.batch_size, Config.n_Tokens])
            self.train_labels = tf.placeholder(tf.int32, shape = [Config.batch_size, parsing_system.numTransitions()])

            self.test_inputs = tf.placeholder(tf.int32, shape = [Config.n_Tokens])

            train_embeddings = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

            # layer1
            print("running 1 layer")

            train_embeddings = tf.reshape(train_embeddings,[Config.batch_size, -1])
            weights_input = tf.Variable(tf.truncated_normal([Config.embedding_size * Config.n_Tokens, Config.hidden_size], stddev = 0.1))
            biases_input = tf.Variable(tf.zeros([Config.hidden_size]))
            weights_output = tf.Variable(tf.truncated_normal([Config.hidden_size, parsing_system.numTransitions()], stddev = 0.1))
            self.prediction = self.forward_pass(train_embeddings, weights_input, biases_input, weights_output)

            train_labels = tf.nn.relu(self.train_labels)

            ce_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.prediction, labels=train_labels)
            l2_loss = Config.lam * (tf.nn.l2_loss(train_embeddings) + tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(biases_input) + tf.nn.l2_loss(weights_output))
            self.loss = tf.reduce_mean(ce_loss + l2_loss)

            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)

            #self.app = optimizer.apply_gradients(grads)

            # For test data, we only need to get its prediction
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            self.test_pred = self.forward_pass(test_embed, weights_input, biases_input, weights_output)

            # intializer
            self.init = tf.global_variables_initializer()

            # 2 hidden layers
            # in this experiment, model is tested with by calling the forward_pass method twice.Output of first hidden layer is being passed as input to the second hidden layer.
            # print("running 2 hidden layers")
            # train_embeddings = tf.reshape(train_embeddings, [Config.batch_size, -1])
            # weights_input1 = tf.Variable(tf.truncated_normal([Config.embedding_size * Config.n_Tokens, Config.hidden_size], stddev = 0.1))
            # weights_input2 = tf.Variable(tf.truncated_normal([Config.hidden_size, Config.hidden_size], stddev = 0.1))
            # biases_input1 = tf.Variable(tf.zeros([Config.hidden_size]))
            # biases_input2 = tf.Variable(tf.zeros([Config.hidden_size]))
            # weights_output = tf.Variable( tf.truncated_normal([Config.hidden_size, parsing_system.numTransitions()], stddev = 0.1))
            #
            # layer1 = self.forward_pass(train_embeddings, weights_input1, biases_input1, weights_input2)
            #
            # self.prediction = self.forward_pass(layer1, weights_input2, biases_input2, weights_output)
            #
            # train_labels = tf.nn.relu(self.train_labels)
            # ce_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.prediction, labels = train_labels)
            # l2_loss = Config.lam * (tf.nn.l2_loss(train_embeddings) + tf.nn.l2_loss(weights_input1) + tf.nn.l2_loss(weights_input2) + tf.nn.l2_loss(biases_input1) + tf.nn.l2_loss(biases_input2) + tf.nn.l2_loss(weights_output))
            #
            # self.loss = tf.reduce_mean(ce_loss + l2_loss)
            #
            # optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            # grads = optimizer.compute_gradients(self.loss)
            # clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            # self.app = optimizer.apply_gradients(clipped_grads)
            #
            # # For test data, we only need to get its prediction
            #
            # test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            # test_embed = tf.reshape(test_embed, [1, -1])
            # test_layer1 = self.forward_pass(test_embed, weights_input1, biases_input1, weights_input2)
            #
            # self.test_pred = self.forward_pass(test_layer1, weights_input2, biases_input2, weights_output)
            #
            # # intializer
            # self.init = tf.global_variables_initializer()



            # 3 hidden layers
            #in this experiment, model is tested with by calling the forward_pass method thrice.Output of first hidden layer is being passed as input to the second hidden layer and so on.
            print("running 3 hidden layers")
            # train_embeddings = tf.reshape(train_embeddings, [Config.batch_size, -1])
            # weights_input1 = tf.Variable(tf.truncated_normal([Config.embedding_size * Config.n_Tokens, Config.hidden_size], stddev = 0.1))
            # weights_input2 = tf.Variable(tf.truncated_normal([Config.hidden_size, Config.hidden_size], stddev = 0.1))
            # weights_input3 = tf.Variable(tf.truncated_normal([Config.hidden_size, Config.hidden_size], stddev = 0.1))
            # biases_input1 = tf.Variable(tf.zeros([Config.hidden_size]))
            # biases_input2 = tf.Variable(tf.zeros([Config.hidden_size]))
            # biases_input3 = tf.Variable(tf.zeros([Config.hidden_size]))
            # weights_output = tf.Variable(tf.truncated_normal([Config.hidden_size, parsing_system.numTransitions()], stddev = 0.1))
            #
            # layer1 = self.forward_pass(train_embeddings, weights_input1, biases_input1, weights_input2)
            # layer2 = self.forward_pass(layer1, weights_input2, biases_input2, weights_input3)
            # self.prediction = self.forward_pass(layer2, weights_input3, biases_input3, weights_output)
            #
            # train_labels = tf.nn.relu(self.train_labels)
            # ce_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.prediction, labels = train_labels)
            #
            # l2_loss = (Config.lam) * (tf.nn.l2_loss(train_embeddings) + tf.nn.l2_loss(weights_input1)+ tf.nn.l2_loss(weights_input2) + tf.nn.l2_loss(weights_input3) + tf.nn.l2_loss(biases_input1) + tf.nn.l2_loss(biases_input2) + tf.nn.l2_loss(biases_input3) + tf.nn.l2_loss(weights_output))
            #
            # self.loss = tf.reduce_mean(ce_loss + l2_loss)
            #
            # optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            # grads = optimizer.compute_gradients(self.loss)
            # clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            # self.app = optimizer.apply_gradients(clipped_grads)
            #
            # test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            # test_embed = tf.reshape(test_embed, [1, -1])
            # test_layer1 = self.forward_pass(test_embed, weights_input1, biases_input1, weights_input2)
            # test_layer2 = self.forward_pass(test_layer1, weights_input2, biases_input2, weights_input3)
            # self.test_pred = self.forward_pass(test_layer2, weights_input3, biases_input3, weights_output)
            #
            # # intializer
            # self.init = tf.global_variables_initializer()


            #   word, pos, label embeddings
            # print("running word, pos and label parallelly")
            #
            # train_embeddings = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            # self.train_input_words, self.train_input_pos, self.train_input_labels = tf.split(self.train_inputs,[18,18,12],1)
            # word_embeddings = tf.nn.embedding_lookup(self.embeddings, self.train_input_words)
            # word_embeddings = tf.reshape(word_embeddings, [Config.batch_size, -1])
            # pos_embeddings = tf.nn.embedding_lookup(self.embeddings, self.train_input_pos)
            # pos_embeddings = tf.reshape(pos_embeddings, [Config.batch_size, -1])
            # label_embeddings = tf.nn.embedding_lookup(self.embeddings, self.train_input_labels)
            # label_embeddings = tf.reshape(label_embeddings, [Config.batch_size, -1])
            #
            # weights_input_words = tf.Variable(tf.truncated_normal([Config.embedding_size * 18, Config.hidden_size], stddev = 0.1))
            # weights_input_pos = tf.Variable(tf.truncated_normal([Config.embedding_size * 18, Config.hidden_size], stddev = 0.1))
            # weights_input_labels = tf.Variable(tf.truncated_normal([Config.embedding_size * 12, Config.hidden_size], stddev = 0.1))
            # biases_input_words = tf.Variable(tf.zeros([Config.hidden_size]))
            # biases_input_pos = tf.Variable(tf.zeros([Config.hidden_size]))
            # biases_input_labels = tf.Variable(tf.zeros([Config.hidden_size]))
            # weights_output_words = tf.Variable(tf.truncated_normal([Config.hidden_size,parsing_system.numTransitions()], stddev = 0.1))
            # weights_output_pos = tf.Variable(tf.truncated_normal([Config.hidden_size,parsing_system.numTransitions()],stddev = 0.1))
            # weights_output_labels = tf.Variable(tf.truncated_normal([Config.hidden_size,parsing_system.numTransitions()],stddev = 0.1))
            #
            # self.prediction_words = self.forward_pass(word_embeddings, weights_input_words, biases_input_words, weights_output_words)
            # self.prediction_pos = self.forward_pass(pos_embeddings, weights_input_pos, biases_input_pos, weights_output_pos)
            # self.prediction_labels = self.forward_pass(label_embeddings, weights_input_labels, biases_input_labels, weights_output_labels)
            #
            # train_labels = tf.nn.relu(self.train_labels)
            # ce_loss_words = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.prediction_words, labels=train_labels)
            # ce_loss_pos = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.prediction_pos, labels=train_labels)
            # ce_loss_labels = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.prediction_labels, labels=train_labels)
            #
            # l2_loss = (Config.lam) * (tf.nn.l2_loss(word_embeddings) + tf.nn.l2_loss(pos_embeddings) + tf.nn.l2_loss(label_embeddings) +tf.nn.l2_loss(weights_input_words) +tf.nn.l2_loss(weights_input_pos) +tf.nn.l2_loss(weights_input_labels)+ tf.nn.l2_loss(biases_input_words)+ tf.nn.l2_loss(biases_input_pos)+ tf.nn.l2_loss(biases_input_labels) + tf.nn.l2_loss(weights_output_words)+  tf.nn.l2_loss(weights_output_pos)+ tf.nn.l2_loss(weights_output_labels))
            #
            # self.loss = tf.reduce_mean(ce_loss_words + ce_loss_pos + ce_loss_labels +  l2_loss )
            #
            # optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            # grads = optimizer.compute_gradients(self.loss)
            # clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            # self.app = optimizer.apply_gradients(clipped_grads)
            #
            # test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            # test_embed_words, test_embed_pos, test_embed_labels = tf.split(test_embed,[18,18,12],1)
            # test_embed_words = tf.reshape(test_embed_words, [1, -1])
            # test_embed_pos = tf.reshape(test_embed_pos, [1, -1])
            # test_embed_labels = tf.reshape(test_embed_labels, [1, -1])
            #
            # test_words = self.forward_pass(test_embed_words, weights_input_words, biases_input_words, weights_output_words)
            # test_pos = self.forward_pass(test_embed_pos, weights_input_pos, biases_input_pos, weights_output_pos)
            # test_labels = self.forward_pass(test_embed_labels, weights_input_labels, biases_input_labels, weights_output_labels)
            #
            # self.test_pred = (test_words + test_pos + test_labels) / 3
            #
            # # intializer
            # self.init = tf.global_variables_initializer()

    def train(self, sess, num_steps):
        """

        :param sess:
        :param num_steps:
        :return:
        """
        self.init.run()
        print "Initailized"

        average_loss = 0
        for step in range(num_steps):
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            _, loss_val = sess.run([self.app, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print "Average loss at step ", step, ": ", average_loss
                average_loss = 0
            if step % Config.validation_step == 0 and step != 0:
                print "\nTesting on dev set at step ", step
                predTrees = []
                for sent in devSents:
                    numTrans = parsing_system.numTransitions()

                    c = parsing_system.initialConfiguration(sent)
                    while not parsing_system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = parsing_system.transitions[j]

                        c = parsing_system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = parsing_system.evaluate(devSents, predTrees, devTrees)
                print result

        print "Train Finished."

    def evaluate(self, sess, testSents):
        """

        :param sess:
        :return:
        """

        print "Starting to predict on test set"
        predTrees = []
        for sent in testSents:
            numTrans = parsing_system.numTransitions()

            c = parsing_system.initialConfiguration(sent)
            while not parsing_system.isTerminal(c):
                # feat = getFeatureArray(c)
                feat = getFeatures(c)
                pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = parsing_system.transitions[j]

                c = parsing_system.apply(c, optTrans)

            predTrees.append(c.tree)
        print "Saved the test results."
        Util.writeConll('result_test.conll', testSents, predTrees)

    def forward_pass(self, embed, weights_input, biases_input, weights_output):

        hidden_layer = tf.add(tf.matmul(embed, weights_input), biases_input)
        #cube function
        hidden_layer = tf.pow(hidden_layer, 3)
        #tanh function
        #hidden_layer = tf.nn.tanh(hidden_layer)
        # relu function
        #hidden_layer = tf.nn.relu(hidden_layer)
        #sigmoid Function
        #hidden_layer = tf.nn.sigmoid(hidden_layer)
        output_layer = tf.matmul(hidden_layer, weights_output)
        return output_layer


def genDictionaries(sents, trees):
    word = []
    pos = []
    label = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n + 1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL, rootLabel]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1

    return wordDict, posDict, labelDict


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]


def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]


def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]



def getFeatures(c):
    wordFeatures = []
    posFeatures = []
    labelFeatures = []
    i = 2;
    while (i >= 0):
        idx = c.getStack(i)
        wordFeatures.append(getWordID(c.getWord(idx)))
        posFeatures.append(getPosID(c.getPOS(idx)))
        i = i - 1
    for i in range(0, 3):
        idx = c.getBuffer(i)
        wordFeatures.append(getWordID(c.getWord(idx)))
        posFeatures.append(getPosID(c.getPOS(idx)))

    for j in range(1, -1, -1):
        idx = c.getStack(j)

        childIdx = c.getLeftChild(idx, 1)
        wordFeatures.append(getWordID(c.getWord(childIdx)))
        posFeatures.append(getPosID(c.getPOS(childIdx)))
        labelFeatures.append(getLabelID(c.getLabel(childIdx)))

        childIdx = c.getRightChild(idx, 1)
        wordFeatures.append(getWordID(c.getWord(childIdx)))
        posFeatures.append(getPosID(c.getPOS(childIdx)))
        labelFeatures.append(getLabelID(c.getLabel(childIdx)))

        childIdx = c.getLeftChild(idx, 2)
        wordFeatures.append(getWordID(c.getWord(childIdx)))
        posFeatures.append(getPosID(c.getPOS(childIdx)))
        labelFeatures.append(getLabelID(c.getLabel(childIdx)))

        childIdx = c.getRightChild(idx, 2)
        wordFeatures.append(getWordID(c.getWord(childIdx)))
        posFeatures.append(getPosID(c.getPOS(childIdx)))
        labelFeatures.append(getLabelID(c.getLabel(childIdx)))

        childIdx = c.getLeftChild(c.getLeftChild(idx, 1), 1)
        wordFeatures.append(getWordID(c.getWord(childIdx)))
        posFeatures.append(getPosID(c.getPOS(childIdx)))
        labelFeatures.append(getLabelID(c.getLabel(childIdx)))

        childIdx = c.getRightChild(c.getRightChild(idx, 1), 1)
        wordFeatures.append(getWordID(c.getWord(childIdx)))
        posFeatures.append(getPosID(c.getPOS(childIdx)))
        labelFeatures.append(getLabelID(c.getLabel(childIdx)))

    allFeatures = []
    allFeatures.extend(wordFeatures)
    allFeatures.extend(posFeatures)
    allFeatures.extend(labelFeatures)

    return allFeatures

    """
    =================================================================
    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)
    =================================================================
    """


def genTrainExamples(sents, trees):
    numTrans = parsing_system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    for i in pbar(range(len(sents))):
        if trees[i].isProjective():
            c = parsing_system.initialConfiguration(sents[i])

            while not parsing_system.isTerminal(c):
                oracle = parsing_system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.canApply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)

                if 1.0 not in label:
                    print i, label
                features.append(feat)
                labels.append(label)
                c = parsing_system.apply(c, oracle)
    return features, labels


def load_embeddings(filename, wordDict, posDict, labelDict):
    dictionary, word_embeds = pickle.load(open(filename, 'rb'))

    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = wordDict.keys()
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01
    print "Found embeddings: ", foundEmbed, "/", len(knownWords)

    return embedding_array


if __name__ == '__main__':

    wordDict = {}
    posDict = {}
    labelDict = {}
    parsing_system = None

    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    embedding_filename = 'word2vec.model'

    embedding_array = load_embeddings(embedding_filename, wordDict, posDict, labelDict)

    labelInfo = []
    for idx in np.argsort(labelDict.values()):
        labelInfo.append(labelDict.keys()[idx])
    parsing_system = ParsingSystem(labelInfo[1:])
    print parsing_system.rootLabel

    print "Generating Traning Examples"
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    print "Done."

    # Build the graph model
    graph = tf.Graph()
    model = DependencyParserModel(graph, embedding_array, Config)

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:

        model.train(sess, num_steps)

        model.evaluate(sess, testSents)

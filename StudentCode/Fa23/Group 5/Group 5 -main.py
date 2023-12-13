'''
How to run:

Works on Python version 3.6 (conda env used for projects)

Install dependencies:
    pip install nltk
    pip install tensorflow
    pip install tflearn

Invoke the following command in the directory containing main.py and WikiQA-train.txt:
    python main.py
'''

import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
import string
import os.path
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import random
import json
import tflearn
import pickle

def parseinput(fname):
    if not os.path.isfile('intents.json'):
        dictlist = []
        c = 0
        f = open('WikiQA-train.txt', 'r', errors='ignore')
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            parts = line.split("\t")
            if parts[2] == '1':
                question = parts[0]
                answer = parts[1]
                tag = str(c)
                dict = {'tag':tag, 'patterns':[question], 'responses':[answer], 'context_set':''}
                dictlist.append(dict)
                c += 1
        final = {'intents':dictlist}
        with open('intents.json', 'w') as outfile:
            json.dump(final, outfile)
        f.close()
        


parseinput('WikiQA-train.txt')

with open('intents.json') as file:
    data = json.load(file)

stemmer = LancasterStemmer()

try:
    
    with open("intents.pickle", "rb") as file:
        words, labels, training, output = pickle.load(file)
except:
    # list of words into words
    # list of patterns and their corresponding tags into the assocaited lists
    # list of all tags into labels
    nltk.download('punkt')
    words, labels, patterns, tags = [], [], [], []
    
    for intent in data['intents']:
        for pattern in intent['patterns']:
            uniqueWords = nltk.word_tokenize(pattern)
            words.extend(uniqueWords)
            patterns.append(uniqueWords)
            tags.append(intent['tag'])
    
    # Add to labels all tags that are in the json file
    labels = [intent["tag"] for intent in data["intents"] if intent["tag"] not in labels]
    labels = sorted(labels)
    
    # Preprocessing
    newWords = []
    
    #Stem each word: Make each word the most base form of each i.e. coding = code, maximum = max
    for word in words:
        if word != "?":
            newWords.append(stemmer.stem(word.lower()))
    
    newWords = sorted(list(set(newWords)))
    words = newWords
    
    #1 hot encoding: figure out the frequency of each word inside of the sentence, and create a frequency array based off of it 
    #training will include our big data of how many ever words are in the sentence and whether specific word exists or not
    #output are tags that correspond to the response
    training, output = [], []
    
    for index, patternWords in enumerate(patterns):
        uniqueWords, bag, rowOut = [], [], []
        for word in patternWords:
            uniqueWords.append(stemmer.stem(word.lower()))
        
        for word in words:
            bag.append(1) if word in uniqueWords else bag.append(0)
        
        training.append(bag)
    
        for _ in range(len(labels)):
            rowOut.append(0)
        
        rowOut[labels.index(tags[index])] = 1
        output.append(rowOut)
    
    
    training = np.array(training)
    output = np.array(output)

    with open("intents.pickle", "wb") as file:
        pickle.dump((words, labels, training, output), file)

# input data layer
net = tflearn.input_data(shape=[None, len(training[0])])

# 3 dense layers (each neuron in each layer is connected to each neuron in preceding layer)
# multi-layered neural network
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
# output layer where scores are converted into probabilities
# set restore equal to false so that the weights are not restored when finetuning 
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")

# defines the training configuration, optional parameters include "loss", "optimizer", "metric"
net = tflearn.regression(net, metric='accuracy')

# create an instance of a deep neural network object using previosuly defined layers
model = tflearn.DNN(net)

# pass in our data and start training the model
# batch size represents the number of training examples used in an iteration
# we should increase this later to 32 to make our model more stable
# we can turn show_metric off later, it is good to keep for testing purposes
model.fit(training, output, n_epoch=1000, batch_size=32, show_metric=True)
model.save("model.tflearn")


# fine tune the data set (like what Anthony was saying we could do as an additional feature)
def finetune(new_training, new_output):
    # we need to add more code for pre-processing the new data

    model.load("model.tflearn")

    model.fit(new_training, new_output, n_epoch=10, batch_size=8, show_metric=True)

    model.save("model.tflearn")

# tokenize each word in the sentence and predict the probability of that neuron in the sentence
def generateSentenceTokens(sentence, bagOfWords):
    newBag = []
    for index in range(len(bagOfWords)):
        newBag.append(0)
    
    sentenceTokens = nltk.word_tokenize(sentence)
    
    #make sure to stem the words
    sentenceTokens = [stemmer.stem(word.lower()) for word in sentenceTokens]

    # fill the bag depending on the frequency of each word
    for tokens in sentenceTokens:
        for index, words in enumerate(bagOfWords):
            if words == tokens:
                newBag[index] = 1

    return np.array(newBag)
    


def chat():
    print("Start talking with HokieBot! To stop the conversation, type quit, exit, or leave.")
    print("DISCLAIMER: HokieBot will attempt to use any prompt for a response, regardless of clarity.")
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit" or sentence.lower() == "exit" or sentence.lower() == "leave":
            break
        
        # predicts the probablitiles for each token in the sentence
        responses = model.predict([generateSentenceTokens(sentence, words)])
        max_index = np.argmax(responses)

        #the predicted tag
        tag = labels[max_index]

        for cur_tag in data["intents"]:
            if cur_tag["tag"] == tag:
                responses = cur_tag["responses"]
        
        print(random.choice(responses))


    
chat()
# made using python 3.6

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import os

#open and read the json file
with open("intents.json") as file:
	data = json.load(file)

try: # has the data been processed once, if yes, then open the data.pickle file
	with open("data.pickle", "rb") as f:
		words, labels, training, output = pickle.load(f)
except: # if no, then process the data
	words = [] # store all the word tokens occuring in patterns
	labels = [] # store the tags
	docs_x = [] # store all the word tokens in the list
	docs_y = [] # store the tags corrosponding to the above pattern

	#store the data in the lists
	for intent in data["intents"]:
		for pattern in intent["patterns"]:
			wrds = nltk.word_tokenize(pattern)
			words.extend(wrds)
			docs_x.append(wrds)
			docs_y.append(intent["tag"])

		if intent["tag"] not in labels:
			labels.append(intent["tag"])

	# How many words has our model seen?
	words = [stemmer.stem(w.lower()) for w in words if w != "?"] # cause we don't need question marks
	words = sorted(list(set(words))) # this removes all the dublicate words cause its a set

	labels = sorted(labels)

	training = []
	output = []

	out_empty = [0 for _ in range(len(labels))]

	for x, doc in enumerate(docs_x):
		bag = []

		wrds = [stemmer.stem(w.lower()) for w in doc]

		for w in words:
			if w in wrds:
				bag.append(1)
			else:
				bag.append(0)

		output_row = out_empty[:] # make a copy
		output_row[labels.index(docs_y[x])] = 1 # look for the tags in the labels list and set it = 1 

		training.append(bag) # the input was this
		output.append(output_row) # this is the output after finding the labels

	training = numpy.array(training)
	output = numpy.array(output)

	with open("data.pickle", "wb") as f: # store the model in data.pickle
		pickle.dump((words, labels, training, output), f)

# building out model with tflearn now

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8) # Hidden layer
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #get probability of each output
net = tflearn.regression(net)

model = tflearn.DNN(net)

if os.path.exists("model.tflearn" + ".meta"): # has the model already been trained once?
	model.load("model.tflearn")
else: # if no, then go here->
	model.fit(training, output, n_epoch=1000, batch_size=8, show_metric = True)
	model.save("model.tflearn")

# start making predictions
def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words] # lowercase the tokens

	for se in s_words:
		for i, w in enumerate(words):
			if w == se: # if the tokens are found in words(which stores tokens)
				bag[i] = 1 # then mark its index as 1

	return numpy.array(bag)

def chat():
	print("Start talking with the bot (type quit to stop)!")
	while True:
		inp = input("You: ")
		if inp.lower() == "quit":
			break

		results = model.predict([bag_of_words(inp, words)])[0]
		results_index = numpy.argmax(results)
		tag = labels[results_index]

		if results[results_index] > 0.7: # if the probability of the word is greater than 70% (our threshold), then return the response with the highest prob		
			for tg in data["intents"]:
				if tg['tag'] == tag:
					response = tg['responses']

			print(random.choice(response))
		else: # if not, then return the error statement
			print("I didn't get that, try again.")

chat()
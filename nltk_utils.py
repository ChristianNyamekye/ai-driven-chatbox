import nltk
import numpy as np
# nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
  return nltk.word_tokenize(sentence)


def stem(word):
  return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
  tokenized_sentence = [stem(w) for w in tokenized_sentence]
  bag = np.zeros(len(all_words), dtype= np.float32)
  for idx, word in enumerate(all_words):
    
    if word in tokenized_sentence:
      bag[idx] = 1.0
      
  return bag
   

# a = ["hello", "how", "are", "you"]
# b = ["hi", "hello", "I", "you", "bye", "thank", "cool"]

# print(bag_of_words(a, b))

# a = "How long does delivery take?"
# print(a)
# a = tokenize(a)
# print(a) 

# words = ["organize", "organizing", "organized"]
# print(words)
# print([stem(w) for w in words])
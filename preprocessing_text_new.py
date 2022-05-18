from nltk.corpus import wordnet as wn
from bs4 import BeautifulSoup
import re
import nltk

#This can have thress types of data
#puncuatation marks removal
#Html tag removals
stop_words=set({'a','about','above','after','again','against','all','am',
 'an','and','any','are','as','at','be','because','been','before','being',
 'below','between','both','but','by','did','do','does','doing','down',
 'during','each','few','for','from','further','had','has','have','having',
 'he','her','here','hers',
 'herself','him','himself','his','how','i','if','in','into','is','it',
 "it's",
 'its',
 'itself',
 'me',
 'more',
 'most',
 'my',
 'myself',
 'of',
 'off',
 'on',
 'once',
 'or',
 'other',
 'our',
 'ours',
 'ourselves',
 'out',
 'over',
 'own',
 'same',
 'she',
 "she's",
 'so',
 'some',
 'such',
 'than',
 'that',
 "that'll",
 'the',
 'their',
 'theirs',
 'them',
 'themselves',
 'then',
 'there',
 'these',
 'they',
 'this',
 'those',
 'through',
 'to',
 'too',
 'under',
 'until',
 'up',
 'very',
 'was',
 'we',
 'were',
 'what',
 'when',
 'where',
 'which',
 'while',
 'who',
 'whom',
 'why',
 'with',
 'you',
 "you'd",
 "you'll",
 "you're",
 "you've",
 'your',
 'yours',
 'yourself',
 'yourselves'})
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
#English stop words are imported
#Stemmer has also been instantiated but not used

#nltk.download('wordnet')
def preprocessing(sentence):  
  #In first phase we will do the decontraction of the words
  nltk.download('wordnet')
  nltk.download('omw-1.4')
  sentence=decontracted(sentence)
  #We will remove any html tags
  sentence=BeautifulSoup(sentence,'lxml').get_text()
  #Now we will remove the special characters
  sentence=re.sub('[^a-zA-Z]+',' ',sentence)
  #Also remove the number we have
  sentence=re.sub('\d+','',sentence)
  #Now we will remove
  sentence=sentence.lower()
  sentence=' '.join([wn.morphy(word) if wn.morphy(word)!=None else word for word in sentence.split() if word not in stop_words])
  return sentence



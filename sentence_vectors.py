from tqdm import tqdm
import pickle
import numpy as np
#we have to load the word2idf weigths and
#word_vec
def sentence_vector(s1:list,word2idf,word_vec):
  sent_vectors=[]
  print("Running")
  for q1 in tqdm(s1):
    #Now i have the q1
    mean_vectors=np.zeros(shape=(300,))
    for word in q1.split():
      #This is the calculation of the word vectors
        try:
          vector=word_vec[str(word)]
        except KeyError:
          vector=np.zeros(shape=(300,))

        #This is the calculation of idf values
        try:
          idf=word2idf[str(word)]
        except KeyError:
          idf=0
        #print(idf,word)
        mean_vectors+=vector*idf
    mean_vector=mean_vectors/len(q1.split())
    sent_vectors.append(mean_vector)
    
  return np.array(sent_vectors)
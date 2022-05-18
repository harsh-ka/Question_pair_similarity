from similarity_functions import jaccard_index,cwc
from preprocessing_text_new import preprocessing
from fuzzywuzzy import fuzz
from sentence_vectors import sentence_vector
import pickle
from scipy.spatial.distance import cosine
import numpy  as np
from gensim.models import KeyedVectors

#word_vec=dict({})

def process_query(q1,q2,word2idf,word_vec):
  features=[]
  #Preprocessing the text such as making it lower case,removing stop words decontraction and html tags
  q1=preprocessing(q1)
  q2=preprocessing(q2)
  #Processing letter_count_q1 and_letter_count q2
  letter_count_q1=len(q1)
  letter_count_q2=len(q2)
  total_letter_count=len(q1)+len(q2)
  features.extend([letter_count_q1,letter_count_q2,total_letter_count])
  #Processing_word count_q1 and q2 and diff_words and total_words
  word_count_q1=len(q1.strip().split())
  word_count_q2=len(q2.strip().split())
  diff_in_words=abs(word_count_q1-word_count_q2)
  total_words=word_count_q1+word_count_q2
  features.extend([word_count_q1,word_count_q2,diff_in_words,total_words])
  #jaccard similarity
  features.extend([jaccard_index(q1,q2)])
  #Now we will create the fuzzy features and scaling them between 0 to 1
  fuzzy_ratio=fuzz.ratio(q1,q2)/100
  fuzzy_partial_ratio=fuzz.partial_ratio(q1,q2)/100
  fuzzy_token_sort_ratio=fuzz.token_sort_ratio(q1,q2)/100
  fuzzy_token_set_ratio=fuzz.token_set_ratio(q1,q2)/100
  features.extend([fuzzy_ratio,fuzzy_partial_ratio,fuzzy_token_sort_ratio,fuzzy_token_set_ratio])

  #Cwc index will be user
  cwc_min=cwc(q1,q2,lambda x,y:min(x,y))
  cwc_max=cwc(q1,q2,lambda x,y:max(x,y))
  features.extend([cwc_min,cwc_max])
  # we will be creating the feature vector for q1 and then q1
  q1_mean_vector=list(sentence_vector([q1],word2idf,word_vec)[0])
  features.extend(q1_mean_vector)
  #Calculation for word 2
  q2_mean_vector=list(sentence_vector([q2],word2idf,word_vec)[0])
  features.extend(q2_mean_vector)
  #pushing cosine similarity value
  val=cosine(q1_mean_vector, q2_mean_vector)
  #if val is nan means one or both vectors are zero than setting val values to be 1
  if np.isnan(val):
    val=1
  features.extend([1-val])
  return features


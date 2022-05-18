import argparse
import pickle
from helper import process_query
import xgboost as xgb


parser=argparse.ArgumentParser(description="This is terminal to test question pair similairty")
required=parser.add_argument_group("Require arguments")
required.add_argument('-s1','--sent1',type=str,help='First sentence to compare',required=True)
required.add_argument('-s2','--sent2',type=str,help='Second sentence to compare',required=True)

args=parser.parse_args()
sentence1=args.sent1
sentence2=args.sent2


#Please provide the weight path where you have saved the weights
model_path='xgbmodel.pkl'
word2idf_path='word2idf.pkl'
gensim_google_weight_300_dims='GoogleNews-vectors-negative300.bin.gz'
model=pickle.load(open(model_path,'rb'))

#loading the word2idf and word2vec
word2idf=pickle.load(open(word2idf_path,'rb'))
word_vec=KeyedVectors.load_word2vec_format(gensim_google_weight_300_dims,binary=True).wv

#processing vector
feature_vector=process_query(sentence1,sentence2,word2idf,word_vec)

query_data=xgb.DMatrix(feature_vector,feature_names=model.feature_names)
val=model.predict(query_data)

if val[0]>.5:
    print(f"Yes {sentence1} and {sentence2} are similar")
else:
    print("No they are not similar")

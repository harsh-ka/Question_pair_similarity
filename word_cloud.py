import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
def Show_Cloud(text_corpus,name=None,mask=None,**kwargs):
  text=''
  if type(text_corpus)==list or type(text_corpus)==set:
    text=' '.join([sent for sent in text_corpus])
  if type(text_corpus)==str:
    text=text_corpus
  #Text courpus is ready 
  #Instantiating the word Courpus
  wc=WordCloud(width=400,height=200,mask=None if mask is None else mask,background_color="white",collocations=False).generate(text)
  plt.figure(figsize=(8,8),dpi=150)
  plt.imshow(wc,interpolation='bilinear')
  plt.axis('off')
  if name is not None:
    wc.to_file(str(name))
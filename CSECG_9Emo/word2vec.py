#coding:utf-8
import logging
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import KeyedVectors,LineSentence
import os
from data import load_data,tokenize
from data import build_vocab as get_vocab
MAX_WORDS_IN_BATCH=1000


logger=logging.Logger(name="word2vec",level=logging.INFO)
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s')
logging.root.setLevel(level=logging.INFO)

model_path="./datasets/temp/word2vec.bin"
text_path="./datasets/temp/post_response.txt"
emb_dim=1000
def build_text():
    iw,vocab,_=get_vocab()
    with open(text_path,'w',encoding='utf-8') as f:
        data=load_data()
        for post,resp in data:
            post=" ".join(tokenize(post[0],vocab=vocab))
            resp=" ".join(tokenize(resp[0],vocab=vocab))
            f.write(post+"\t"+resp+"\n")
        
            
                
def train_word2vec():
    '''训练词项向量
    '''
    if not os.path.exists(text_path):
        build_text()
    model=Word2Vec(sentences=LineSentence(text_path),size=emb_dim,window=5,min_count=5,iter=10)
    model.wv.save_word2vec_format(model_path,binary=True)
    return model
    
def get_embedding():
    emb_path="datasets/temp/embedding.np"
    if os.path.exists(emb_path):
        return np.load(open(emb_path,'rb'))
    else:
        model=KeyedVectors.load_word2vec_format(model_path,binary=True)
        iw,vocab,_=get_vocab()
        size=len(list(vocab.keys()))
        emb=np.zeros(shape=[size,emb_dim])
        for word,index in vocab.items():
            if index in [0,1] or word not in model.vocab:
                continue
            emb[index]=model[word]
        np.save(open(emb_path,"wb"),emb)
        return emb
    
if __name__=="__main__":
    train_word2vec()
    model=KeyedVectors.load_word2vec_format(model_path,binary=True)
    emb=get_embedding()
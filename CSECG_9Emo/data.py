# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 16:49:47 2017

@author: SWING
"""
import re
import jieba
import json
import nltk
import pickle
import numpy as np
import os

post_len = 20
resp_len = 20
response_len = 20

######################预处理数据##################################
'''
read lines from file 
return [list of lines]
'''
def read_data(filename):
   with open(filename,'r',encoding='utf-8') as f:
      data = f.read().split('\n')
   return data

'''load json file'''
def load_data(data_path="datasets/temp/filter_sen.json"):
   with open(data_path,encoding='utf-8') as f:
      data=json.load(f)
   return data

def cut_withPOS(sentence):
    words =jieba.cut(sentence)
    wordlist=[word for word in words]
    return wordlist

'''
过滤标点并重新分词
'''
def filter_punc(data):
   fp_list = []
   
   for post,resp in data:
      pr_pair = []
      #print('post',post)
      post_with_E=[]
      resp_with_E=[]
      post,postE=post
      resp,respE=resp
      p = re.compile("[\s+\.\!\/_,$%^*、(+\"\')]+|[+——()?=【】“”!~@#￥%&*（）]+")
      post = p.sub(" ",post).strip().split() #去掉特殊标点
      resp = p.sub(" ",resp).strip().split()
#      post = p.sub(" ",post) #去掉特殊标点
#      resp = p.sub(" ",resp)
      post = cut_withPOS("".join(post)) #重新分词
      resp = cut_withPOS("".join(resp))
      post = " ".join(post)
      resp = " ".join(resp)
      post_with_E.append(post)
      post_with_E.append(postE)
#      print(post_with_E)
      resp_with_E.append(resp)
      resp_with_E.append(respE)
#      print(resp_with_E)
      pr_pair.append(post_with_E)
      pr_pair.append(resp_with_E)
      fp_list.append(pr_pair)
#      print(fp_list)
   return fp_list    
     
'''
filter cantonese from the corpus
过滤包含两个粤语词以上的句子
'''
def filter_cantonese(cantonese,data):
   sen_list = []

   for post,resp in data:
      pr_pair = []
      post_with_E=[]
      resp_with_E=[]
      post,postE=post
      resp,respE=resp
      #分词后看是否包含粤语
      post_words=post.split()
      resp_words=resp.split()
      print(post_words)
      post_temp = [word for word in post_words if word in cantonese]
      resp_temp = [word for word in resp_words if word in cantonese]
      if len(set(post_temp)) >0 or len(set(resp_temp)) >0: #如果包含粤语就去掉继续循环
         print('post containing canronese:')
         print(post,post_temp)
         print('resp containing cantonese')
         print(resp,resp_temp)
         continue
      post_with_E.append(post)
      post_with_E.append(postE)
#      print(post_with_E)
      resp_with_E.append(resp)
      resp_with_E.append(respE)
      pr_pair.append(post_with_E)
      pr_pair.append(resp_with_E)
      sen_list.append(pr_pair)
   return sen_list
   

   
'''细分不在vocab中的词
最大前向匹配
'''
def get_prefix_in_vocab(word,vocab):
   l = len(word) #得到词语长度
   for i in range(0,l+1):
      prefix = word[0:l-i]
      if prefix in vocab:
         return prefix,word[l-i:l]
      if i==l:
         return word,''
         
'''分词，对不在词表的词进行最大前向匹配'''
def tokenize(sentence,vocab=None):
   words = sentence.strip().split() # ['哈哈', '…', '生日', '快乐', '。']
   if vocab is None:#这一步的意义是什么
      return words
   else:
        temp=[]
        for word in words: #循环遍历每个词语
            if word in vocab:
                temp.append(word)
            else:
                while word!='':
                    prefix,word=get_prefix_in_vocab(word,vocab)
                    temp.append(prefix)
        return temp

        
 # 对原始数据进行重新分词，并得到40000词的词典
def  recut_data(data):
   def word_gen():
      for post,resp in data:
            post,postE=post
            resp,respE=resp
            post=post.strip().split() 
            resp=resp.strip().split()
#            post = cut_withPOS("".join(post))
#            resp = cut_withPOS("".join(resp))
#            print('post',post)
#            print('resp',resp)
            for word in post+resp:
                yield word               
   gen=word_gen()
   freq=nltk.FreqDist(gen) #词频字典
   print("len:",len(freq))
   
   # get vocabulary of 'vocab_size' most used words
   vocab = freq.most_common(40000)
   # index2word
   index2word = [x[0] for x in vocab]
   return index2word

#########################构建词表#############################

'''统计词频'''
def get_word_freq():
   def word_gen():
      data=load_data("datasets/temp/filter_sen.json")
      for post,resp in data:
         post,postE=post
         resp,respE=resp
         post=post.strip().split() 
         resp=resp.strip().split()
         #print('post',post)
         for word in post+resp:
            #print(word)
            yield word
   gen=word_gen()
   freq=nltk.FreqDist(gen) #词频字典
   print("len:",len(freq))
   return freq
   
'''建立词典v=12819
1.初始化词表的大小为：10223，即选取的频率>=80的词语 (11491 f>=80)
2.细分不在初始词表中的词语为字符，将不在词表中的字符加入词表
3.词表最终最小的词语频率：80
'''
def build_vocab(init_vocab_size=11491,min_freq=80):
   dump_path = 'datasets/temp/vocab.pkl'
   if os.path.exists(dump_path): 
      return pickle.load(open(dump_path, 'rb'))
   else:
      freq = get_word_freq()
      all_words = freq.most_common() 
      vocab = dict(all_words[:init_vocab_size]) #得到频率大于等于80的词语
      #将低频词进行细分
      for w,i in all_words[init_vocab_size:]:
         chars = list(w)
         for c in chars:
            if c in vocab: #如果字符在词表中
               vocab[c] += i
            else:
               vocab[c] = i
   vocab = sorted(vocab.items(),key=lambda x:x[1],reverse=True)
   freq = vocab
   idx2w = ['<END>','<UNK>']+[w for w,i in vocab if i>min_freq]         
   w2idx = dict([(w,i) for i,w in enumerate(idx2w)])    
   pickle.dump([idx2w,w2idx,freq],open(dump_path,'wb'))        
   return idx2w,w2idx,freq   
def get_vocab():
    return build_vocab()
############################处理训练数据########################

'''统计长度'''
def get_length(data):
   # 定义两个字典来存储大于对应长度的句子个数
   post_lens = dict([(i,0) for i in range(5,41,2)])
   resp_lens = dict([(i,0) for i in range(5,41,2)])
   
   idx2w,w2idx,freq = build_vocab()
   j=0
   for post,resp in data:
      j+=1
      post,postE=post
      resp,respE=resp
      post = tokenize(post,w2idx)
      resp = tokenize(resp,w2idx)
      print(j)
      for i in range(5,41,2):
         if len(post) > i:
            post_lens[i]+=1
         if len(resp) > i:
            resp_lens[i]+=1
   print("post_lens:")
   print(list(sorted(post_lens.items(),key=lambda x:x[0])))
   print("response lengths:")
   print(list(sorted(resp_lens.items(),key=lambda x:x[0])))

'''不够长的补0.多了的截去'''
def padding(sentences,max_len=20,value=0):
   padded = []
   for seq in sentences:
      l = len(seq)
      seq = list(seq)+[value for i in range(max_len-l)]#不够
      seq = seq[:max_len]#够了
      padded.append(seq)
   return np.array(padded) #tensorflow默认输入为np数组

'''训练数据
    1. 数据预处理
    去掉post或response长度太短的样本,
    2. padding 数据
'''
def get_train_data():
   dump_path="datasets/temp/train_data.pkl"
#   dump_path="C:/Users/SWING/Desktop/train_data.pkl"
   if os.path.exists(dump_path):
      return pickle.load(open(dump_path,'rb'))
   else:
      idx2w,w2idx,freq = build_vocab()
      data = load_data("datasets/temp/filter_sen.json") #是重新分词之后的呀
      meta={}
      posts=[]
      postEs=[]
      postLens=[]
      resps=[]
      respEs=[]
      respLens=[]
      for post,resp in data:
         post,postE=post
         resp,respE=resp
         post = tokenize(post,w2idx)
         resp = tokenize(resp,w2idx)
         #得到每个词对应的索引，如果w2id中不存在这个词，返回索引1（对应的是词UNK）
         post = [w2idx.get(w,1) for w in post]
         resp = [w2idx.get(w,1) for w in resp]
         if len(post)<3 or len(resp)<5:
            continue #删除len(post)<3或者len(resp)<5的post-response对
         posts.append(post)
         postEs.append(postE)
         postLens.append(min(len(post)+1,post_len))
         resps.append(resp)
         respEs.append(respE)
         respLens.append(min(len(resp)+1,resp_len))
      meta['posts']=padding(posts,max_len=post_len)
      meta['postE']=np.array(postEs)
      meta['postLen']=np.array(postLens)
      meta['resps']=padding(resps,max_len=resp_len)
      meta['respE']=np.array(respEs)
      meta['respLen']=np.array(respLens)
      pickle.dump(meta,open(dump_path,'wb'))
      return meta

#########################计算词语的tf-idf值#############################
'''
tf_ij=n_ij/sum(n_kj) ----词语的词频
n_ij表示特征词t_i在文本d_j中出现的次数  分母表示文本d_j中所有特征词的个数
'''
def tf():  
   meta=get_train_data()
   keys=['posts','postE','postLen','resps','respE','respLen']
   values = [meta[k] for k in keys]
   length = len(values[0])
   post_tf=[]
   resp_tf=[]
   for i in range(0,length):
      post_tf_list=[]
      resp_tf_list=[]
      
      post_words_dict={}
      resp_words_dict={}

      #得到这一个post,resp中每个词的出现次数
      postLen=values[2][i]-1 #该post的真实长度（之前有加1）
      for j in range(0,postLen): #遍历每个索引 索引作为key,在该post中出现的次数作为value
         if list(values[0][i])[j] not in post_words_dict:
            post_words_dict[list(values[0][i])[j]] = 1
         else:
            post_words_dict[list(values[0][i])[j]] +=1
                            
      respLen=values[5][i]-1
      for j in range(0,respLen): #遍历每个索引 索引作为key,在该post中出现的次数作为value
         if list(values[3][i])[j] not in resp_words_dict:
            resp_words_dict[list(values[3][i])[j]] = 1
         else:
            resp_words_dict[list(values[3][i])[j]] +=1

      #计算这一个post和resp词语的tf值
      for j in range(0,postLen):
          post_word_tf=post_words_dict[list(values[0][i])[j]]/postLen#word的tf值
          post_tf_list.append(post_word_tf)
      for j in range(0,respLen):
          resp_word_tf=resp_words_dict[list(values[3][i])[j]]/respLen#word的tf值
          resp_tf_list.append(resp_word_tf)
      post_tf.append(post_tf_list)
      resp_tf.append(resp_tf_list)
      
   print(len(post_tf))
   print(len(resp_tf))
   return post_tf,resp_tf

'''
idf_i=log(|D|/(1+|D_ti|))----词语的逆文档频率
|D|表示语料中的文本数  |D_ti|表示文本中包含特征词t_i的文本数量'''
def idf():
   id2w,w2id,freq_=build_vocab() #得到词典
   meta=get_train_data()
   keys=['posts','postE','postLen','resps','respE','respLen']
   values = [meta[k] for k in keys]
   length = len(values[0])#posts的长度
   l =length*2 #语料库中的文本数包括post+response
   f_dict={}#value是包含该词的文本个数
   idf_dict={}#key-word,value-idf值
   for i in range(0,length):
      for word_index in set(list(values[0][i])):#不重复的每个词
#         if word_index==10130:
#            print(word_index)
         if word_index not in f_dict:
            f_dict[word_index]=1
         else:
            f_dict[word_index]+=1
      for word_index in set(list(values[3][i])):
#         if word_index==10130:
#            print(word_index)
         if word_index not in f_dict:
            f_dict[word_index]=1
         else:
            f_dict[word_index]+=1
   print('l',l)
   
   for w,i in w2id.items():
      print('w',w)
      if i in f_dict:
          f=f_dict[i]
      else:
          f=0
      word_idf=np.log(l/(1+f)) #通过索引得到包含该词的文本数量
#      idf_dict[w]=word_idf
      idf_dict[i]=word_idf #key-索引 value-idf值
   print(len(idf_dict))
   
   return idf_dict

'''计算tf-idf值'''
def cal_tfidf(post_tf,resp_tf,idf_dict):
  
   meta=get_train_data()
   keys=['posts','postE','postLen','resps','respE','respLen']
   values = [meta[k] for k in keys]
   length = len(values[0])
   post_tfidf=[]
   resp_tfidf=[]
   for i in range(0,length):
      post_temp=[]#这一行的
      postLen=values[2][i]-1
      for j in range(0,postLen): #真实长度postLen
         tfidf = post_tf[i][j]*idf_dict[list(values[0][i])[j]]
         post_temp.append(tfidf) #这一个句子的tf-idf值
      post_tfidf.append(post_temp) #所有post的tf-idf值
      
      resp_temp=[]
      respLen=values[5][i]-1
      for j in range(0,respLen):
         tfidf = resp_tf[i][j]*idf_dict[list(values[3][i])[j]]
         resp_temp.append(tfidf) #这一个句子的tf-idf值
      resp_tfidf.append(resp_temp) #所有resp的tf-idf值
   print(len(post_tfidf))
   print(len(resp_tfidf))
         
   return post_tfidf,resp_tfidf

def get_tfidf():
    dump_path="datasets/temp/tfidf.pkl"
    if os.path.exists(dump_path):
        return pickle.load(open(dump_path,'rb'))
    else:
        ##################计算tf-idf#########################
        post_tf,resp_tf=tf()
        idf_dict=idf()
        
        post_tfidf,resp_tfidf=cal_tfidf(post_tf,resp_tf,idf_dict)
        tfidf={}
        tfidf["post_tfidf"]=post_tfidf
        tfidf["resp_tfidf"]=resp_tfidf
#        tfidf["idf_dict"]=idf_dict
#        tfidf['post_tf']=post_tf
#        tfidf['resp_tf']=resp_tf
        pickle.dump(tfidf,open("tfidf.pkl",'wb'))   
        return tfidf

#####################切分训练数据为训练集 验证集和测试集###################
def split_data(ratio = [0.98, 0.015, 0.005]):
   meta = get_train_data()
   tfidf=get_tfidf()
   meta['post_tfidf']=padding(tfidf['post_tfidf'],max_len=post_len,value=0.5)  #句子末尾的<END>权重不能为0
   meta['resp_tfidf']=padding(tfidf['resp_tfidf'],max_len=resp_len,value=0.5)
   keys=['posts','postE','postLen','resps','respE','respLen','post_tfidf','resp_tfidf']
   
   values = [meta[k] for k in keys]
   data_len = len(values[0])
   ids = list(range(data_len))
   lens=[int(data_len*item) for item in ratio]
  
   train_ids,val_ids,test_ids=ids[:lens[0]],ids[lens[0]:lens[0]+lens[1]],ids[-lens[-1]:]
   
   train_data=dict(zip(keys,[v[train_ids] for v in values]))
   val_data=dict(zip(keys,[v[val_ids] for v in values]))
   test_data=dict(zip(keys,[v[test_ids] for v in values]))
   
   return train_data,val_data,test_data
       
if __name__=="__main__":       
   ################得到预处理后的句子################
   #load data
#   train_data=load_data('temp/train_data.json')

   # 过滤标点
#   fp_list = filter_punc(train_data)
   
   #得到重新分词后的词典
#   vocab = recut_data(fp_list)
  
   #得到不在40000个词中的粤语词表
#   with open('temp/metadata.pkl', 'rb') as f:
#        metadata = pickle.load(f)
#   vocab = metadata['idx2w']
#   cantonese = read_data('temp/广州话.txt')
#   c = [word for word in cantonese if word not in vocab] 
#
#   #   #过滤粤语
#   sentences = filter_cantonese(c, fp_list)
#   with open("filter_sen.json",'w',encoding='utf-8') as json_file:
#     json.dump(sentences,json_file,ensure_ascii=False)

   ##################构建词表并得到训练数据###################
   # 过滤后的句子     
#   data = load_data("retemp/filter_sen.json") #是重新分词之后的呀
#   id2w,w2id,freq_=build_vocab() #得到词典
#   get_length(data) #统计句子长度
#   meta=get_train_data(data) #得到训练数据
#   train_data,val_data,test_data=split_data(data) #分割数据集


    meta=get_train_data()
    tf_idf=get_tfidf()
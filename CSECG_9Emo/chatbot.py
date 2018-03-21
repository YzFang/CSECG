#coding:utf-8
from data import split_data,tokenize,get_vocab,padding
from seq2seq_attention_9emo import Seq2SeqAttentionMinDis,Seq2SeqAttentionMaxDis,Seq2SeqAttentionEmoContent
from seq2seq_attention_9emo import Seq2SeqAttentionHappy,Seq2SeqAttentionSad,Seq2SeqAttentionAnger,Seq2SeqAttentionDisgust
from seq2seq_attention_9emo import Seq2SeqAttentionLike,Seq2SeqAttentionSurprise,Seq2SeqAttentionFear
from pprint import pprint
import pandas as pd
import numpy as np
import jieba

##分词器 thulac
import thulac   
thu1 = thulac.thulac(seg_only=True)  #只进行分词，不进行词性标注
text = thu1.cut("我爱北京天安门", text=True)  #进行一句话分词
print(text)

#分词器 pynlpir
import pynlpir
pynlpir.open()

s = '欢迎科研人员、技术工程师、企事业单位与个人参与NLPIR平台的建设工作。'
pynlpir.segment(s,pos_tagging=False)


_,vocab,_=get_vocab()
def sent2ids(sent):
#    sent=" ".join(jieba.lcut(sent))
#    sent=" ".join(thu1.cut(sent,text=True))
    sent=" ".join(pynlpir.segment(sent,pos_tagging=False))
    words=tokenize(sent,vocab)
    print(words)
    ids=[vocab.get(w,1) for w in words]
    print(ids)
    l=len(ids)
    return padding([ids],max_len=20),np.array([l]),np.array([20])

if __name__=="__main__":
    keys=['post','postLen','resp','respLen']
    model=Seq2SeqAttentionHappy(isTrain=True)
    model.restore_last_session()
    
    emo="EmoContent"
    
    sent="你好~~"
    while sent.strip()!="END":
        sent=input("Human:").strip()
        if len(sent)==0:
            print("please say something...")
            continue
        ids,l,dec_len=sent2ids(sent)
        results=model.predict([ids,l,ids,dec_len])
        print("%s Bot:%s"%(emo,results[0]))
    
    

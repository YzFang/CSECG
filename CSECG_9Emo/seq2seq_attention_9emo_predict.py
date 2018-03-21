#coding:utf-8
from data import split_data,padding,resp_len
from seq2seq_attention import Seq2SeqAttention
from seq2seq_attention_9emo import Seq2SeqAttentionMinDis,Seq2SeqAttentionMaxDis,Seq2SeqAttentionEmoContent
from seq2seq_attention_9emo import Seq2SeqAttentionHappy,Seq2SeqAttentionSad,Seq2SeqAttentionAnger,Seq2SeqAttentionDisgust
from seq2seq_attention_9emo import Seq2SeqAttentionLike#,Seq2SeqAttentionSurprise,Seq2SeqAttentionFear
from pprint import pprint
import pandas as pd
import os
import numpy as np

from emo_cls.classification import Classification

# Emotion Classifier
emo_clas = Classification(sequence_length=20,
                      num_classes=6,
                      l2_reg_lambda=0.1)
emo_clas.restore_last_session(base_path="./emo_cls")

def predict(model,test_datas):
    results,ids=model.predict(test_datas,return_ids=True)
    ids=padding(ids,max_len=resp_len)
    emo_results=model.predict_emotion(test_datas,emo_cls=emo_clas)
    emo_results=np.array(emo_results)
    emo_ids=emo_results.argmax(-1)
    emo_dict={}
    for i in emo_ids:
        if i not in emo_dict:
            emo_dict[i]=1
        else:
            emo_dict[i]+=1
    print(emo_dict)
    
    posts=model.id2text(test_datas[0])
    
    pairs=list(zip(posts,results))
    
#    pprint(pairs[10:50])
    
    df=pd.DataFrame()
    df['posts']=posts
    df['responses']=results
    
    #写到excel中
    name=model.__class__.__name__
    df.to_excel(os.path.join("results",name+"_results.xlsx"),sheet_name="results",index=False,header=True)
    np.save(os.path.join("results",name+"_ids.npy"),ids)
    return results,ids,posts,emo_ids
    
def evaluate(model,test_datas):
    results,loss=model.evaluate(test_datas)
    #计算困惑度
    perplex=np.exp(loss)
    perplex2=np.exp2(loss)
    
    print("Model: %s,loss:%s,困惑度:%s,困惑度exp2:%s"%(model.__class__.__name__,loss,perplex,perplex2))
    return loss,perplex

if __name__=="__main__":
    train_datas,val_datas,test_datas=split_data()
    keys=['posts','postLen','resps','respLen','resp_tfidf']
    train_datas=[train_datas[k] for k in keys]
    val_datas=[val_datas[k] for k in keys]
#    val_datas=[d[20:50] for d in val_datas]
    test_datas=[test_datas[k] for k in keys]
    gen_models=[Seq2SeqAttentionMinDis,Seq2SeqAttentionMaxDis,Seq2SeqAttentionEmoContent,
                Seq2SeqAttentionHappy,Seq2SeqAttentionSad,Seq2SeqAttentionAnger,Seq2SeqAttentionDisgust,
                Seq2SeqAttentionLike,Seq2SeqAttention]
    post_emo=emo_clas.predicat_emotion(test_datas[0],test_datas[1])
    post_emo_ids=post_emo.argmax(-1)
    post_emo_dict={}
    for i in post_emo_ids:
        if i not in post_emo_dict:
            post_emo_dict[i]=1
        else:
            post_emo_dict[i]+=1
    print(post_emo_dict)
    '''
    for Model in gen_models[5:6]:            
        model=Model(isTrain=True)
        model.restore_last_session()
        results,ids,posts,emo_ids=predict(model,test_datas)
        loss,p=evaluate(model,test_datas)
    '''
        
        
#    test_resp=test_datas[2]
#    resps=model.id2text(test_resp)
#    df=pd.DataFrame()
#    df['posts']=posts
#    df['responses']=resps
#    df.to_excel(os.path.join("results","Gold_results.xlsx"),sheet_name="results",index=False,header=True)
#    np.save(os.path.join("results","Gold_ids.npy"),test_resp)
        
    

#coding:utf-8
import tensorflow as tf
from data import get_train_data,get_vocab,split_data,response_len,post_len,padding
import random
import os

from pprint import pprint
import numpy as np
import time

id2w,w2id,freq=get_vocab()

from emo_cls.classification import Classification
from seq2seq_attention_9emo import Seq2SeqAttentionMinDis,Seq2SeqAttentionMaxDis,Seq2SeqAttentionEmoContent
from seq2seq_attention_9emo import Seq2SeqAttentionHappy,Seq2SeqAttentionSad,Seq2SeqAttentionAnger,Seq2SeqAttentionDisgust
from seq2seq_attention_9emo import Seq2SeqAttentionLike#,Seq2SeqAttentionSurprise,Seq2SeqAttentionFear

train_datas,val_datas,test_datas=split_data()

keys=['posts','postLen','resps','respLen','resp_tfidf']
train_datas=[train_datas[k] for k in keys]
val_datas=[val_datas[k] for k in keys]
print("train num:%s"%len(train_datas[0]))

seq_len=20
batch_size=128
D_step=5
G_step=1
is_debug=True

# Emotion Classifier
emo_clas = Classification(sequence_length=20,
                      num_classes=6,
                      l2_reg_lambda=0.1)
emo_clas.restore_last_session(base_path="./emo_cls")


def random_batch_generator(datas,batch_size=batch_size):
    '''随机生成一个batch的数据
    '''
    num=len(datas[0])
    all_ids=list(range(num))
    while True:
        ids=random.sample(all_ids,batch_size)
        batch=[d[ids] for d in datas]
        yield batch
def train_SeqEmotion(gen_model,iter_num=10):
    log_path=os.path.join("summary/",gen_model.__class__.__name__)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    writer = tf.summary.FileWriter(log_path, graph=gen_model.sess.graph)
    batch_gen=random_batch_generator(train_datas,batch_size=batch_size)
    global_step=0
    total_data_num=len(train_datas[0])
    start_time=time.time()
    global is_debug
    is_debug=is_debug
    for i in range(iter_num):
        if is_debug and global_step>=50:
            end_time=time.time()
            long=end_time-start_time
            print("global step:%s, total time:%s seconds"%(global_step,long))
        while True:
            global_step+=1
            if global_step%500==0:
                val_texts,val_loss=gen_model.evaluate([v[20:50] for v in val_datas])
                print("valid loss:%s"%val_loss)
                pprint(val_texts)
            batch_data=next(batch_gen)
            #emotion MLE
            emo_loss,penalty,sum_,emo_rewards=gen_model.train_step_emotion(batch_data,emotion_classifier=emo_clas,is_debug=is_debug)
            mle_loss=emo_loss
            loss=emo_loss
            #####################  Save model #################################
            if global_step%500==0:
                writer.add_summary(sum_, global_step)
                print("iter_num:%s,global step: %s, teacher forece loss: %s, mle loss: %s, penalty: %s"%(i,global_step,loss,mle_loss,penalty))
            if global_step%1000==0:
                gen_model.save_weights(global_step=global_step,saver=gen_model.saver)
            
            if (global_step*batch_size)>=(total_data_num*(i+1)):
                break
        
        gen_model.save_weights(global_step=global_step)
if __name__=="__main__":
    gen_models=[Seq2SeqAttentionMinDis,Seq2SeqAttentionMaxDis,
                Seq2SeqAttentionEmoContent,Seq2SeqAttentionHappy,
                Seq2SeqAttentionSad,Seq2SeqAttentionAnger,
                Seq2SeqAttentionDisgust,Seq2SeqAttentionLike]
#    ckpt_path="weights/Seq2SeqAttention"            
    for Model in gen_models[0:8]:
#    for Model in gen_models[::-1][0:4]:
#    for Model in gen_models[5:]:
#    for Model in gen_models[-1:]:
        gen_model=Model()
#        gen_model.restore_last_session(ckpt_path)
        gen_model.restore_last_session()
        train_SeqEmotion(gen_model,iter_num=5)
    

        
    







    

#cocing:utf-8
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
from word2vec import get_embedding
from data import get_train_data,get_vocab,split_data,response_len,post_len,padding
import random
import os

from tensorflow.python.layers.core import Dense
import tensorflow.contrib.seq2seq as seq2seq
from pprint import pprint
import numpy as np
from myutils.virsulize import visualize_vectors
from discriminator import Discriminator

embedding=get_embedding().astype("float32")
embedding=embedding/np.max(embedding,keepdims=False)
id2w,w2id,freq=get_vocab()

mc_sample_num=5
emb_dim=1000
num_units=1000
layer_num=4
vocab_size=len(id2w)
post_len=post_len
resp_len=response_len
beam_size=100
initializer=tf.random_uniform_initializer(minval=-0.1,maxval=0.1)
batch_size=128

class Seq2SeqAttention(object):
    def __init__(self,isTrain=True):
        self.ckpt_path=os.path.join("weights/",self.__class__.__name__)
        self.log_path=os.path.join("summary/",self.__class__.__name__)
        if not os.path.exists(self.ckpt_path):
            os.mkdir(self.ckpt_path)
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        tf.reset_default_graph()
#        with tf.Graph().as_default():
        
        self.enc_in=tf.placeholder(dtype=tf.int32,shape=[None,post_len])
        self.enc_len=tf.placeholder(dtype=tf.int32,shape=[None,])
        self.emotion_in=tf.placeholder(dtype=tf.int32,shape=[None,])
        
        self.dec_len=tf.placeholder(dtype=tf.int32,shape=[None,])
        self.labels=tf.placeholder(dtype=tf.int32,shape=[None,resp_len])
        self.dec_in=tf.transpose(tf.stack([tf.zeros(shape=[tf.shape(self.labels)[0],],dtype=tf.int32)]+tf.unstack(tf.transpose(self.labels))[:-1]))
        self.keep_prob=tf.placeholder(dtype=tf.float32,name="keep_prob")
        self.lr=tf.placeholder(dtype=tf.float32,name="lr")
        self.rewards=tf.placeholder(dtype=tf.float32,shape=[None,None],name="rewards")
        
        self.isTrain=isTrain
        
        self.build_model()
        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.weights=tf.trainable_variables()
        self.saver=tf.train.Saver(max_to_keep=5,var_list=self.weights)
        
    def get_basicLSTMCell(self):
        basic_cell=tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units,activation=tf.nn.tanh),output_keep_prob=self.keep_prob)
        return basic_cell
    def build_model(self):
        with tf.variable_scope("embedding"):
            embedding_enc=tf.get_variable(name="embedding",initializer=embedding,dtype=tf.float32)
            tf.summary.histogram(name='embedding_enc',values=embedding_enc)
            embeded_post=tf.nn.embedding_lookup(embedding_enc,self.enc_in)
            embedding_dec=embedding_enc
            embeded_resp=tf.nn.embedding_lookup(embedding_dec,self.dec_in)
        
        with tf.variable_scope("encoder",initializer=initializer):
            fw_cells=[self.get_basicLSTMCell() for i in range(layer_num)]
            bw_cells=[self.get_basicLSTMCell() for i in range(layer_num)]
            fw_cell=tf.nn.rnn_cell.MultiRNNCell(fw_cells)
            bw_cell=tf.nn.rnn_cell.MultiRNNCell(bw_cells)
            
            (output_fw, output_bw),(output_state_fw, output_state_bw)=tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,
                                                                        inputs=embeded_post,
                                                                        sequence_length=self.enc_len-1,
                                                                        dtype=tf.float32,
                                                                        parallel_iterations=1)
            encoder_hs=[output_state_fw[i].h+output_state_bw[i].h for i in range(layer_num)]
            encoder_cs=[output_state_fw[i].c+output_state_bw[i].c for i in range(layer_num)]
            encoder_final_states=tuple([tf.nn.rnn_cell.LSTMStateTuple(encoder_cs[i],encoder_hs[i]) for i in range(layer_num)])
            encoder_outputs=output_fw+output_bw
            self.hidden_vec=encoder_hs[-1]
        with tf.variable_scope("decoder",initializer=initializer):  
            max_decoder_len=tf.reduce_max(self.dec_len) #当前batch中最大的response长度
            # 定义输出层
            fc = Dense(vocab_size,name="output_projection")
            if self.isTrain:
                att_cell,initial_state=self.getDecoderCell(encoder_outputs,encoder_final_states)
            else:
                att_cell,initial_state=self.getBeamSearchDecoderCell(encoder_outputs,encoder_final_states)
            ##############  decoder during training ################################
            #decode得到的句子最大长度与self.dec_len的最大值相同
            train_helper=tf.contrib.seq2seq.TrainingHelper(inputs=embeded_resp,sequence_length=self.dec_len) 
            # 定义BasicDecoder
            train_decoder = tf.contrib.seq2seq.BasicDecoder(cell=att_cell,
                                                      helper=train_helper,
                                                      initial_state=initial_state,
                                                      output_layer=fc)
            #生成的句子长度seq_len<=maxinum_iterations && seq_len<=max(self.dec_len)
            [self.decode_outputs, self.train_decode_ids], self.decode_states, self.decode_seq_lengths = \
                                 seq2seq.dynamic_decode(decoder=train_decoder,
                                                        output_time_major=False,
                                                        impute_finished=True,
                                                        maximum_iterations=max_decoder_len)  #decode最大迭代次数，
            ############ decoder during testing ##################################
            test_helper=tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding_dec,
                                                                start_tokens=tf.ones(shape=[tf.shape(self.enc_in)[0],],dtype=tf.int32)*0,
                                                                end_token=0)
    
            
            # 定义BasicDecoder
            test_decoder = tf.contrib.seq2seq.BasicDecoder(cell=att_cell,
                                                      helper=test_helper,
                                                      initial_state=initial_state,
                                                      output_layer=fc)
            [self.test_decode_outputs, self.test_decode_ids], self.test_decode_states, self.test_decode_seq_lengths = \
                                 seq2seq.dynamic_decode(decoder=test_decoder,
                                                        output_time_major=False,
                                                        impute_finished=True,
                                                        maximum_iterations=resp_len)
        
            ######################### 计算loss  ################################   
            labels=tf.slice(self.labels,begin=[0,0],size=[-1,max_decoder_len]) #切片，是label的个数与dynamic decoder得到的输出一致                  
            #监督学习MLE损失
            self.weighted_loss,self.mle_loss=self.getMLELoss(labels,self.decode_outputs,max_decoder_len)
            tf.summary.scalar("weigted_loss",self.weighted_loss)
            tf.summary.scalar("mle_loss",self.mle_loss)
            #惩罚repeat
            self.penalty=self.getRepeatPenalty(self.decode_outputs,max_decoder_len)
            tf.summary.scalar("penalty",self.penalty)
            self.weighted_loss+=self.penalty
            self.loss=self.weighted_loss
            self.loss+=self.penalty
            
            #计算梯度
#            opt=tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            opt=tf.train.AdamOptimizer(learning_rate=0.001)
            mle_grads_and_vars=opt.compute_gradients(self.weighted_loss)
            mle_capped_grad=[(tf.clip_by_value(g,-5,5),v) for g,v in mle_grads_and_vars if g is not None]
            self.mle_opt=opt.apply_gradients(mle_capped_grad)
            
            grads_and_vars=opt.compute_gradients(self.loss)
            capped_grad=[(tf.clip_by_value(g,-5,5),v) for g,v in grads_and_vars if g is not None]
            self.opt=opt.apply_gradients(capped_grad)
#            for g,v in capped_grad
#                tf.summary.histogram(v.name.replace(":","_"),g)
#            tf.summary.scalar(name="lr",tensor=self.lr)
            self.summary_op=tf.summary.merge_all()
        
    def getDecoderCell(self,encoder_outputs,encoder_final_states):
        basic_cells=[self.get_basicLSTMCell() for i in range(layer_num)]
        basic_cell=tf.nn.rnn_cell.MultiRNNCell(basic_cells)
        initial_state=encoder_final_states
        #attention 
        attention_mechanism=seq2seq.BahdanauAttention(num_units=num_units,
                                                      memory=encoder_outputs,
                                                      memory_sequence_length=self.enc_len
                                                      )
        att_cell=seq2seq.AttentionWrapper(basic_cell,
                                         attention_mechanism=attention_mechanism,
                                         attention_layer_size=num_units,
                                         alignment_history=False,
                                         cell_input_fn=None,
                                         initial_cell_state=initial_state)
        
        initial_state=att_cell.zero_state(batch_size=tf.shape(self.enc_in)[0],dtype=tf.float32)   
#            att_state.clone(cell_state=encoder_final_state)

        return att_cell,initial_state
    def getBeamSearchDecoderCell(self,encoder_outputs,encoder_final_states):
        basic_cells=[self.get_basicLSTMCell() for i in range(layer_num)]
        basic_cell=tf.nn.rnn_cell.MultiRNNCell(basic_cells)
        tiled_encoder_outputs=seq2seq.tile_batch(encoder_outputs,multiplier=beam_size)
        tiled_encoder_final_states=[seq2seq.tile_batch(state,multiplier=beam_size) for state in encoder_final_states]
        tiled_sequence_length=seq2seq.tile_batch(self.enc_len,multiplier=beam_size)
        initial_state=tuple(tiled_encoder_final_states)
        #attention 
        attention_mechanism=seq2seq.BahdanauAttention(num_units=num_units,
                                                      memory=tiled_encoder_outputs,
                                                      memory_sequence_length=tiled_sequence_length
                                                      )
        att_cell=seq2seq.AttentionWrapper(basic_cell,
                                         attention_mechanism=attention_mechanism,
                                         attention_layer_size=num_units,
                                         alignment_history=False,
                                         cell_input_fn=None,
                                         initial_cell_state=initial_state)
        
        initial_state=att_cell.zero_state(batch_size=tf.shape(self.enc_in)[0]*beam_size,dtype=tf.float32)   
#            att_state.clone(cell_state=encoder_final_state)

        return att_cell,initial_state
                    
        
    def getRepeatPenalty(self,decoder_outputs,max_decoder_len):
        '''#惩罚项，是生成的单词具有差异性，防止生成重复的单词
        '''
        probs=tf.nn.softmax(self.decode_outputs,dim=-1)
        penalty=tf.matmul(probs,probs,transpose_b=True)-tf.expand_dims(tf.diag(tf.ones(shape=[max_decoder_len,])),axis=0)
        penalty_mask=tf.sequence_mask(lengths=self.dec_len,maxlen=max_decoder_len,dtype=tf.float32,name='penalty_mask')
        penalty=tf.reduce_sum(tf.square(penalty)*tf.expand_dims(penalty_mask,axis=2))/(tf.reduce_sum(penalty_mask)+1e-12)
        return penalty
        
    def getMLELoss(self,target,decoder_outputs,max_decoder_len):
        '''MLE损失
        '''
        masks=tf.sequence_mask(lengths=self.dec_len,maxlen=max_decoder_len,dtype=tf.float32,name='masks')
        weights=tf.slice(self.rewards,begin=[0,0],size=[-1,max_decoder_len])
        weights=tf.exp(weights)
        weights=tf.clip_by_value(weights,clip_value_min=1,clip_value_max=tf.reduce_min(weights)*3)
#        weights=weights/tf.reduce_max(weights)
        weights=weights*masks
        
        #计算每个单词的加权 loss
        weighted_loss=seq2seq.sequence_loss(logits=decoder_outputs,
                                    targets=target,
                                    weights=weights,
                                    average_across_timesteps=True,
                                    average_across_batch=True)
        #mle loss
        loss=seq2seq.sequence_loss(logits=decoder_outputs,
                                    targets=target,
                                    weights=masks,
                                    average_across_timesteps=True,
                                    average_across_batch=True)
        return weighted_loss,loss
        
          
    def batch_generator(self,datas,batch_size=128,shuffle=False):
        '''生成一个batch的数据
        '''
        num=len(datas[0])
        if shuffle:
            ids=random.sample(list(range(num)),num)
            datas=[d[ids] for d in datas]
            lens=[len(d) for d in datas]
            print("lens",lens)
            
        for i in range((num+batch_size-1)//batch_size):
            s=i*batch_size
            e=(i+1)*batch_size
            batch=[d[s:e] for d in datas]
            yield batch
                
    def get_feed_dict(self,placeholders,datas):
        return dict(zip(placeholders,datas))
    def id2text(self,ids,sep="",max_repeat_num=2):
        '''id转换成文本
        '''
        texts=[]
        for idx in ids:
            words=[]
            for i in idx:
                if i==0:
                    break
                word=id2w[i]
                
                if len(words)>=max_repeat_num: #跳过重复单词
                    repeat=list(set(words[-max_repeat_num:]))
                    if len(repeat)==1 and word in repeat:
                        continue
                words.append(word)
            texts.append(sep.join(words))
        return texts
    def pre_train(self,train_datas,val_datas,iter_num=20,lr=0.5,batch_size=128,keep_prob=0.8,shuffle=True,is_debug=False):
        print(len(train_datas[0]))
        global_step=0
        writer = tf.summary.FileWriter(self.log_path, graph=self.sess.graph)
#        saver=tf.train.Saver(max_to_keep=5,var_list=self.weights)
        for i in range(iter_num):
            if i>5:
                lr=lr*0.5
            gen=self.batch_generator(train_datas,batch_size,shuffle=True)
            for d in gen:
                global_step+=1
                placeholders=[self.enc_in,self.enc_len,self.labels,self.dec_len,self.rewards] #tf_idf 作为reward
                feed_dict=self.get_feed_dict(placeholders,d)
                feed_dict[self.lr]=lr
                feed_dict[self.keep_prob]=keep_prob
                _,loss,penalty,sum_=self.sess.run([self.mle_opt,self.mle_loss,self.penalty,self.summary_op],feed_dict=feed_dict)
                if is_debug:
                    print("iter_num:%s,global step: %s, batch loss: %s, penalty: %s"%(i,global_step,loss,penalty))
                if global_step%500==0:
                    writer.add_summary(sum_, global_step)
                if global_step%1000==0:
                    self.save_weights(global_step=global_step,saver=self.saver)
                    print("iter_num:%s,global step: %s, batch loss: %s, penalty: %s"%(i,global_step,loss,penalty))
                    
                    val_texts,val_loss=self.evaluate([v[20:50] for v in val_datas])
                    print("valid loss:%s"%val_loss)
                    pprint(val_texts)
            if i%5==0:
                self.save_weights(global_step=global_step,saver=self.saver)
        self.save_weights(global_step=global_step)
        
        
    def train_step_emotion(self,batch_data,emotion_classifier,is_debug=False,lr=1.0,keep_prob=0.8):
        '''Emotion MLE
        '''
        placeholders=[self.enc_in,self.enc_len,self.labels,self.dec_len,self.rewards]
        enc_in=batch_data[0]
        enc_lens=batch_data[1]
        dec_in=batch_data[2]
        dec_lens=batch_data[3]
        tf_idf=batch_data[4]
        emo_rewards=self.get_emotion_rewards(enc_in,dec_in,emotion_classifier)
        rewards=np.sqrt(np.expand_dims(emo_rewards,1)*tf_idf)
        
        feed_dict=self.get_feed_dict(placeholders,[enc_in,enc_lens,dec_in,dec_lens,rewards])
        feed_dict[self.lr]=lr
        feed_dict[self.keep_prob]=1.0
        
        #1. 先使用生成器G进行预测
        _,loss,penalty,sum_=self.sess.run([self.opt,self.loss,self.penalty,self.summary_op],
                                                 feed_dict)
        
        
        return loss,penalty,sum_,rewards
    def get_emotion_rewards(self,enc_in,dec_in,emotion_classifier):
        rewards=emotion_classifier.get_rewards(enc_in,dec_in)
        return rewards
        
    def get_rewards(self,batch_data,discriminator,emotion_classifier=None):
        scores=discriminator.get_rewards(batch_data)
        enc_in=batch_data[0]
        dec_in=batch_data[2]
        if emotion_classifier is not None:
            emo_socres=emotion_classifier.get_rewards(enc_in,dec_in)
            scores=np.sqrt(scores*emo_socres) 
        return scores
    def generate_step(self,batch_data):
        '''生成一个batch的数据
        '''
        placeholders=[self.enc_in,self.enc_len]
        feed_dict=self.get_feed_dict(placeholders,batch_data[:2])
        feed_dict[self.keep_prob]=1.0
        r=random.randint(0,10)
        if r>=5:
            output_ids,lens=self.sess.run([self.test_decode_ids,self.test_decode_seq_lengths],feed_dict)
        else:
            output_ids,lens=self.sess.run([self.ss_test_decode_ids,self.ss_test_decode_seq_lengths],feed_dict)
        return output_ids,lens
    def predict(self,test_datas,batch_size=128,keep_prob=1.0,return_ids=False):
        gen=self.batch_generator(test_datas,batch_size,shuffle=False)
        results=[]
        ids=[]
        for data in gen:
            placeholders=[self.enc_in,self.enc_len]
            feed_dict=self.get_feed_dict(placeholders,data[:2])
            feed_dict[self.keep_prob]=keep_prob
            output_ids=self.sess.run(self.test_decode_ids,feed_dict)
            ids.extend(output_ids)
            results.extend(self.id2text(output_ids))
        if return_ids:
            return results,ids
        else:
            return results
    def predict_emotion(self,test_datas,emo_cls,batch_size=128,keep_prob=1.0):
        '''预测情感'''
        gen=self.batch_generator(test_datas,batch_size,shuffle=False)
        results=[]
        for data in gen:
            placeholders=[self.enc_in,self.enc_len]
            feed_dict=self.get_feed_dict(placeholders,data[:2])
            feed_dict[self.keep_prob]=keep_prob
            output_ids,output_lens=self.sess.run([self.test_decode_ids,self.test_decode_seq_lengths],feed_dict)
            output_ids=padding(output_ids,max_len=resp_len)
            emo_dist=emo_cls.predicat_emotion(output_ids,output_lens)
            results.extend(emo_dist)
        return results
    def evaluate(self,val_datas,batch_size=128):
        gen=self.batch_generator(val_datas,batch_size,shuffle=False)
        results=[]
        losses=[]
        for data in gen:
            placeholders=[self.enc_in,self.enc_len,self.labels,self.dec_len,self.rewards]
            feed_dict=self.get_feed_dict(placeholders,data)
            feed_dict[self.keep_prob]=1.0
            output_ids,loss=self.sess.run([self.test_decode_ids,self.mle_loss],feed_dict)
            results.extend(self.id2text(output_ids))
            losses.append(loss)
        return results,np.mean(losses)
    def sent2vec(self,sents,lens,batch_size=128):
        gen=self.batch_generator([sents,lens],batch_size,shuffle=False)
        vectors=[]
        for data in gen:
            placeholders=[self.enc_in,self.enc_len]
            feed_dict=self.get_feed_dict(placeholders,data) 
            feed_dict[self.keep_prob]=1.0
            vec=self.sess.run(self.hidden_vec,feed_dict)
            vectors.extend(vec)
        return np.array(vectors)
    def save_weights(self,global_step=None,saver=None):
        if saver is None:
            saver=tf.train.Saver(max_to_keep=5,var_list=self.weights)
        saver.save(self.sess,save_path=os.path.join(self.ckpt_path,"weights.ckpt"),global_step=global_step)
        
    def restore_last_session(self,ckpt_path=None):
        saver = tf.train.Saver(var_list=self.weights)
        if ckpt_path is None:
            ckpt_path=self.ckpt_path
        # get checkpoint state
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        # restore session
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("fail to restore..., ckpt:%s"%ckpt)
    
if __name__=="__main__":
    train_datas,val_datas,test_datas=split_data()
    keys=['posts','postLen','resps','respLen','resp_tfidf']
    train_datas=[train_datas[k] for k in keys]
    val_datas=[val_datas[k] for k in keys]
    test_datas=[test_datas[k] for k in keys]
    
    model=Seq2SeqAttention()
    model.restore_last_session()
    model.pre_train(train_datas=train_datas,val_datas=val_datas,is_debug=True)
    
            

#coding:utf-8
from seq2seq_attention import Seq2SeqAttention
import numpy as np

emotions=['none','happiness','sadness','anger','disgust','like']#,'surprise','fear']
emo2id=dict([(e,i) for i,e in enumerate(emotions)])
emo_vecs=np.diag([1]*6)
none_emo=emo_vecs[0]
happy_emo=emo_vecs[1]
sad_emo=emo_vecs[2]
anger_emo=emo_vecs[3]
disgust_emo=emo_vecs[4]
like_emo=emo_vecs[5]
#surprise_emo=emo_vecs[6]
#fear_emo=emo_vecs[7]

class Seq2SeqAttentionMinDis(Seq2SeqAttention):
    def __init__(self,isTrain=True):
        super(Seq2SeqAttentionMinDis,self).__init__(isTrain)
        
    def get_emotion_rewards(self,enc_in,enc_lens,dec_in,dec_lens,emotion_classifier):
        rewards=emotion_classifier.get_rewards(enc_in,enc_lens,dec_in,dec_lens) #
        return rewards
        
class Seq2SeqAttentionMaxDis(Seq2SeqAttention):
    def __init__(self,isTrain=True):
        super(Seq2SeqAttentionMaxDis,self).__init__(isTrain)
        
    def get_emotion_rewards(self,enc_in,enc_lens,dec_in,dec_lens,emotion_classifier):
        rewards=1-emotion_classifier.get_rewards(enc_in,enc_lens,dec_in,dec_lens)+0.01  #最大化情感差异
        return rewards
class Seq2SeqAttentionEmoContent(Seq2SeqAttention):
    '''情感内容'''
    def __init__(self,isTrain=True):
        super(Seq2SeqAttentionEmoContent,self).__init__(isTrain)
        self.emotion=none_emo
        
    def get_emotion_rewards(self,enc_in,enc_lens,dec_in,dec_lens,emotion_classifier):
        rewards=1-emotion_classifier.get_emo_rewards(dec_in,dec_lens,emotion=self.emotion)+0.01  #内容富有情感
        return rewards
        
class Seq2SeqAttentionEmo(Seq2SeqAttention):
    '''具有指定的情感'''
    def __init__(self,isTrain=True):
        super(Seq2SeqAttentionEmo,self).__init__(isTrain)
        self.emotion=none_emo
    def get_emotion_rewards(self,enc_in,enc_lens,dec_in,dec_lens,emotion_classifier):
        rewards=emotion_classifier.get_emo_rewards(dec_in,dec_lens,emotion=self.emotion)  
        return rewards
class Seq2SeqAttentionHappy(Seq2SeqAttentionEmo):
    def __init__(self,isTrain=True):
        super(Seq2SeqAttentionHappy,self).__init__(isTrain)
        self.emotion=happy_emo
class Seq2SeqAttentionSad(Seq2SeqAttentionEmo):
    def __init__(self,isTrain=True):
        super(Seq2SeqAttentionSad,self).__init__(isTrain)
        self.emotion=sad_emo
class Seq2SeqAttentionAnger(Seq2SeqAttentionEmo):
    def __init__(self,isTrain=True):
        super(Seq2SeqAttentionAnger,self).__init__(isTrain)
        self.emotion=anger_emo
class Seq2SeqAttentionDisgust(Seq2SeqAttentionEmo):
    def __init__(self,isTrain=True):
        super(Seq2SeqAttentionDisgust,self).__init__(isTrain)
        self.emotion=disgust_emo
class Seq2SeqAttentionLike(Seq2SeqAttentionEmo):
    def __init__(self,isTrain=True):
        super(Seq2SeqAttentionLike,self).__init__(isTrain)
        self.emotion=like_emo
#class Seq2SeqAttentionSurprise(Seq2SeqAttentionEmo):
#    def __init__(self,isTrain=True):
#        super(Seq2SeqAttentionSurprise,self).__init__(isTrain)
#        self.emotion=surprise_emo
#class Seq2SeqAttentionFear(Seq2SeqAttentionEmo):
#    def __init__(self,isTrain=True):
#        super(Seq2SeqAttentionFear,self).__init__(isTrain)
#        self.emotion=fear_emo
        

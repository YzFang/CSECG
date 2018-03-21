import tensorflow as tf
import numpy as np
import random
import sys

# An alternative to tf.nn.rnn_cell._linear function, which has been removed in Tensorfow 1.0.1
# The highway1 layer is borrowed from https://github.com/mkroutikov/tf-lstm-char-cnn
def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term

def highway1(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway1'):
    """Highway1 Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway1_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway1_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output
    
data_path='data_v3_6emo'
    
embedding=np.load(open(data_path+"/embedding.np",'rb')).astype('float32')
vocab_size=embedding.shape[0]
embedding_size=embedding.shape[1]
num_units=256
layer_num=1

class Classification(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        tf.reset_default_graph()
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y") # one-hot 二分类 [0,1]正例 [1,0]负例
        self.lens=tf.placeholder(tf.int32,[None,],name='lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.session = tf.Session()

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        
        with tf.variable_scope('classification'):

            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
#                self.W = tf.Variable(
#                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
#                    name="W")
                self.W=tf.get_variable("embedding",initializer=embedding,trainable=True)
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x) #shape(batch_size,seq_len,embedding_size)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1) #shape(batch_size,seq_len,embedding_size,1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for filter_size, num_filter in zip(filter_sizes, num_filters): #zip将其打包为元组的列表，元素个数与最短的列表一致
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filter] #[filter_height, filter_width, in_channels, out_channels]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                    conv = tf.nn.conv2d(  # return： A `Tensor`. Has the same type as `input`.
                        input=self.embedded_chars_expanded, #[batch, in_height, in_width, in_channels]
                        filter=W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")  # [batch, in_height, in_width, in_channels]
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool( #return： 返回一个Tensor，类型不变，shape仍然是[batch, height, width, in_channels]
                        value=h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1], #池化窗口的大小  一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
                        strides=[1, 1, 1, 1], #窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled) #长度为 ?len(num_filter) shape(len(num_filter),batch, in_height, in_width, in_channels)
            
            # Combine all the pooled features
            num_filters_total = sum(num_filters)
            self.h_pool = tf.concat(pooled_outputs, 3) #[batch, height, width, num_filters_total] 在第三维上拼起来
#            print('pool:',pooled_outputs, tf.shape(self.h_pool))

            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # Add highway1
            with tf.name_scope("highway1"):
                self.h_highway1 = highway1(input_=self.h_pool_flat, size=self.h_pool_flat.get_shape()[1], num_layers=1, bias=0) #

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_highway1, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
#                print('output:  score ',self.scores)
                self.ypred_for_auc = tf.nn.softmax(self.scores)
                self.predictions = tf.argmax(self.scores, 1, name="predictions")
                
                correct_predition = tf.equal(self.predictions,tf.argmax(self.input_y,1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predition,tf.float32))

            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        self.params = [param for param in tf.trainable_variables() if 'classification' in param.name]
        d_optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2) #指定组合梯度的方法2
        self.train_op = d_optimizer.apply_gradients(grads_and_vars)
        
        self.session.run(tf.global_variables_initializer())


    def gen_batch(self,x_data,y_data,x_lens,batch_size,shuffle = False):
        num = len(y_data)
        if shuffle:
            ids = random.sample(list(range(num)),num)
            x_data = x_data[ids]
            y_data = y_data[ids]
            x_lens=x_lens[ids]

        num_batch = (len(x_data) + batch_size -1) // batch_size
        for i in range(num_batch):
            start = i * batch_size
            end = start + batch_size
            yield x_data[start:end], y_data[start:end],x_lens[start:end]

    def train(self,x_input,y_input,x_lens,x_eval,y_eval,lens_eval,testX,testY,testLens,batch_size,epochs):
        
        sys.stdout.write('***********Training start***********\n')
        for epoch_i in range(epochs):
            gen_batch = self.gen_batch(x_input,y_input,x_lens,batch_size,shuffle=True)
            batch_i = 0
            for batch_x,batch_y,batch_lens in gen_batch:
                _,loss,train_acc = self.session.run([self.train_op,self.loss,self.accuracy],feed_dict={self.input_x:batch_x,
                                                                      self.input_y:batch_y,
                                                                      self.lens:batch_lens,
                                                                      self.dropout_keep_prob:0.5})
                batch_i += 1
                if batch_i and batch_i % 50 == 0:
                    val_loss, val_acc = self.evaluation(x_eval,y_eval,lens_eval,batch_size)
#                    print('epoch{:>2} -- batch{:>4} -- train loss{:.4f} --train_acc{:.4f} -- val_loss{:.4f} -- val_acc{:.4f}'.format(epoch_i,batch_i,loss,train_acc,val_loss, val_acc))
                    test_loss, test_acc = self.evaluation(testX,testY,testLens,batch_size)
                    #print('epoch{:>2} -- batch{:>4} -- train loss{:.4f} --train_acc{:.4f} -- test_loss{:.4f} -- test_acc{:.4f}'.format(epoch_i,batch_i,loss,train_acc,val_loss, val_acc))
                    print('epoch{:>2} -- batch{:>4} -- train loss{:.4f} --train_acc{:.4f}--epoch{:>2} -- val_loss{:.4f} -- val_acc{:.4f} - test_acc{:.4f}:'.format(epoch_i,batch_i,loss,train_acc,epoch_i,val_loss,val_acc,test_acc))
            if epoch_i and epoch_i % 2 == 0:
                self.save_weights(global_step=epoch_i)
                print('save weights -- epoch{:>2} -- val_loss{:.4f} -- val_acc{:.4f} -- test_loss{:.4f} -- test_acc{:.4f}:'.format(epoch_i,val_loss,val_acc,test_loss,test_acc))


    def evaluation(self,x_eval,y_eval,lens_eval,batch_size):
        gen_batch = self.gen_batch(x_eval,y_eval,lens_eval,batch_size=batch_size,shuffle=True)
        loss=[]
        acc =[]
        for batch_x,batch_y,batch_lens in gen_batch:
            eval_loss, val_acc = self.session.run([self.loss, self.accuracy],feed_dict={self.input_x:batch_x,
                                                              self.input_y: batch_y,
                                                              self.lens:batch_lens,
                                                              self.dropout_keep_prob: 1
                                                              })
            loss.append(eval_loss)
            acc.append(val_acc)
        return np.mean(loss), np.mean(acc)



    def predict(self,x_test,y_test,batch_size=128):
        gen_batch = self.gen_batch(x_test,y_test,batch_size=batch_size,shuffle=True)
        y_predicts=[]
        loss_predicts=[]
        accs=[]
        for batch_x,batch_y in gen_batch:
            y_predict,loss_predict,acc = self.session.run([self.ypred_for_auc,self.loss,self.accuracy],feed_dict={self.input_x:batch_x,
                                                                                                            self.input_y:batch_y,
                                                                                                            self.dropout_keep_prob:1.0})
            y_predicts.extend(y_predict)
            loss_predicts.append(loss_predict)
            accs.append(acc)
        return y_predicts,np.mean(loss_predicts),np.mean(accs)


    def save_weights(self,global_step=None):
        saver = tf.train.Saver()
        saver.save(self.session,save_path='weights/model.ckpt',global_step=global_step)


    def restore_last_session(self):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir='weights/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.session,save_path=ckpt.model_checkpoint_path)
        else:
            print('**************fail to restore ..., ckpt:%s*******************'%ckpt)
            
def split_data(X,Y,Lens):
    '''重新划分数据'''
    import os
    import pickle
    dump_path=data_path+"/data.pkl"
    if os.path.exists(dump_path):
        return pickle.load(open(dump_path,'rb'))
    else:
        
        Targets=Y.argmax(1)
        target2Ids=[(t,i) for i,t in enumerate(Targets)]
        groups={}
        for t2i in target2Ids:
            t,i=t2i
            if t not in groups:
                groups[t]=[i]
            else:
                groups[t].append(i)
        train_ids=[]
        valid_ids=[]
        test_ids=[]
        for t,ids in groups.items():
            num=len(ids)
            print(num)
            ids=random.sample(ids,num)
            train_num=int(num*0.8)
            print("train_num:",train_num)
            test_num=int(train_num+num*0.1)
            train_ids.extend(ids[:train_num])
            test_ids.extend(ids[train_num:test_num])
            valid_ids.extend(ids[test_num:])
        print(len(train_ids))    
        trainX=X[train_ids]
        trainY=Y[train_ids]
        trainLens=Lens[train_ids]
        print("trainX",trainX.shape)
        
        testX=X[test_ids]
        testY=Y[test_ids]
        testLens=Lens[test_ids]
        
        validX=X[valid_ids]
        validY=Y[valid_ids]
        validLens=Lens[valid_ids]
        
        
        pickle.dump([trainX,trainY,trainLens,validX,validY,validLens,testX,testY,testLens],open(dump_path,'wb'))
    
        return trainX,trainY,trainLens,validX,validY,validLens,testX,testY,testLens
if __name__ == '__main__':

    # filter_sizes = [1,2,3,4,5,6,7,8,9,10,15,20]
    # num_filers = [100,200,200,200,200,100,100,100,100,100,160,160]

    filter_sizes = [1,2,3,4,5]
    num_filers = [300,300,300,300,300]

    clas = Classification(sequence_length=20,
                          num_classes=6,
                          vocab_size=12819,
                          embedding_size=256,
                          filter_sizes=filter_sizes,
                          num_filters=num_filers,
                          l2_reg_lambda=0.1)

    #x = np.random.randint(1000,size=(1000,30))
    #y = np.random.random(size=(1000,8))
    import pickle

    train_data = pickle.load(open(data_path+'/emotion_train_data.pkl', 'rb')) # dict {'posts':..., 'postLen':...}
    valid_data = pickle.load(open(data_path+'/emotion_valid_data.pkl', 'rb'))
    test_data= pickle.load(open(data_path+'/emotion_test_data.pkl', 'rb'))
    
    train_label = np.load(data_path+'/f_trainEmo.npy')[:,:-2]
    valid_label = np.load(data_path+'/f_validEmo.npy')[:,:-2]
    test_label = np.load(data_path+'/f_testEmo.npy')[:,:-2]

    trainX = train_data['posts']
    trainLens=train_data['postLen']
    trainY = train_label

    
    validX = valid_data['posts']
    validLens=valid_data['postLen']
    validY = valid_label
    
    testX = test_data['posts']
    testLens=test_data['postLen']
    testY = test_label
    
    X=np.concatenate([trainX,validX,testX],axis=0)
    Y=np.concatenate([trainY,validY,testY],axis=0)
    Lens=np.concatenate([trainLens,validLens,testLens],axis=0)
    
    trainX,trainY,trainLens,testX,testY,testLens,validX,validY,validLens=split_data(X,Y,Lens)

   
    print('trainX:',np.shape(trainX))
    print('trainX:',np.shape(trainY))
    
    print('validX:',np.shape(validX))
    print('validY:',np.shape(validY))
    
    print('testX:',np.shape(testX))
    print('testY:',np.shape(testY))
    
    
    #统计每个类别的数量
    def statistic_labels(labels):
        d={}
        for i in labels:
            if i not in d:
                d[i]=1
            else:
                d[i]+=1
        return d
    trainLabels=statistic_labels(trainY.argmax(1))
    validLabels=statistic_labels(validY.argmax(1))
    testLabels=statistic_labels(testY.argmax(1))
    print("trainLabels",trainLabels)
    print("validLabels",validLabels)
    print("testLabels",testLabels)
    
    trainSents=[]
    for sent in trainX:
        sent=' '.join(sent.astype(str))
        trainSents.append(sent)
    count = 0
    for sent in testX:
        sent=' '.join(sent.astype(str))
        if sent in trainSents:
            print(sent)
            count += 1
        
    print('test重复的数量：',count)
    
    
    
#    clas.restore_last_session()
    clas.train(trainX,trainY,trainLens,validX,validY,validLens,testX,testY,testLens,batch_size=128,epochs=10)

    
    
#    clas.restore_last_session()

    print('********************predict************************')
    y_predict,loss_predict,acc = clas.predict(testX,testY,testLens)


#    print(y_predict)
    print('prediction loss:',loss_predict)
    print('prediction accuracy:',acc)


import os
import numpy as np
import tensorflow as tf
import sys
import time
import random
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.utils import class_weight
import csv

class text_cnn_multitask(object):

    def __init__(self,embedding_matrix,num_classes,max_words,num_tasks,
                 num_filters=300,dropout_keep=0.5,class_weights=None):

        self.embedding_size = embedding_matrix.shape[1]
        self.embeddings = embedding_matrix.astype(np.float32)
        self.mw = max_words
        self.dropout_keep = dropout_keep
        self.dropout = tf.placeholder(tf.float32)
        self.num_tasks = num_tasks

        #inputs and outputs for each task are stored in a list
        self.predictions = []
        self.labels = []
        self.losses = []
        self.loss = 0
        self.cs_vars = []

        #doc input and mask
        self.doc_inputs = tf.placeholder(tf.int32, shape=[None,max_words])
        word_embeds = tf.gather(tf.get_variable('embeddings',initializer=self.embeddings,dtype=tf.float32),self.doc_inputs)

        #each task has its own path
        for i in range(num_tasks):

            reuse = None
            if i > 0:
                reuse = True

            with tf.variable_scope('cnn',reuse=reuse):
                #word convolutions
                conv3_outs = []
                conv4_outs = []
                conv5_outs = []
                for j in range(num_tasks):
                    conv3_W = tf.get_variable('conv3_W_%i' % j,[3,self.embedding_size,num_filters],tf.float32,tf.orthogonal_initializer())
                    conv3_b = tf.get_variable('conv3_b_%i' % j,[num_filters],tf.float32,tf.zeros_initializer())
                    conv3_out = tf.nn.relu(tf.nn.conv1d(word_embeds,conv3_W,1,'SAME') + conv3_b)
                    conv3_outs.append(tf.expand_dims(conv3_out,0))

                    conv4_W = tf.get_variable('conv4_W_%i' % j,[4,self.embedding_size,num_filters],tf.float32,tf.orthogonal_initializer())
                    conv4_b = tf.get_variable('conv4_b_%i' % j,[num_filters],tf.float32,tf.zeros_initializer())
                    conv4_out = tf.nn.relu(tf.nn.conv1d(word_embeds,conv4_W,1,'SAME') + conv4_b)
                    conv4_outs.append(tf.expand_dims(conv4_out,0))

                    conv5_W = tf.get_variable('conv5_W_%i' % j,[5,self.embedding_size,num_filters],tf.float32,tf.orthogonal_initializer())
                    conv5_b = tf.get_variable('conv5_b_%i' % j,[num_filters],tf.float32,tf.zeros_initializer())
                    conv5_out = tf.nn.relu(tf.nn.conv1d(word_embeds,conv5_W,1,'SAME') + conv5_b)
                    conv5_outs.append(tf.expand_dims(conv5_out,0))

            with tf.variable_scope('cross_stitch'):

                #convolution cross stitch operations
                init_weights = np.ones((num_tasks,1,1,1)).astype(np.float32) * (0.1/(num_tasks-1))
                init_weights[i] = 0.9

                conv3_cs = tf.get_variable('conv3_cs_%i' % i,dtype=tf.float32,initializer=init_weights)
                conv3_cs_out = tf.reduce_sum(conv3_cs * tf.concat(conv3_outs,0),0)

                conv4_cs = tf.get_variable('conv4_cs_%i' % i,dtype=tf.float32,initializer=init_weights)
                conv4_cs_out = tf.reduce_sum(conv4_cs * tf.concat(conv4_outs,0),0)

                conv5_cs = tf.get_variable('conv5_cs_%i' % i,dtype=tf.float32,initializer=init_weights)
                conv5_cs_out = tf.reduce_sum(conv5_cs * tf.concat(conv5_outs,0),0)

                self.cs_vars.append(conv3_cs)
                self.cs_vars.append(conv4_cs)
                self.cs_vars.append(conv5_cs)

            #pooling operations
            pool3 = tf.reduce_max(conv3_cs_out,1)
            pool4 = tf.reduce_max(conv4_cs_out,1)
            pool5 = tf.reduce_max(conv5_cs_out,1)

            #concatenate
            doc_embed = tf.concat([pool3,pool4,pool5],1)
            doc_embed = tf.nn.dropout(doc_embed,self.dropout)

            #classification functions
            with tf.variable_scope('cnn'):
                W_softmax = tf.get_variable('W_softmax_%i' % i,(num_filters*3,num_classes[i]),tf.float32,tf.orthogonal_initializer())
                b_softmax = tf.get_variable('b_softmax_%i' % i,(num_classes[i]),tf.float32,tf.zeros_initializer())

            output = tf.matmul(doc_embed,W_softmax) + b_softmax
            self.predictions.append(tf.nn.softmax(output))

            #loss, accuracy, and training functions
            self.labels.append(tf.placeholder(tf.float32,shape=[None,num_classes[i]]))

            #scale cross entropy by number of classes
            if class_weights != None:
                weight_per_sample = tf.squeeze(tf.matmul(self.labels[i],tf.expand_dims(tf.constant(class_weights[i],dtype=tf.float32),1)))
                losses = tf.losses.softmax_cross_entropy(self.labels[i],output,weight_per_sample)
            else:
                losses = tf.losses.softmax_cross_entropy(self.labels[i],output,num_classes[i]**.5)
                #losses = tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=self.labels[i]) * (num_classes[i]**.5)
                #losses = tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=self.labels[i])
            self.losses.append(tf.reduce_mean(losses))
            self.loss += self.losses[i]

        self.loss /= self.num_tasks
        self.optimizer = tf.train.AdamOptimizer(0.001,0.9,0.99).minimize(self.loss)
        #self.optimizer = tf.train.AdadeltaOptimizer(1.0).minimize(self.loss)
        '''
        #different learning rates for cross stitch weights and cnn weights
        self.optimizer_cs = tf.train.AdamOptimizer(0.1,0.9,0.999).minimize(self.loss,var_list=self.var_cs)
        self.optimizer_cnn = tf.train.AdamOptimizer(0.001,0.9,0.999).minimize(self.loss,var_list=self.var_cnn)
        '''
        #init op
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=None)
        self.sess = tf.Session()
        self.sess.run(self.init_op)

    def train(self,data,labels,batch_size=16,epochs=30,validation_data=None,
              savebest=False,filepath=None):

        #print size of validation set
        if validation_data != None:
            for i in range(self.num_tasks):
                validation_size = len(validation_data[i][0])
                print('task %i: training on %i documents, validating on %i documents' \
                      % (i+1, len(data), validation_size))

        #track best model for saving
        prevbest = [0 for j in range(self.num_tasks)]

        #train
        for ep in range(epochs):

            #shuffle data
            ys = zip(*labels)
            shuffle = list(zip(data,ys))
            random.shuffle(shuffle)
            data,ys = zip(*shuffle)
            labels = zip(*ys)
            data = np.array(data)
            labels = list(labels)

            #track training accuracy per task and train time
            start = time.time()
            y_true = [[] for j in range(self.num_tasks)]
            y_pred = [[] for j in range(self.num_tasks)]

            #get batches
            for i in range(0,len(data),batch_size):

                if i+batch_size < len(data):
                    stop = i+batch_size
                else:
                    stop = len(data)

                feed_dict = {self.dropout:self.dropout_keep}
                retvals = []

                for j in range(self.num_tasks):
                    feed_dict[self.doc_inputs] = data[i:stop]
                    feed_dict[self.labels[j]] = labels[j][i:stop]
                    retvals.append(self.predictions[j])
                    y_true[j].append(np.argmax(labels[j][i:stop],1))
                retvals.append(self.loss)
                retvals.append(self.optimizer)

                retvals = self.sess.run(retvals,feed_dict=feed_dict)
                cost = retvals[-2]

                for j in range(self.num_tasks):
                    pred = retvals[j]
                    y_pred[j].append(np.argmax(pred,1))

                sys.stdout.write("epoch %i - sample %i of %i, losses: %f      \r"\
                                 % (ep+1,i+1,len(data),cost))
                sys.stdout.flush()

            print()
            #print("training time: %.2f" % (time.time()-start))

            for j in range(self.num_tasks):
                y_t = np.concatenate(y_true[j],0)
                y_p = np.concatenate(y_pred[j],0)
                acc = np.sum(y_t==y_p)/float(len(y_t))
                print("epoch %i task %i training accuracy: %.4f%%" % (ep+1,j+1,acc*100))

            #validate
            if validation_data != None:
                for j in range(self.num_tasks):
                    micro,macro = self.score(validation_data[j][0],validation_data[j][1],j)
                    print("epoch %i task %i validation accuracy: %.4f, %.4f" % (ep+1,j+1,micro,macro))

                    #save if performance better than previous best
                    if savebest and micro >= prevbest[j]:
                        prevbest[j] = micro
                        self.save(filepath + '_task%i.ckpt' % j)

            self.cross_stitch_weights()

            #save progress
            self.save(filepath + '.ckpt')

    def score(self,data,labels,task,batch_size=16):

        y_true = []
        y_pred = []
        for start in range(0,len(data),batch_size):

            #get batch index
            if start+batch_size < len(data):
                stop = start+batch_size
            else:
                stop = len(data)

            feed_dict = {self.doc_inputs:data[start:stop],self.dropout:1.0}
            pred = self.sess.run(self.predictions[task],feed_dict=feed_dict)
            y_true.append(np.argmax(labels[start:stop],1))
            y_pred.append(np.argmax(pred,1))

        y_true = np.concatenate(y_true,0)
        y_pred = np.concatenate(y_pred,0)
        micro = f1_score(y_true,y_pred,average='micro')
        macro = f1_score(y_true,y_pred,average='macro')
        return micro,macro

    def cross_stitch_weights(self):

        vals = self.sess.run(self.cs_vars)
        for i in range(self.num_tasks):
            c3 = np.squeeze(vals[i*3])
            c4 = np.squeeze(vals[i*3+1])
            c5 = np.squeeze(vals[i*3+2])
            print('task %i conv3 cross stitch vals: ' % (i+1), c3)
            print('task %i conv4 cross stitch vals: ' % (i+1), c4)
            print('task %i conv5 cross stitch vals: ' % (i+1), c5)

    def predict(self,data,task,batch_size=16):

        y_pred = []
        y_prob = []
        for start in range(0,len(data),batch_size):

            #get batch index
            if start+batch_size < len(data):
                stop = start+batch_size
            else:
                stop = len(data)

            feed_dict = {self.doc_inputs:data[start:stop],self.dropout:1.0}
            pred = self.sess.run(self.predictions[task],feed_dict=feed_dict)
            y_pred.append(np.argmax(pred,1))
            y_prob.append(np.max(pred,1))

        y_pred = np.concatenate(y_pred,0)
        y_prob = np.concatenate(y_prob,0)
        return y_pred, y_prob

    def save(self,filename):
        '''
        save the model weights to a file

        parameters:
          - filepath: string
            path to save model weights

        outputs:
            None
        '''
        self.saver.save(self.sess,filename)

    def load(self,filename):
        '''
        load model weights from a file

        parameters:
          - filepath: string
            path from which to load model weights

        outputs:
            None
        '''
        self.saver.restore(self.sess,filename)

if __name__ == "__main__":


    #data params
    tasks = ['site', 'laterality', 'behavior', 'grade']

    all_y_true = [[] for i in range(len(tasks))]
    all_y_pred = [[] for i in range(len(tasks))]
    all_y_prob = [[] for i in range(len(tasks))]

    def one_hot(labels,classes):
        size = len(labels)
        array = np.array(labels)
        retval = np.zeros((size,classes))
        retval[np.arange(size),array] = 1
        return retval


    val_data = []
    num_classes = []
    y_trains = []
    y_tests = []
    wv_len = 300
    seq_len = 1500

    X_trains = np.load( './data/train_X.npy' )
    y_train = np.load('./data/train_Y.npy')

    X_tests = np.load( './data/test_X.npy' )
    y_test = np.load( './data/test_Y.npy' )

    for task in range(len(y_train[0, :])):
        cat = np.unique(y_train[:, task])
        y_train[:, task] = [np.where(cat == x)[0][0] for x in y_train[:, task]]
        y_test[:, task] = [np.where(cat == x)[0][0] for x in y_test[:, task]]

    max_vocab = np.max( X_trains )
    max_vocab2 = np.max( X_tests )
    if max_vocab2 > max_vocab:
        max_vocab = max_vocab2

    wv_mat = np.random.randn( max_vocab + 1, wv_len ).astype( 'float32' ) * 0.1

    for t in range(len(tasks)):
        classes = np.max( y_train[:,t] ) + 1
        num_classes.append(classes)
        X_val = X_tests
        y_val = y_test[:,t]

        #one hot encoding for labels
        y_trains.append(one_hot(y_train[:,t],classes))
        y_val = one_hot(y_val,classes)
        y_tests.append(one_hot(y_test[:,t],classes))

        val_data.append((X_val,y_val))


    #create savedmodels directory
    if not os.path.exists('savedmodels'):
        os.makedirs('savedmodels')

    #train nn
    nn = text_cnn_multitask(wv_mat,num_classes,seq_len,len(num_classes))
    nn.train(X_trains,y_trains,epochs=5,validation_data=val_data,savebest=True,
             filepath='./savedmodels/cnn_multitask_model')

    #load best nn and make predictions
    for t in range(len(tasks)):
        nn.load('./savedmodels/cnn_multitask_model_task%i.ckpt' % (t))
        y_pred,y_prob = nn.predict(X_tests,t)
        y_true = np.argmax(y_tests[t],1)

        all_y_pred[t] = y_pred
        all_y_true[t] = y_true
        all_y_prob[t] = y_prob

    #reset graph
    tf.reset_default_graph()

    #save data
    for t in range(len(tasks)):
        y_pred = all_y_pred[t]
        y_true = all_y_true[t]
        y_prob = all_y_prob[t]
        print(y_pred.shape)
        with open('%s.csv' % tasks[t],'w') as w:
            writer = csv.writer(w)
            writer.writerow(['true','pred','prob'])
            for i in range(len(y_pred)):
                writer.writerow([y_true[i],y_pred[i],y_prob[i]])

    #get f-scores
    for t in range(len(tasks)):
        y_pred = all_y_pred[t]
        y_true = all_y_true[t]
        y_prob = all_y_prob[t]
        micro = f1_score(y_true,y_pred,average='micro')
        macro = f1_score(y_true,y_pred,average='macro')
        print('task %s test f-score: %.4f,%.4f' % (tasks[t],micro,macro))
        print(confusion_matrix(y_true,y_pred))

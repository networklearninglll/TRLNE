# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function
import tensorflow as tf

from modules import *
import os, codecs
from tqdm import tqdm
import numpy as np 
import sys

import warnings 
warnings.filterwarnings('ignore')

from classify import Classifier
from sklearn.linear_model import LogisticRegression


class Graph():
    def __init__(self, num_blocks, num_heads, node_num, fea_dim, seq_len,node_fea=None, node_fea_trainable=False, lr=0.001, dropout_rate=0.1, is_training=True, sinusoid=False):
        # self.graph = tf.Graph()
        # with self.graph.as_default():
        self.lr = lr
        self.node_num, self.fea_dim, self.seq_len = node_num, fea_dim, seq_len
        #print('feature_dim', self.fea_dim)
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.sinusoid = sinusoid
        self.num_blocks = num_blocks

        self.input_seqs = tf.placeholder(tf.int32, shape=(None, self.seq_len), name='input_seq')
        self.output_seqs = tf.placeholder(tf.int32, shape=(None, self.seq_len), name='output_seq')

        if node_fea is not None:
          #assert self.node_num == node_fea.shape[0] and self.fea_dim == node_fea.shape[1]
          self.embedding_W = tf.Variable(initial_value=node_fea, name='encoder_embed', trainable=node_fea_trainable)
        else:
          self.embedding_W = tf.Variable(initial_value=tf.random_uniform(shape=(self.node_num, self.fea_dim)),
                                       name='encoder_embed', trainable=node_fea_trainable)
        
        # define decoder inputs
        #print('output_seq dim', self.output_seqs.get_shape())
        self.decoder_inputs = tf.concat((tf.ones_like(self.output_seqs[:, :1])*2, self.output_seqs[:, :-1]), -1) # 2:<S>
        #print('decoder output dim ', self.decoder_inputs.get_shape())

                      
          # Encoder
        with tf.variable_scope("encoder"):
            ## Embedding
            self.enc = tf.nn.embedding_lookup(self.embedding_W, self.input_seqs, name='input_embed_lookup')
            self.enc = tf.layers.dense(self.enc, self.fea_dim)
            print(self.enc.get_shape)
            #self.enc = tf.layers.dense(self.enc, self.node_num)
            #print(self.enc.get_shape())
            # self.enc = embedding(self.enc_pre, 
            #                         vocab_size=self.node_num, 
            #                         num_units=hp.hidden_units, 
            #                         scale=True,
            #                         scope="enc_embed")              
            ## Positional Encoding
            if self.sinusoid:
                self.enc += positional_encoding(self.input_seqs,
                                  num_units=self.fea_dim, 
                                  zero_pad=False, 
                                  scale=False,
                                  scope="enc_pe")
            else:
                self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seqs)[1]), 0), [tf.shape(self.input_seqs)[0], 1]),
                                  vocab_size=self.seq_len, 
                                  num_units=self.fea_dim, 
                                  zero_pad=False, 
                                  scale=False,
                                  scope="enc_pe")
                
             
            ## Dropout
            self.enc = tf.layers.dropout(self.enc, 
                                        rate=self.dropout_rate, 
                                        training=tf.convert_to_tensor(is_training))
            
            ## Blocks
            #self.enc_mbedd = tf.zeros(shape=self.enc.get_shape().as_list())
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    self.enc = multihead_attention(queries=self.enc, 
                                                    keys=self.enc, 
                                                    num_units=None, 
                                                    num_heads=self.num_heads, 
                                                    dropout_rate=self.dropout_rate,
                                                    is_training=is_training,
                                                    causality=False)
                    
                    ### Feed Forward
                    self.enc = feedforward(self.enc, num_units=[4*self.fea_dim, self.fea_dim])
                    # if i == 0:
                    #     self.encoder_output = self.enc
                    # else:
                    #     self.encoder_output += self.enc
        self.encoder_output = self.enc
                    #self.encoder_output = tf.concat(self.enc, axis=-1)
        
        # Decoder
        with tf.variable_scope("decoder"):
            ## Embedding
            self.dec = embedding(self.decoder_inputs, 
                                  vocab_size=self.node_num, 
                                  num_units=self.fea_dim,
                                  scale=True, 
                                  scope="dec_embed")
            
            ## Positional Encoding
            if self.sinusoid:
                self.dec += positional_encoding(self.decoder_inputs,
                                  num_units=self.fea_dim, 
                                  zero_pad=False, 
                                  scale=False,
                                  scope="dec_pe")
            else:
                self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1]),
                                  vocab_size=self.seq_len,
                                  num_units=self.fea_dim, 
                                  zero_pad=False, 
                                  scale=False,
                                  scope="dec_pe")
            
            ## Dropout
            self.dec = tf.layers.dropout(self.dec, 
                                        rate=self.dropout_rate, 
                                        training=tf.convert_to_tensor(is_training))
            
            ## Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ## Multihead Attention ( self-attention)
                    self.dec = multihead_attention(queries=self.dec, 
                                                    keys=self.dec, 
                                                    num_units=self.fea_dim, 
                                                    num_heads=self.num_heads, 
                                                    dropout_rate=self.dropout_rate,
                                                    is_training=is_training,
                                                    causality=True, 
                                                    scope="self_attention")
                    
                    ## Multihead Attention ( vanilla attention)
                    self.dec = multihead_attention(queries=self.dec, 
                                                    keys=self.enc, 
                                                    num_units=self.fea_dim, 
                                                    num_heads=self.num_heads,
                                                    dropout_rate=self.dropout_rate,
                                                    is_training=is_training, 
                                                    causality=False,
                                                    scope="vanilla_attention")
                    
                    ## Feed Forward
                    self.dec = feedforward(self.dec, num_units=[4*self.fea_dim, self.fea_dim])
            
        # Final linear projection
        self.logits = tf.layers.dense(self.dec, self.node_num)
        self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1))
        self.istarget = tf.to_float(tf.not_equal(self.output_seqs, 0))
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.output_seqs))*self.istarget)/ (tf.reduce_sum(self.istarget))
        tf.summary.scalar('acc', self.acc)
              
        if is_training:  
            # Loss
            self.y_smoothed = label_smoothing(tf.one_hot(self.output_seqs, self.node_num))
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
            self.reward = -tf.nn.softmax_cross_entropy_with_logits(labels=self.y_smoothed, logits=self.logits)
            self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))
           
            # Training Scheme
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
            self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
               
            # Summary 
            tf.summary.scalar('mean_loss', self.mean_loss)
            self.merged = tf.summary.merge_all()

        self.tvars = tf.trainable_variables()

        # manual update parameters
        self.tvars_holders = []
        for idx, var in enumerate(self.tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.tvars_holders.append(placeholder)

        self.update_tvar_holder = []
        for idx, var in enumerate(self.tvars):
            update_tvar = tf.assign(var, self.tvars_holders[idx])
            self.update_tvar_holder.append(update_tvar)

def read_node_sequences(filename):
    seq = []
    fin = open(filename, 'r')
    for l in fin.readlines():
        vec = l.split()
        seq.append(np.array([int(x) for x in vec]))
    fin.close()
    return np.array(seq)

def read_node_features(filename):
    fea = []
    fin = open(filename, 'r')
    for l in fin.readlines():
        vec = l.split()
        fea.append(np.array([float(x) for x in vec[1:]]))
    fin.close()
    return np.array(fea, dtype='float32')

def read_node_label(filename):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')

        if len(vec) == 2:
            X.append(int(vec[0]))
            Y.append([int(v) for v in vec[1:]])
    fin.close()
    return X, Y

def read_bag_node_list(filename):
    node_list = []
    fin = open(filename, 'r')
    for l in fin.readlines():
        vec = l.split()
        if int(vec[0]) not in node_list:
            node_list.append(int(vec[0]))
    fin.close()
    return np.array(node_list)


def read_bag_node_sequences(filename):
    seq = {}
    fin = open(filename, 'r')
    for l in fin.readlines():
        vec = l.split()
        if int(vec[0]) not in seq:
            seq[int(vec[0])] = []
        seq[int(vec[0])].append(np.array([int(x) for x in vec]))
    fin.close()
    return seq


def reduce_seq2seq_hidden_add(sum_dict, count_dict, seq, seq_h_batch, seq_len, batch_start):
    for i_seq in range(seq_h_batch.shape[0]):
        for j_node in range(seq_len):
            nid = seq[i_seq + batch_start, j_node]
            if nid in sum_dict:
                sum_dict[nid] = sum_dict[nid] + seq_h_batch[i_seq, j_node, :]
            else:
                sum_dict[nid] = seq_h_batch[i_seq, j_node, :]
            if nid in count_dict:
                count_dict[nid] = count_dict[nid] + 1
            else:
                count_dict[nid] = 1
    return sum_dict, count_dict


def reduce_seq2seq_hidden_avg(sum_dict, count_dict, node_num):
    vectors = []
    for nid in range(node_num):
        vectors.append(sum_dict[nid] / count_dict[nid])
    return np.array(vectors)


def node_classification(session, bs, seqne, sequences, seq_len, node_n, samp_idx, label, ratio):
    enc_sum_dict = {}
    node_cnt = {}
    s_idx, e_idx = 0, bs
    while e_idx < len(sequences):
        batch_enc = session.run(seqne.encoder_output,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.output_seqs: sequences[s_idx: e_idx]})
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, sequences,
                                                           batch_enc.astype('float32'), seq_len, s_idx)
        s_idx, e_idx = e_idx, e_idx + bs

    if s_idx < len(sequences):
        batch_enc = session.run(seqne.encoder_output,
                                feed_dict={seqne.input_seqs: sequences[s_idx: len(sequences)], seqne.output_seqs: sequences[s_idx: e_idx]})
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, sequences,
                                                           batch_enc.astype('float32'), seq_len, s_idx)

    node_enc_mean = reduce_seq2seq_hidden_avg(sum_dict=enc_sum_dict, count_dict=node_cnt, node_num=node_n)
    lr = Classifier(vectors=node_enc_mean, clf=LogisticRegression())
    f1_micro, f1_macro = lr.split_train_evaluate(samp_idx, label, ratio)
    return f1_micro

def check_all_node_trained(trained_set, seq_list, total_node_num):
    for seq in seq_list:
        trained_set.update(seq)
    if len(trained_set) == total_node_num:
        return True
    else:
        print("node_num: ", len(trained_set))
        return False


class interaction():

    def __init__(self,sess,save_path='../model/cora/model.ckpt3'):


        self.num_blocks = 3
        self.num_heads = 5
        self.node_fea = read_node_features('../data/cora/cora.features')
        self.fea_dim = 500
        self.s_len = 10
        self.b_s = 128
        self.lr= 0.001
        self.dropout_rate = 0.1

        self.bag_node_fea = read_bag_node_sequences('../data/cora/node_sequences_10_10.txt')


        self.sess = sess
        self.model = Graph(is_training=True, sinusoid=False, num_blocks=self.num_blocks, num_heads=self.num_heads, node_num=self.node_fea.shape[0], 
          fea_dim=self.fea_dim, seq_len=self.s_len, node_fea=self.node_fea, node_fea_trainable=False, lr=self.lr, dropout_rate=self.dropout_rate)



        #self.node_fea = read_node_features('../data/cora/cora.features')
        #self.node_seq = read_node_sequences('../data/cora/node_sequences_10_10.txt')
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, save_path)


    def reward(self, batch_node_seq_input, batch_node_seq_output):

        feed_dict = {}
        feed_dict[self.model.input_seqs] = batch_node_seq_input
        feed_dict[self.model.output_seqs] = batch_node_seq_output
        outputs = (self.sess.run(self.model.reward, feed_dict = feed_dict))
        return (outputs)


    def seq_ebd(self, batch_node_seq_input, batch_node_seq_output):
        feed_dict = {}
        feed_dict[self.model.input_seqs] = batch_node_seq_input
        feed_dict[self.model.output_seqs] = batch_node_seq_output
        outputs = self.sess.run(self.model.encoder_output, feed_dict = feed_dict)
        return  (outputs)



    def update_stne(self,update_node_seq, updaterate):


        num_steps = len(update_node_seq) // self.b_s

        with self.sess.as_default():

            tvars_old = self.sess.run(self.model.tvars)

            for i in range(num_steps):

                batch_seq = update_node_seq[i* self.b_s:(i+1)*self.b_s]

                feed_dict = {}
                feed_dict[self.model.input_seqs] = batch_seq
                feed_dict[self.model.output_seqs] = batch_seq

                #train_op = tf.train.RMSPropOptimizer(lr).minimize(model.loss_ce, global_step=model.global_step)
                #_, loss, accuracy = sess.run([self.model.train_op,self.model.final_loss, self.model.accuracy], feed_dict=feed_dict)
                # self.sess.run(self.model.train_op, feed_dict=feed_dict)
                self.sess.run([self.model.train_op, self.model.mean_loss, self.model.global_step], feed_dict=feed_dict)

            # get tvars_new
            tvars_new = self.sess.run(self.model.tvars)

            # update old variables of the target network
            tvars_update = self.sess.run(self.model.tvars)
            for index, var in enumerate(tvars_update):
                tvars_update[index] = updaterate * tvars_new[index] + (1 - updaterate) * tvars_old[index]

            feed_dict = dictionary = dict(zip(self.model.tvars_holders, tvars_update))
            self.sess.run(self.model.update_tvar_holder, feed_dict)

    def produce_new_embedding(self):

        # produce reward sentence_ebd  average_reward

        node_fea = self.node_fea
        bag_node_fea = self.bag_node_fea
        #node_seq = self.node_seq
        all_sentence_ebd = []
        all_reward = []
        all_reward_list = []
        #len_batch = len(node_seq) // self.b_s

        with self.sess.as_default():

            for batch in bag_node_fea:
                #start_idx, end_idx = 0, b_s
                batch_seq = bag_node_fea[batch]

                tmp_sentence_ebd = self.seq_ebd(batch_seq, batch_seq)
                tmp_reward = self.reward(batch_seq, batch_seq)

                all_sentence_ebd.append(tmp_sentence_ebd)
                all_reward.append(tmp_reward)
                all_reward_list += list(tmp_reward)

            all_reward_list = np.array(all_reward_list)
            average_reward = np.mean(all_reward_list)
            average_reward = np.array(average_reward)

            all_sentence_ebd = np.array(all_sentence_ebd)
            all_reward = np.array(all_reward)

            return average_reward,all_sentence_ebd,all_reward

    def save_stnemodel(self,save_path):
        with self.sess.as_default():
            self.saver.save(self.sess, save_path=save_path)

    def tvars(self):
        with self.sess.as_default():
            tvars = self.sess.run(self.model.tvars)
            return tvars

    def update_tvars(self,tvars_update):
        with self.sess.as_default():
            feed_dict = dictionary = dict(zip(self.model.tvars_holders, tvars_update))
            self.sess.run(self.model.update_tvar_holder, feed_dict)

    def classification_result(self):
        folder = '../data/cora/'
        fn = '../data/cora/result.txt'
        X, Y = read_node_label(folder + 'labels.txt')
        node_seq = read_node_sequences(folder + 'node_sequences_10_10.txt')
        clf_ratio = [0.1, 0.2, 0.3, 0.4, 0.5]
        f1_mi = []
        for ratio in clf_ratio:
            f1_mi.append(node_classification(session=self.sess, bs=self.b_s, seqne=self.model, sequences=node_seq,
                                                             seq_len=self.s_len, node_n=self.node_fea.shape[0], samp_idx=X,
                                                     label=Y, ratio=ratio))
        return f1_mi

    def classification_selected_result(self, trained_node_set, node_seq):
        folder = '../data/cora/'
        fn = '../data/cora/result.txt'
        X, Y = read_node_label(folder + 'labels.txt')
        all_trained = check_all_node_trained(trained_node_set, node_seq, self.node_fea.shape[0])
        clf_ratio = [0.1, 0.2, 0.3, 0.4, 0.5]
        f1_mi = []
        if all_trained:
            
            for ratio in clf_ratio:
                f1_mi.append(node_classification(session=self.sess, bs=self.b_s, seqne=self.model, sequences=node_seq,
                                                                 seq_len=self.s_len, node_n=self.node_fea.shape[0], samp_idx=X,
                                                         label=Y, ratio=ratio))
            return f1_mi
        else:
            return f1_mi


# produce reward sentence_ebd  average_reward
def produce_rldata(save_path):

    with tf.Session() as sess:
        # start = time.time()
        interact = interaction(sess, save_path)
        average_reward, all_sentence_ebd, all_reward = interact.produce_new_embedding()

        np.save('../model/cora/average_reward.npy', average_reward)
        np.save('../model/cora/all_sentence_ebd.npy', all_sentence_ebd)
        np.save('../model/cora/all_reward.npy', all_reward)

        print (average_reward)


if __name__ == '__main__':                



    node_seq = read_node_sequences('../data/wiki/node_sequences_10_10.txt')
    #node_seq_old = read_node_sequences('../data/cora/node_sequences_10_10.txt')
    #node_seq = np.load('../model/cora/selected_seq.npy')
    node_feature = read_node_features('../wiki/wiki/wiki.features')
    X, Y = read_node_label('../data/wiki/labels.txt')
    node_bag_seq = read_bag_node_sequences('../data/wiki/node_sequences_10_10.txt')
    node_bag_list = read_bag_node_list('../data/wiki/node_sequences_10_10.txt')

    num_epochs = 5
    b_s = 128
    s_len = 20
    fea_dim = 500
    clf_ratio = [0.1, 0.2, 0.3, 0.4, 0.5]

    f1_mi_best = []
    for i in range(0, 5):
        f1_mi_best.append(0)

    config = tf.ConfigProto(allow_soft_placement=True)

    #最多占gpu资源的40%
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)

    #开始不会给tensorflow全部gpu资源 而是按需增加
    config.gpu_options.allow_growth = False
    num_blocks_list = [1, 2, 3, 4, 5, 6]
    num_heads_list = [2, 4, 5, 10]




    with tf.Session() as sess:
    # Construct graph
      model = Graph(is_training=True, sinusoid=False, num_blocks=4, num_heads=1, node_num=node_feature.shape[0], fea_dim=fea_dim, seq_len=s_len, node_fea=node_feature, node_fea_trainable=False, lr=0.001, dropout_rate=0.1)
      print("Graph loaded")
      sess.run(tf.global_variables_initializer())
      saver = tf.train.Saver()
    

      trained_node_set = set()
      all_trained = False
      for epoch in range(num_epochs): 
        start_idx, end_idx = 0, b_s
        while end_idx < len(node_seq):
          _, loss, step = sess.run([model.train_op, model.mean_loss, model.global_step], feed_dict={model.input_seqs: node_seq[start_idx:end_idx], model.output_seqs: node_seq[start_idx:end_idx]})
          if not all_trained:
            all_trained = check_all_node_trained(trained_node_set, node_seq[start_idx:end_idx],
                                                         node_feature.shape[0])
          # if step >= 90:
          #   break
          if step % 10 == 0:
            print(epoch, '\t', step, '\t', loss)
            if all_trained:
              f1_mi = []
              for ratio in clf_ratio:
                f1_mi.append(node_classification(session=sess, bs=1024, seqne=model, sequences=node_seq,
                                                 seq_len=s_len, node_n=node_feature.shape[0], samp_idx=X,
                                                 label=Y, ratio=ratio))
              for ii in range(0, len(f1_mi)):
                if f1_mi[ii] > f1_mi_best[ii]:
                    f1_mi_best[ii] = f1_mi[ii]
                print(f1_mi[ii]) 
                # if ii ==  (len(f1_mi) -1)  and f1_mi[ii] >= 0.83 :
                #     saver.save(sess, save_path="../model/cora/stne_model_transformer1.ckpt")
                #     print('保存模型')

          start_idx, end_idx = end_idx, (end_idx + b_s)
        if start_idx < len(node_seq):
          sess.run([model.train_op, model.mean_loss, model.global_step], feed_dict={
                model.input_seqs: node_seq[start_idx:end_idx], model.output_seqs: node_seq[start_idx:end_idx]})

        if all_trained:
          f1_mi = []
          for ratio in clf_ratio:
            f1_mi.append(node_classification(session=sess, bs=1024, seqne=model, sequences=node_seq,
                                             seq_len=s_len, node_n=node_feature.shape[0], samp_idx=X,
                                             label=Y, ratio=ratio))
          for ii in range(0, len(f1_mi)):
            if f1_mi[ii] > f1_mi_best[ii]:
                f1_mi_best[ii] = f1_mi[ii]
            print(f1_mi[ii])

        saver.save(sess, save_path="../model/wiki/trlne_model.ckpt")

        
      
        

    for ii in range(0, len(f1_mi_best)):
        #fw.write(str(f1_mi_best[ii]) + "\n")
        print(f1_mi_best[ii])
    #fw.close()

    
    # print ('produce reward seq_emdedding  average_reward for rlmodel')
    # produce_rldata(save_path='../model/wiki/stne_model_transformer.ckpt')

    

    


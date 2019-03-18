import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
from tqdm import tqdm
import time
import random
import tqdm
import sys

from stne_transformer_new_reward import read_node_features
from stne_transformer_new_reward import read_node_sequences
from stne_transformer_new_reward import read_bag_node_sequences
from stne_transformer_new_reward import read_bag_node_list
from stne_transformer_new_reward import read_node_degree
from classify import Classifier, read_node_label
from stne_transformer_new_reward import Graph
import stne_transformer_new_reward

import numpy as np


class environment():

    def __init__(self,seq_len):
        self.seq_len = seq_len


    def reset(self,batch_seq_ebd,batch_reward):

        self.batch_reward = batch_reward
        self.batch_len = len(batch_seq_ebd)
        self.seq_ebd = batch_seq_ebd

        self.current_step = 0
        self.num_selected = 0
        self.list_selected = []

        self.vector_current = np.sum(self.seq_ebd[self.current_step], axis=0)/len(self.seq_ebd[self.current_step])

        self.vector_mean = np.array([0.0 for x in range(self.seq_len)],dtype=np.float32)

        self.vector_sum = np.array([0.0 for x in range(self.seq_len)],dtype=np.float32)

        current_state = [self.vector_current, self.vector_mean]
        return current_state


    def step(self,action):

        if action == 1:
            self.num_selected +=1
            self.list_selected.append(self.current_step)

        self.vector_sum =self.vector_sum + action * self.vector_current
        
        if self.num_selected == 0:
            self.vector_mean = np.array([0.0 for x in range(self.seq_len)],dtype=np.float32)
        else:
            self.vector_mean = self.vector_sum / self.num_selected

        self.current_step +=1

        if (self.current_step < self.batch_len):
            self.vector_current = np.sum(self.seq_ebd[self.current_step], axis=0)/len(self.seq_ebd[self.current_step])

        current_state = [self.vector_current, self.vector_mean]
        return current_state

    def reward(self):
        assert (len(self.list_selected) == self.num_selected)
        reward = [self.batch_reward[x] for x in self.list_selected]
        reward = np.array(reward)
        reward = np.mean(reward)
        return reward


def get_action(prob):

    tmp = prob[0]
    result = np.random.rand()
    if result>0 and result< tmp:
        return 1
    elif result >=tmp and result<1:
        return 0


def decide_action(prob):
    tmp = prob[0]
    if tmp>=0.5:
        return 1
    elif tmp < 0.5:
        return 0




class agent():
    def __init__(self, lr,s_size):


        #get action

        self.state_in = tf.placeholder(shape=[None, s_size*2], dtype=tf.float32)
        self.prob = tf.reshape(layers.fully_connected(self.state_in,1,tf.nn.sigmoid),[-1])

        #compute loss
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.float32)

        #the probability of choosing 0 or 1
        self.pi  = self.action_holder * self.prob + (1 - self.action_holder) * (1 - self.prob)

        #loss
        self.loss = -tf.reduce_sum(tf.log(self.pi) * self.reward_holder)

        # minimize loss
        optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = optimizer.minimize(self.loss)

        self.tvars = tf.trainable_variables()

        #manual update parameters
        self.tvars_holders = []
        for idx, var in enumerate(self.tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.tvars_holders.append(placeholder)

        self.update_tvar_holder = []
        for idx, var in enumerate(self.tvars):
            update_tvar = tf.assign(var, self.tvars_holders[idx])
            self.update_tvar_holder.append(update_tvar)


        #compute gradient
        self.gradients = tf.gradients(self.loss, self.tvars)

        #update parameters using gradient
        self.gradient_holders = []
        for idx, var in enumerate(self.tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, self.tvars))


def train(num_epoch, full_seq_name):

    folder = '../data/wiki/'
    X, Y = read_node_label(folder + 'labels.txt')
    node_fea = read_node_features(folder + 'wiki.features')
    node_seq = read_node_sequences(folder + 'node_sequences_10_10.txt')
    node_bag_seq = read_bag_node_sequences(folder + 'node_sequences_10_10.txt')
    node_bag_list = read_bag_node_list(folder + 'node_sequences_10_10.txt')
    node_degree = read_node_degree(folder + 'node_degree.txt')

    all_sentence_ebd = np.load('../model/wiki/all_sentence_ebd.npy')
    print(all_sentence_ebd.shape)
    #print("sentence length", len(all_sentence_ebd[0][0]))
    all_reward = np.load('../model/wiki/all_reward.npy')
    average_reward = np.load('../model/wiki/average_reward.npy')


    g_stne = tf.Graph()
    g_rl = tf.Graph()
    sess1 = tf.Session(graph=g_stne)
    sess2 = tf.Session(graph=g_rl)


    with g_stne.as_default():
        with sess1.as_default():
            interact = stne_transformer_new_reward.interaction(sess1,save_path='../model/wiki/stne_model_transformer.ckpt')
            tvars_best_cnn = interact.tvars()
            for index, var in enumerate(tvars_best_cnn):
                tvars_best_cnn[index] = var * 0

    g_stne.finalize()
    env = environment(500)
    best_score = -100000



    with g_rl.as_default():
        with sess2.as_default():


            myAgent = agent(0.02,500)
            updaterate = 0.01
            #num_epoch = 10
            sampletimes = 3
            best_reward = -100000

            init = tf.global_variables_initializer()
            sess2.run(init)
            saver = tf.train.Saver()
            saver.restore(sess2, save_path='../model/wiki/stne_transformer_model_rl_model.ckpt')

            tvars_best_rl = sess2.run(myAgent.tvars)
            for index, var in enumerate(tvars_best_rl):
                tvars_best_rl[index] = var * 0

            tvars_old = sess2.run(myAgent.tvars)


            gradBuffer = sess2.run(myAgent.tvars)
            for index, grad in enumerate(gradBuffer):
                gradBuffer[index] = grad * 0

            g_rl.finalize()

            trained_node_set = set()
            update_full_seq = []
            for epoch in range(num_epoch):

                update_seq = []

                all_list = list(range(len(all_sentence_ebd)))
                total_reward = []

                # shuffle bags
                random.shuffle(all_list)

                print ('update the rlmodel')
                for batch in tqdm.tqdm(all_list):
                #for batch in tqdm.tqdm(range(10000)):

                    batch_node = node_bag_list[batch]
                    batch_sentence_ebd = all_sentence_ebd[batch]
                    batch_reward = all_reward[batch]
                    batch_len = len(batch_sentence_ebd)

                    batch_seq = node_bag_seq[batch_node]



                    list_list_state = []
                    list_list_action = []
                    list_list_reward = []
                    avg_reward  = 0


                    # add sample times
                    for j in range(sampletimes):
                        #reset environment
                        state = env.reset(batch_sentence_ebd, batch_reward)
                        list_action = []
                        list_state = []
                        old_prob = []


                        #get action
                        #start = time.time()
                        for i in range(batch_len):

                            state_in = np.append(state[0],state[1])
                            feed_dict = {}
                            feed_dict[myAgent.state_in] = [state_in]
                            prob = sess2.run(myAgent.prob,feed_dict = feed_dict)
                            old_prob.append(prob[0])
                            action = get_action(prob)
                            '''
                            if action == None:
                                print (123)
                            action = 1
                            '''
                            #add produce data for training cnn model
                            list_action.append(action)
                            list_state.append(state)
                            state = env.step(action)
                        #end = time.time()
                        #print ('get action:',end - start)

                        if env.num_selected == 0:
                            tmp_reward = average_reward
                        else:
                            tmp_reward = env.reward()

                        avg_reward += tmp_reward
                        list_list_state.append(list_state)
                        list_list_action.append(list_action)
                        list_list_reward.append(tmp_reward)


                    avg_reward = avg_reward / sampletimes
                    # add sample times
                    for j in range(sampletimes):

                        list_state = list_list_state[j]
                        list_action = list_list_action[j]
                        reward = list_list_reward[j]

                        # compute gradient
                        # start = time.time()
                        list_reward = [reward - avg_reward for x in range(batch_len)]
                        list_state_in = [np.append(state[0],state[1]) for state in list_state]

                        feed_dict = {}
                        feed_dict[myAgent.state_in] = list_state_in
                        feed_dict[myAgent.reward_holder] = list_reward
                        feed_dict[myAgent.action_holder] = list_action

                        grads = sess2.run(myAgent.gradients, feed_dict=feed_dict)
                        for index, grad in enumerate(grads):
                            gradBuffer[index] += grad
                        #end = time.time()
                        #print('get loss and update:', end - start)

                    #decide action and compute reward
                    state = env.reset(batch_sentence_ebd, batch_reward)
                    old_prob = []
                    for i in range(batch_len):
                        state_in = np.append(state[0], state[1])
                        feed_dict = {}
                        feed_dict[myAgent.state_in] = [state_in]
                        prob = sess2.run(myAgent.prob, feed_dict=feed_dict)
                        old_prob.append(prob[0])
                        action = decide_action(prob)
                        state = env.step(action)
                    chosen_reward = [batch_reward[x] for x in env.list_selected]
                    total_reward += chosen_reward

                    update_seq += [batch_seq[x] for x in env.list_selected]
                    # if epoch == 0:
                    #     pass
                    # else:
                    #     update_full_seq += [batch_seq[x] for x in env.list_selected]
                    update_full_seq += [batch_seq[x] for x in env.list_selected]
                print ('finished')

                #print (len(update_word),len(update_pos1),len(update_pos2),len(update_y),updaterate)

                #train and update cnnmodel
                print('update the stnemodel')
                interact.update_stne(update_seq, updaterate)
                print('finished')

                # classification result
                print('classification result')
                f1_mi = interact.classification_result()
                for f1 in f1_mi:
                    print(f1)
                # classification new result
                # print('classification new result')
                # f1_mi_new = interact.classification_selected_result(trained_node_set, np.array(update_seq))
                # for f1_new in f1_mi_new:
                #     print(f1_new)
                # print('finished')

                #produce new embedding
                print ('produce new embedding')
                average_reward, all_sentence_ebd, all_reward = interact.produce_new_embedding()
                np.save('../model/wiki/average_reward_new.npy', average_reward)
                np.save('../model/wiki/all_sentence_ebd_new.npy', all_sentence_ebd)
                np.save('../model/wiki/all_reward_new.npy', all_reward)
                average_score = average_reward
                print ('finished')

                #update the rlmodel
                #apply gradient
                feed_dict = dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                sess2.run(myAgent.update_batch, feed_dict=feed_dict)
                for index, grad in enumerate(gradBuffer):
                    gradBuffer[index] = grad * 0

                #get tvars_new
                tvars_new = sess2.run(myAgent.tvars)

                # update old variables of the target network
                tvars_update = sess2.run(myAgent.tvars)
                for index, var in enumerate(tvars_update):
                    tvars_update[index] = updaterate * tvars_new[index] + (1-updaterate) * tvars_old[index]

                feed_dict = dictionary = dict(zip(myAgent.tvars_holders, tvars_update))
                sess2.run(myAgent.update_tvar_holder, feed_dict)
                tvars_old = sess2.run(myAgent.tvars)
                #break


                #find the best parameters
                chosen_size = len(total_reward)
                total_reward = np.mean(np.array(total_reward))


                if (total_reward > best_reward):
                    best_reward = total_reward
                    tvars_best_rl = tvars_old

                if  average_score > best_score:
                    best_score = average_score
                    #tvars_best_rl = tvars_old
                print ('epoch:',epoch)
                print ('chosen seq size:',chosen_size)
                print ('total_reward:',total_reward)
                print ('best_reward',best_reward)
                print ('average score',average_score)
                print ('best score',best_score)


            #set parameters = best_tvars
            feed_dict = dictionary = dict(zip(myAgent.tvars_holders, tvars_best_rl))
            sess2.run(myAgent.update_tvar_holder, feed_dict)
            #save model
            saver.save(sess2, save_path='../model/wiki/union_rl_model.ckpt')
            update_full_seq = np.array(update_full_seq)
            np.save(full_seq_name, update_full_seq)

    #interact.update_tvars(tvars_best_cnn)
    interact.save_stnemodel(save_path='../model/wiki/union_cnn_model.ckpt')



def select(save_path, seq_name):

    folder = '../data/wiki/'
    X, Y = read_node_label(folder + 'labels.txt')
    node_fea = read_node_features(folder + 'wiki.features')
    node_seq = read_node_sequences(folder + 'node_sequences_10_10.txt')
    node_bag_seq = read_bag_node_sequences(folder + 'node_sequences_10_10.txt')
    node_bag_list = read_bag_node_list(folder + 'node_sequences_10_10.txt')

    all_sentence_ebd = np.load('../model/wiki/all_sentence_ebd_new.npy')
    all_reward = np.load('../model/wiki/all_reward_new.npy')
    average_reward = np.load('../model/wiki/average_reward_new.npy')



    selected_seq = []
    print("selected_seq")

    g_rl = tf.Graph()
    sess2 = tf.Session(graph=g_rl)
    env = environment(500)


    with g_rl.as_default():
        with sess2.as_default():

            myAgent = agent(0.02, 500)
            init = tf.global_variables_initializer()
            sess2.run(init)
            saver = tf.train.Saver()
            saver.restore(sess2, save_path=save_path)
            g_rl.finalize()


            for epoch in range(1):

                total_reward = []
                num_chosen = 0

                all_list = list(range(len(all_sentence_ebd)))

                for batch in tqdm.tqdm(all_list):

                    batch_node = node_bag_list[batch]
                    batch_sentence_ebd = all_sentence_ebd[batch]
                    batch_reward = all_reward[batch]
                    batch_len = len(batch_sentence_ebd)

                    batch_seq = node_bag_seq[batch_node]

                    # reset environment
                    state = env.reset(batch_sentence_ebd, batch_reward)
                    old_prob = []

                    # get action
                    # start = time.time()
                    for i in range(batch_len):
                        state_in = np.append(state[0], state[1])
                        feed_dict = {}
                        feed_dict[myAgent.state_in] = [state_in]
                        prob = sess2.run(myAgent.prob, feed_dict=feed_dict)
                        old_prob.append(prob[0])
                        action = decide_action(prob)
                        # produce data for training cnn model
                        state = env.step(action)
                        if action == 1:
                            num_chosen+=1
                    #print (old_prob)
                    chosen_reward = [batch_reward[x] for x in env.list_selected]
                    total_reward += chosen_reward

                    selected_seq += [batch_seq[x] for x in env.list_selected]
                print(num_chosen)
    selected_seq = np.array(selected_seq)

    np.save(seq_name, selected_seq)


if __name__ =='__main__':
    num_epoch = int(sys.argv[1])
    full_seq_name = '../model/wiki/update_full_seq_' + str(num_epoch) + '.npy'
    train(num_epoch=num_epoch, full_seq_name=full_seq_name)

    seq_name = '../model/wiki/selected_seq_' + str(num_epoch) + '.npy'

    print('select training data')
    select(save_path = '../model/wiki/union_rl_model.ckpt', seq_name=seq_name)

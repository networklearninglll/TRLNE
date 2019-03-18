import tensorflow as tf
import numpy as np
import random
import tensorflow.contrib.layers as layers
from tqdm import tqdm
import time
import stne_transformer
import random
import tqdm
from stne_transformer import read_node_features
from stne_transformer import read_node_sequences
from stne_transformer import read_bag_node_sequences
from stne_transformer import read_bag_node_list
from classify import Classifier, read_node_label
from stne_transformer import Graph
#from stne_transformer import check_all_node_trained

import sys

import warnings 
warnings.filterwarnings('ignore')

class environment():

    def __init__(self,seq_len):
        self.seq_len = seq_len


    def reset(self, batch_seq_ebd, batch_reward):
        self.batch_reward = batch_reward
        self.batch_len = len(batch_seq_ebd)
        self.seq_ebd = batch_seq_ebd
        self.current_step = 0
        self.num_selected = 0
        self.list_selected = []
        #print(type(self.seq_ebd[self.current_step]))
        #print("self.seq_ebd[self.current_step].shape", self.seq_ebd[self.current_step].shape)
        self.vector_current = np.sum(self.seq_ebd[self.current_step], axis=0)/len(self.seq_ebd[self.current_step])
        #print("self.vector_current.shape", self.vector_current.shape)
        self.vector_mean = np.array([0.0 for x in range(self.seq_len)],dtype=np.float32)
        #print(self.vector_mean.shape)
        self.vector_sum = np.array([0.0 for x in range(self.seq_len)],dtype=np.float32)

        current_state = [self.vector_current, self.vector_mean]
        return current_state


    def step(self, action):

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
    def __init__(self, lr, s_size):


        #get action
        #node_embedding = tf.get_variable(name = 'node_embedding',initializer=node_ebd, trainable=False)
        


        self.state_in = tf.placeholder(shape=[None, s_size*2], dtype=tf.float32)
        # self.node_seq  = tf.placeholder(dtype=tf.int32, shape=[None], name='node_seq')

        # self.seq_ebd = tf.nn.embedding_lookup(node_embedding, self.node_seq)


        # self.input = tf.concat(axis=1,values = [self.state_in,self.seq_ebd])

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


def train():

    folder = '../data/wiki/'
    X, Y = read_node_label(folder + 'labels.txt')
    node_fea = read_node_features(folder + 'wiki.features')
    node_seq = read_node_sequences(folder + 'node_sequences_10_10.txt')
    node_bag_seq = read_bag_node_sequences(folder + 'node_sequences_10_10.txt')
    node_bag_list = read_bag_node_list(folder + 'node_sequences_10_10.txt')

    all_sentence_ebd = np.load('../model/wiki/all_sentence_ebd.npy')
    print(all_sentence_ebd.shape)
    #print("sentence length", len(all_sentence_ebd[0][0]))
    all_reward = np.load('../model/wiki/all_reward.npy')
    average_reward = np.load('../model/wiki/average_reward.npy')

    g_rl = tf.Graph()
    sess2 = tf.Session(graph=g_rl)
    env = environment(500)


    with g_rl.as_default():
        with sess2.as_default():

            myAgent = agent(0.03, 500)
            updaterate = 1
            num_epoch = 5
            sampletimes = 3
            best_reward = -100000

            init = tf.global_variables_initializer()
            sess2.run(init)
            saver = tf.train.Saver()
            #saver.restore(sess2, save_path='rlmodel/rl.ckpt')

            # 对于需要训练的变量置位零
            tvars_best = sess2.run(myAgent.tvars)
            for index, var in enumerate(tvars_best):
                tvars_best[index] = var * 0

            # 保存历史的需要训练的变量
            tvars_old = sess2.run(myAgent.tvars)

            # 梯度的置为零
            gradBuffer = sess2.run(myAgent.tvars)
            for index, grad in enumerate(gradBuffer):
                gradBuffer[index] = grad * 0

            g_rl.finalize()

            for epoch in range(num_epoch):

                all_list = list(range(len(all_sentence_ebd)))
                total_reward = []

                # shuffle bags
                random.shuffle(all_list)
                # 对Bag进行shuffle

                for batch in tqdm.tqdm(all_list):
                    #print("batch", batch)
                #for batch in tqdm.tqdm(range(10000)):

                    # 取出来bag的实体对和对应的sentence，以及对应的reward
                    bath_node = node_bag_list[batch]
                    batch_sentence_ebd = all_sentence_ebd[batch]
                    #print("batch_sentence_ebd", batch_sentence_ebd.shape)
                    batch_reward = all_reward[batch]
                    batch_len = len(batch_sentence_ebd)




                    list_list_state = []
                    list_list_action = []
                    list_list_reward = []
                    avg_reward  = 0


                    # add sample times
                    for j in range(sampletimes):
                        #reset environment
                        # 环境的reset，返回当前state，历史state平均，实体对
                        state = env.reset(batch_sentence_ebd, batch_reward)
                        #print('state shape' ,state[0].shape, state[1].shape)
                        list_action = []
                        list_state = []
                        old_prob = []


                        #get action
                        #start = time.time()
                        for i in range(batch_len):

                            state_in = np.append(state[0],state[1])
                            # print("state num", i)
                            # print("state_in.shape", state_in.shape)
                            feed_dict = {}
                            #feed_dict[myAgent.node_seq] = [state[1]]
                            feed_dict[myAgent.state_in] = [state_in]
                            # 根据state计算概率，并根据概率选择action
                            prob = sess2.run(myAgent.prob, feed_dict = feed_dict)
                            # print("prob", prob)
                            old_prob.append(prob[0])
                            action = get_action(prob)
                            #add produce data for training cnn model
                            # 把action和state进行记录，方便后续更新cnnmodel
                            list_action.append(action)
                            list_state.append(state)
                            # 根据采取的action更新state
                            state = env.step(action)
                        #end = time.time()
                        #print ('get action:',end - start)

                        if env.num_selected == 0:
                            tmp_reward = average_reward
                        else:
                            tmp_reward = env.reward()
                        # 累加reward
                        avg_reward += tmp_reward
                        # 记录采取的action和reward
                        list_list_state.append(list_state)
                        list_list_action.append(list_action)
                        list_list_reward.append(tmp_reward)


                    avg_reward = avg_reward / sampletimes
                    # add sample times
                    for j in range(sampletimes):


                        # 取出来上面进行探索的随影的action，state和reward
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
                        '''
                        loss =sess2.run(myAgent.loss, feed_dict=feed_dict)
                        if loss == float("-inf"):
                            probs,pis = sess2.run([myAgent.prob,myAgent.pi], feed_dict=feed_dict)
                            print(' ')
                            print ('batch:',batch)
                            print (old_prob)
                            print (list_action)
                            print(probs)
                            print (pis)
                            print('error!')
                            return 0
                        '''
                        # 计算梯度
                        grads = sess2.run(myAgent.gradients, feed_dict=feed_dict)
                        for index, grad in enumerate(grads):
                            gradBuffer[index] += grad
                        #end = time.time()
                        #print('get loss and update:', end - start)
                        '''
                        print (len(list_state),len(list_action),len(list_reward),len(list_entity1),len(list_entity2))
                        print (list_action)
                        print (list_reward)
                        print (list_entity1)
                        print (list_entity2)
                        break
                        '''
                    #decide action and compute reward
                    # reset环境
                    state = env.reset(batch_sentence_ebd, batch_reward)
                    old_prob = []
                    for i in range(batch_len):
                        # 决定action，计算reward
                        state_in = np.append(state[0], state[1])
                        feed_dict = {}
                        #feed_dict[myAgent.node_seq] = [state[1]]
                        feed_dict[myAgent.state_in] = [state_in]
                        prob = sess2.run(myAgent.prob, feed_dict=feed_dict)
                        old_prob.append(prob[0])
                        action = decide_action(prob)
                        state = env.step(action)
                    chosen_reward = [batch_reward[x] for x in env.list_selected]
                    total_reward += chosen_reward


                #apply gradient 计算梯度之后进行应用梯度
                feed_dict = dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                sess2.run(myAgent.update_batch, feed_dict=feed_dict)
                for index, grad in enumerate(gradBuffer):
                    gradBuffer[index] = grad * 0

                #get tvars_new 计算最新的需要的变量
                tvars_new = sess2.run(myAgent.tvars)

                # update old variables of the target network
                # 更新参数
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
                    tvars_best = tvars_old
                #print ('chosen sentence size:',chosen_size)
                #print ('total_reward:',total_reward)
                #print ('best_reward',best_reward)


            #set parameters = best_tvars
            feed_dict = dictionary = dict(zip(myAgent.tvars_holders, tvars_best))
            sess2.run(myAgent.update_tvar_holder, feed_dict)
            #save model
            saver.save(sess2, save_path='../model/wiki/stne_transformer_model_rl_model.ckpt')


def select(save_path):

    folder = '../data/cora/'
    X, Y = read_node_label(folder + 'labels.txt')
    node_fea = read_node_features(folder + 'cora.features')
    node_seq = read_node_sequences(folder + 'node_sequences_10_10.txt')
    node_bag_seq = read_bag_node_sequences(folder + 'node_sequences_10_10.txt')
    node_bag_list = read_bag_node_list(folder + 'node_sequences_10_10.txt')

    all_sentence_ebd = np.load('../model/cora/all_sentence_ebd.npy')
    all_reward = np.load('../model/cora/all_reward.npy')
    average_reward = np.load('../model/cora/average_reward.npy')



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

    np.save('../model/cora/selected_seq.npy', selected_seq)


def train_stne(node_seq):
    folder = '../data/cora/'
    fn = '../data/cora/result.txt'

    dpt = 1            # Depth of both the encoder and the decoder layers (MultiCell RNN)
    h_dim = 500        # Hidden dimension of encoder LSTMs
    s_len = 10         # Length of input node sequence
    epc = 20            # Number of training epochs
    trainable = False  # Node features trainable or not
    dropout = 0.2      # Dropout ration
    clf_ratio = [0.1, 0.2, 0.3, 0.4, 0.5]  # Ration of training samples in subsequent classification
    b_s = 128          # Size of batches
    lr = 0.001         # Learning rate of RMSProp

    start = time.time()
    fobj = open(fn, 'w')
    X, Y = read_node_label(folder + 'labels.txt')
    node_fea = read_node_features(folder + 'cora.features')
    

    with tf.Session() as sess:
        model = STNE(hidden_dim=h_dim, node_fea_trainable=trainable, seq_len=s_len, depth=dpt, node_fea=node_fea,
                     node_num=node_fea.shape[0], fea_dim=node_fea.shape[1],  lr=0.001)
        #train_op = tf.train.RMSPropOptimizer(lr).minimize(model.loss_ce, global_step=model.global_step)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        trained_node_set = set()
        all_trained = False
        for epoch in range(epc):
            start_idx, end_idx = 0, b_s
            print('Epoch,\tStep,\tLoss,\t#Trained Nodes')
            while end_idx < len(node_seq):
                _, loss, step = sess.run([model.train_op, model.loss_ce, model.global_step], feed_dict={
                    model.input_seqs: node_seq[start_idx:end_idx], model.dropout: dropout})
                if not all_trained:
                    all_trained = check_all_node_trained(trained_node_set, node_seq[start_idx:end_idx],
                                                         node_fea.shape[0])

                if step % 10 == 0:
                    print(epoch, '\t', step, '\t', loss, '\t', len(trained_node_set))
                    # if all_trained:
                    #     f1_mi = []
                    #     for ratio in clf_ratio:
                    #         f1_mi.append(node_classification(session=sess, bs=b_s, seqne=model, sequences=node_seq,
                    #                                          seq_len=s_len, node_n=node_fea.shape[0], samp_idx=X,
                    #                                          label=Y, ratio=ratio))

                    #     print('step ', step)
                    #     fobj.write('step ' + str(step) + ' ')
                    #     for f1 in f1_mi:
                    #         print(f1)
                    #         fobj.write(str(f1) + ' ')
                    #     fobj.write('\n')
                start_idx, end_idx = end_idx, end_idx + b_s

            if start_idx < len(node_seq):
                sess.run([model.train_op, model.loss_ce, model.global_step],
                         feed_dict={model.input_seqs: node_seq[start_idx:len(node_seq)], model.dropout: dropout})

            minute = np.around((time.time() - start) / 60)
            print('\nepoch ', epoch, ' finished in ', str(minute), ' minutes\n')

            saver.save(sess, save_path="../model/cora/select_stne_model.ckpt")




if __name__ =='__main__':
    print ('train rlmodel')
    train()

    # print('select training data')
    # select(save_path = '../model/cora/stne_transformer_model_rl_model.ckpt')

    # print ('use the selected data to train stne model')
    # # node_seq = read_node_sequences('../data/cora/node_sequences_10_10.txt')
    # node_seq = np.load("../rlmodel/cora/selected_seq.npy")
    # train_stne(node_seq)

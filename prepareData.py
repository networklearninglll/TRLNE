import random
import codecs

f = open('./zhihu/zhihu.edgelist','rb')
ratio_list = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
#ratio_list = [0.95]
edges = [i for i in f]
random.shuffle(edges)


for ratio in ratio_list:
    train_file = './zhihu/train_graph_' + str(ratio) + '.txt'
    test_file = './zhihu/test_graph_' + str(ratio) + '.txt'

    selected = random.sample(edges,int(len(edges)*ratio))
    remain = [i for i in edges if i not in selected]

    fw1 = open(train_file,'wb')
    fw2 = open(test_file,'wb')

    for i in selected:
        fw1.write(i)
    for i in remain:
        fw2.write(i)
    fw1.close()
    fw2.close()
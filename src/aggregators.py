import tensorflow as tf
from abc import abstractmethod

LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Aggregator(object):
    def __init__(self, batch_size, dim, dropout, act, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings,neighbor_num):
        outputs = self._call(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings,neighbor_num)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        # dimension:
        # self_vectors: [batch_size, -1, dim]
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim]
        # neighbor_relations: [batch_size, -1, n_neighbor, dim]
        # user_embeddings: [batch_size, dim]
        pass

    
    
class  BiInteractionAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(BiInteractionAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
#            self.weights = tf.get_variable(
#                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
#            self.weights1 = tf.get_variable(
#                shape=[self.dim*2, self.dim*2], initializer=tf.contrib.layers.xavier_initializer(), name='weights1')
#            self.weights2 = tf.get_variable(
#                shape=[self.dim*2, self.dim*2], initializer=tf.contrib.layers.xavier_initializer(), name='weights2')
            self.weights3 = tf.get_variable(
                shape=[self.dim*2, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights3')
#           self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')
#            self.bias1 = tf.get_variable(shape=[self.dim*2], initializer=tf.zeros_initializer(), name='bias1')
#            self.bias2 = tf.get_variable(shape=[self.dim*2], initializer=tf.zeros_initializer(), name='bias2')
            self.bias3 = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias3')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings,neighbor_num):
        #user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])
        self_vectors1 = tf.tile(self_vectors, multiples=[1, neighbor_num, 1])
        self_vectors1 = tf.expand_dims(self_vectors1, axis=1)
        user_relation_scores = tf.concat([self_vectors1,neighbor_relations],axis = -1)
        #user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=2)  # 128,1,1,16 * 128,1,8,16
        user_relation_scores = tf.reshape(user_relation_scores,[-1,self.dim*2])
#        user_relation_scores = tf.nn.relu(tf.matmul(user_relation_scores, self.weights1) + self.bias1)
#        user_relation_scores = tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(user_relation_scores, self.weights2) + self.bias2),self.weights3)+self.bias3)
        user_relation_scores = tf.nn.relu(user_relation_scores)
        user_relation_scores = tf.nn.relu(tf.matmul(tf.nn.relu(user_relation_scores),self.weights3)+self.bias3)
        user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1)  # [-1,16]
        user_relation_scores_normalized = tf.reshape(user_relation_scores_normalized,[self.batch_size,-1,self.dim])
        user_relation_scores_normalized = tf.expand_dims(user_relation_scores_normalized, axis=2)  # [128,-1,1,16]
        # [128,-1,1,16] * [128,-1,8,16] = 128,1,8,16 -->128,1,16
        neighbors_agg = tf.reduce_mean(user_relation_scores_normalized * neighbor_vectors, axis=2)
#        neighbors_agg = (1/neighbor_num) * neighbor_vectors
#        neighbors_agg = tf.reduce_mean(neighbors_agg,axis = 2)
        
        # self_vectors: [batch_size, -1, dim]
        # [-1, dim]
        output1 = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output2 = tf.reshape(self_vectors * neighbors_agg, [-1, self.dim])
        
        
        # (o1 + o2) * w + b
        output = output1+output2
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
#        output = tf.matmul(output, self.weights) + self.bias
        
#         # (o1 * w + b) + (o2 * w1 + b1)
#         output1 = self.act(tf.matmul(output1, self.weights)+ self.bias)      
#         output2 = self.act(tf.matmul(output2, self.weights1)+ self.bias1)
#         output = output1 + output2


        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])
        output = tf.reduce_mean(output, axis=1)
        output = tf.expand_dims(output, axis=1)
        

        return output
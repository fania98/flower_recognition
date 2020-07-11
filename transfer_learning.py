import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets.inception_v2 import inception_v2
from tensorflow.contrib.slim.python.slim.nets.inception_v2 import inception_v2_arg_scope as inception_v2_arg_scope
from generate_batch import generate_stochastic_train_batch, generate_stochastic_valid_batch, generate_stochastic_test_batch


class transferLearningModel:
    def __init__(self):
        self.using_inception = True
        self.train_picture = 21303
        self.height = 224
        self.width = 224
        self.batch_size = 50
        self.label_num = 102
        self.learning_rate = 0.0001
        self.w_len = 1001
        self.fc_len = 256
        self.onehot = self.generatelabel()
        self.max_acc_save = 0
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.is_from_previous = False

        self.img_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.height, self.width, 3], name='input')
        #self.img_input_expand = tf.expand_dims(self.img_input, -1)
        self.img_input_expand = self.img_input
        self.label_id = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None], name='label')
        self.label_onehot = tf.nn.embedding_lookup(self.onehot, self.label_id)
        self.keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')

        if self.using_inception:
            self.feature_vector = self.inception_v2_layer() #特征向量
        else:
            self.feature_vector = self.vgg_layer()
        self.full_connect_out = self.full_connect_layer() #用于生成哈希码及分类
        self.output_class = self.softmax_layer() #分类结果
        self.output_hash = self.hash_layer() #哈希结果

        self.loss = self.get_loss()
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.output_class, 1), tf.argmax(self.label_onehot, 1)), tf.float32), name="my_accuracy")

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        if self.using_inception:
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.loss)
        else:
            self.train_op = self.optimizer.minimize(self.loss)
        self.saver = tf.compat.v1.train.Saver()

        inception_except_logits = slim.get_variables_to_restore(exclude=['Inceptionv2/Logits','Inceptionv2/AuxLogits'])
        self.loaded_weights = slim.assign_from_checkpoint_fn("models/inception_v2.ckpt", inception_except_logits, ignore_missing_vars=True)
        # self.loaded_weights = slim.assign_from_checkpoint_fn("models/inception_v2.ckpt",inception_except_logits)
        self.summ = tf.norm(self.output_class,ord=1)
        self.the_class = tf.argmax(self.output_class,1)


    def generatelabel(self):
        return np.identity(self.label_num)

    def inception_v2_layer(self):
        out_dimension = self.w_len
        with slim.arg_scope(inception_v2_arg_scope()):
            out, _ = inception_v2(inputs=self.img_input_expand, num_classes=out_dimension,
                                           dropout_keep_prob=self.keep_prob,is_training=self.is_training)
        return out

    def vgg_layer(self):
        conv1 = self.conv_layer([3,3,1,64],[64],self.img_input_expand,1) #222*222*64
        conv2 = self.conv_layer([3,3,64,64],[64],conv1,2) ##220*220*64
        pool1 = tf.nn.max_pool(conv2, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID') #110*110*64
        conv2 = self.conv_layer([3, 3, 64, 128], [128], pool1, 3) #108*108*128
        conv4 = self.conv_layer([3,3,128,128],[128], conv2, 4) #106*106*128
        pool2 = tf.nn.max_pool(conv4, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')#53*53*128
        conv5 = self.conv_layer([3,3,128,256],[256],pool2, 5) #51*51*256
        conv6 = self.conv_layer([3,3,256,256],[256], conv5, 6) #49*49*256
        pool3 = tf.nn.max_pool(conv6, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 25*25*256
        conv7 = self.conv_layer([3, 3, 256, 256], [256], pool3, 5)  # 23*23*256
        conv8 = self.conv_layer([3, 3, 256, 256], [256], conv7, 6)  # 21*21*256
        pool4 = tf.nn.max_pool(conv8, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 11*11*256
        conv9 = self.conv_layer([3, 3, 256, 256], [256], pool4, 5)  # 9*9*256
        conv10 = self.conv_layer([3, 3, 256, 256], [256], conv9, 6)  # 7*7*256
        pool5 = tf.nn.max_pool(conv10, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 4*4*256

        flat = tf.reshape(pool5,[-1,4*4*256])
        wfc = self.get_variable([4*4*256,1000],"vgg_fc_w")
        bfc = self.get_variable([1000], "vgg_fc_b")
        out = tf.nn.relu(tf.matmul(flat, wfc)+bfc)
        return out


    def conv_layer(self,wshape,bshape,input,num):
        W = self.get_variable(wshape, "conv_w"+str(num))
        b = self.get_variable(bshape, "conv_b"+str(num))
        h = tf.nn.relu(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='VALID') + b);
        h_dropout = tf.nn.dropout(h, keep_prob=self.keep_prob)
        return h_dropout

    def get_variable(self,shape,name):
        return tf.Variable(initial_value=tf.random.truncated_normal(shape,mean=0,stddev=0.01),name=name)

    def full_connect_layer(self):
        W_fc = tf.Variable(initial_value=tf.random.truncated_normal([self.w_len, self.fc_len], mean=0, stddev=0.01),name="full_connect_w");
        b_fc = tf.Variable(initial_value=tf.random.truncated_normal([self.fc_len], mean=0, stddev=0.01),name="full_connect_b");
        h_fc = tf.tanh(tf.matmul(self.feature_vector, W_fc) + b_fc);
        h_fc1_dropout = tf.nn.dropout(h_fc, self.keep_prob, name="feature_vector");
        return h_fc1_dropout

    def hash_layer(self):
        pass

    def softmax_layer(self):
        W_s = tf.Variable(initial_value=tf.random.truncated_normal([self.fc_len, self.label_num], mean=0, stddev=0.01));
        b_s = tf.Variable(initial_value=tf.random.truncated_normal([self.label_num], mean=0, stddev=0.01));
        h_s= tf.nn.softmax(tf.matmul(self.full_connect_out, W_s) + b_s, name="out_class");
        return h_s

    def get_loss(self):
        #alpha = 0.01
        cross_entropy = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=self.label_onehot, logits=self.output_class)
        #full_connect_output_norm = tf.norm(self.full_connect_out,axis=1)
        #tf.compat.v1.losses.add_loss(tf.reduce_mean(62-full_connect_output_norm)*alpha)
        tf.compat.v1.losses.add_loss(cross_entropy)
        return tf.compat.v1.losses.get_total_loss(add_regularization_losses=True)

    def train(self,):
        max_epoch = 10000
        with tf.compat.v1.Session() as sess:
            if self.is_from_previous:
                module_file = tf.compat.v1.train.latest_checkpoint("models/")
                self.saver.restore(sess, module_file)
                max_accuracy = 0.4
            else:
                sess.run(tf.compat.v1.global_variables_initializer())
                max_accuracy = 0
            self.loaded_weights(sess)
            for i in range(1, max_epoch):
                train_batch = generate_stochastic_train_batch(self.batch_size, True)
                for step,batch in enumerate(train_batch):
                    (img_batch, label_batch) = batch
                    sess.run(self.train_op, feed_dict={self.img_input:img_batch, self.label_id: label_batch, self.keep_prob:0.7, self.is_training:False})

                    if step%100 ==0:
                        valid_accuracy,valid_loss = self.valid(sess)

                        if valid_accuracy > max_accuracy:
                            max_accuracy = valid_accuracy
                            if valid_accuracy > 0.2 and valid_accuracy - self.max_acc_save >= 0.02:
                                self.max_acc_save = max_accuracy
                                best_models = "models/best_models_{}_{}_{:.4f}_{}_{}.ckpt".format(i,step, max_accuracy,self.w_len,self.fc_len)
                                print('------save:{}'.format(best_models))
                                self.max_acc_save = max_accuracy
                                self.saver.save(sess,best_models)
                        print("epoch: {}  step: {}  accuracy: {} max accuracy: {} loss:{}".format(i, step, valid_accuracy, max_accuracy, valid_loss))

    def valid(self, sess):
        valid_batch = generate_stochastic_valid_batch(self.batch_size, True)
        valid_accuracy = []
        valid_loss = []
        for batch in valid_batch:
            (v_img_batch, v_label_batch) = batch
            acc = sess.run(self.accuracy,
                                      feed_dict={self.img_input: v_img_batch, self.label_id: v_label_batch,
                                                 self.keep_prob: 1, self.is_training:False})
            valid_accuracy.append(acc)
            valid_loss.append(sess.run(self.loss, feed_dict={self.img_input: v_img_batch, self.label_id: v_label_batch, self.keep_prob: 1,self.is_training:False}))

        mean_acc = np.array(valid_accuracy, dtype=np.float32).mean()
        mean_loss = np.array(valid_loss, dtype=np.float32).mean()
        return mean_acc,mean_loss

def test():
    model_path = "models/best_models_69_100_0.9431_1001_256.ckpt"
    saver = tf.train.import_meta_graph(model_path+".meta")  # 加载图结构
    graph = tf.get_default_graph()
    # tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
    # with open("name.txt",'w+')as f:
    #     for n in tensor_name_list:
    #         f.write(n+"\n")
    img_input = graph.get_tensor_by_name("input:0")
    label_id = graph.get_tensor_by_name("label:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    accuracy = graph.get_tensor_by_name("my_accuracy:0")
    is_train = graph.get_tensor_by_name("is_training:0")
    test_batch = generate_stochastic_test_batch(100, False)
    test_accuracy = []
    with tf.Session() as sess:
        #ckpt = tf.train.get_checkpoint_state('models/')
        saver.restore(sess,"models/best_models_69_100_0.9431_1001_256.ckpt")
        for batch in test_batch:
            (v_img_batch, v_label_batch) = batch
            acc = sess.run(accuracy, feed_dict={img_input: v_img_batch, label_id: v_label_batch,
                                                 keep_prob: 1, is_train: False})
            test_accuracy.append(acc)
            print(acc)
            #valid_loss.append(sess.run(self.loss, feed_dict={self.img_input: v_img_batch, self.label_id: v_label_batch, self.keep_prob: 1}))

        mean_acc = np.array(test_accuracy, dtype=np.float32).mean()
       # mean_loss = np.array(valid_loss, dtype=np.float32).mean()
    print(mean_acc)

# t = transferLearningModel()
# t.train()
test()
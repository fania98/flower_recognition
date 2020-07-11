import tensorflow as tf
from generate_batch import generate_batch
from db_connection import get_connection
from image_preprocessing import get_name_list
import re
#几点改进：1.hash code 更短一些
#         2. hash code每个元素更接近于1或-1
def generate_feature_vector_and_hash(img_batch):
    model_path = "models/best_models_27_0_0.7128_1024_512.ckpt"
    saver = tf.train.import_meta_graph(model_path+".meta")  # 加载图结构
    graph = tf.get_default_graph()
    img_input = graph.get_tensor_by_name("input:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    hash_layer = graph.get_tensor_by_name("Tanh:0")
    is_train = graph.get_tensor_by_name("is_training:0")
    feature_layer = graph.get_tensor_by_name("InceptionV2/Predictions/Reshape_1:0")
    with tf.Session() as sess:
        #ckpt = tf.train.get_checkpoint_state('models/')
        saver.restore(sess, "models/best_models_27_0_0.7128_1024_512.ckpt")
        #for batch in valid_batch:
        feature_vector = sess.run(feature_layer, feed_dict={img_input: img_batch, keep_prob: 1, is_train: False})
        hash_code = tohash(sess.run(hash_layer, feed_dict={img_input: img_batch, keep_prob: 1, is_train: False}))
        return feature_vector, hash_code

def tohash(inputs):
    output1 = 0
    outputs = []
    for input in inputs:
        for i in range(0,512):
            if input[i]>0:
                code = 1
            else:
                code = 0
            output1 =output1*2+code
        outputs.append(output1)
        output1 = 0
    return outputs

def save_to_db():
    size = 100
    img_batches = generate_batch(size)
    for batch in img_batches:
        (img_input, img_labels, img_urls) = batch
        features, hashcodes = generate_feature_vector_and_hash(img_input)
        conn1 =get_connection()
        cursor1 = conn1.cursor()
        name_list = get_name_list()
        for i in range(0, len(img_urls)):
            feature = features[i].tostring()
            hashcode = hashcodes[i]
            label_id = int(img_labels[i])
            img_id = re.search(r"image_(.*).jpg", img_urls[i]).group(1)
            cursor1.execute("INSERT INTO img_info ( img_id, class, class_name, img_url) VALUES(%s, %s, %s, %s);",
                            [img_id, label_id, name_list[label_id].encode("utf-8"), img_urls[i]])
            cursor1.execute("INSERT INTO hash_code ( img_id, hash_code) VALUES(%s, %s);",
                            [img_id, str(hashcode)])
            cursor1.execute("INSERT INTO feature ( img_id, feature) VALUES(%s, %s);",
                            [img_id, feature])
            conn1.commit()

def save_to_db_hash_code():
    size = 100
    img_batches = generate_batch(size)
    for batch in img_batches:
        (img_input, img_labels, img_urls) = batch
        features, hashcodes = generate_feature_vector_and_hash(img_input)
        conn1 = get_connection()
        cursor1 = conn1.cursor()
        for i in range(0, len(img_urls)):
            hashcode = hashcodes[i]
            img_id = re.search(r"image_(.*).jpg", img_urls[i]).group(1)
            cursor1.execute("INSERT INTO hash_code_full ( img_id, hash_code) VALUES(%s, %s);",
                            [img_id, str(hashcode)])
            conn1.commit()

if __name__ == "__main__":
    save_to_db_hash_code()
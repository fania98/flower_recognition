import os
import re

import cv2
import numpy as np
import pickle
from db_connection import get_connection
import random
from feature_hash import generate_feature_vector_and_hash
import heapq
from find_pics import calculate_same_bit
import csv

def open_new_file(num):
    f = open("test_transfer_img_data_"+str(num)+".pkl", 'wb')
    return f
def img_preprocess():
    data_record = []
    i = 0
    for dir in os.listdir("transfer/test"):
        for img in os.listdir(os.path.join("transfer/test", dir)):
            if (re.search("jpg",img) is None):
                continue
            img_gray = cv2.imread(os.path.join("transfer/test", dir, img), 0)
            #print(os.path.join("transfer/train", dir, img))
            #img_gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
            size = (224, 224)
            if img_gray is None:
                print(os.path.join("transfer/test", dir, img))
            img_resize = cv2.resize(img_gray, size, interpolation=cv2.INTER_AREA)
            #img_data = img_resize.astype(np.uint8)
            if i % 2000 != 0 or i == 0:
                data_record.append({"data": img_resize, "label": dir})
                i += 1
            else:
                i += 1
                f = open_new_file(i // 2000)
                pickle.dump(data_record, f)
                f.close()
                data_record = []
                data_record.append({"data": img_resize, "label": dir})

    f = open_new_file((i // 2000) + 1)
    pickle.dump(data_record, f)
    f.close()

def generate_stochastic_train_batch(size, israndom=False):
    cur_train_batch = np.zeros((size, 224, 224))
    cur_label = np.zeros(size, dtype=np.int8)
    img_urls = []
    cur_num = 0
    img_num = 0
    img_data = []
    for i in range(1, 4):
        file_name = "transfer_img_data_" + str(i) + ".pkl"
        with open(file_name, 'rb') as f:
            img_data += pickle.load(f)
    if israndom:
        random.shuffle(img_data)

    for img in img_data:
        cur_train_batch[cur_num] = img['data'] / 256
        cur_label[cur_num] = img['label']
        img_num += 1
        cur_num += 1
        if cur_num == size:
            yield (cur_train_batch, cur_label)
            cur_num = 0
            cur_train_batch = np.zeros((size, 224, 224))
            cur_label = np.zeros(size, dtype=np.int8)
            #img_urls.append()
    yield (cur_train_batch, cur_label)

def save_to_db():
    size = 100
    img_batches = generate_stochastic_train_batch(size)
    img_id = 0
    batch_id = 0
    for batch in img_batches:
        (img_input, img_labels) = batch
        if (batch_id<=37):
            batch_id += 1
            img_id += 100
            continue
        features, hashcodes = generate_feature_vector_and_hash(img_input)
        conn1 =get_connection()
        cursor1 = conn1.cursor()
        for i in range(0, len(img_labels)):
            feature = features[i].tostring()
            hashcode = hashcodes[i]
            label_id = int(img_labels[i])
            print(img_id)
            cursor1.execute("INSERT INTO transfer_img_info ( img_id, class) VALUES(%s, %s);",
                            [img_id, label_id])
            cursor1.execute("INSERT INTO transfer_hash_code ( img_id, hash_code) VALUES(%s, %s);",
                            [img_id, str(hashcode)])
            img_id +=1
            conn1.commit()

def generate_stochastic_test_batch(size, israndom=False):
    cur_train_batch = np.zeros((size, 224, 224))
    cur_label = np.zeros(size,dtype=np.int8)
    cur_num = 0
    file_name = "test_transfer_img_data_1.pkl"
    with open(file_name, 'rb') as f:
        img_data = pickle.load(f)
    if israndom:
        random.shuffle(img_data)
    for img in img_data:
        # print(img)
        cur_train_batch[cur_num]=img['data']/256
        cur_label[cur_num] = img['label']
        cur_num += 1
        if cur_num==size:
            # print(cur_train_batch)
            # print(cur_label)
            yield (cur_train_batch,cur_label)
            cur_num = 0
            cur_train_batch = np.zeros((size, 224, 224))
            cur_label = np.zeros(size, dtype=np.int8)
    yield (cur_train_batch, cur_label)

def transfer_test():
    test_batch = generate_stochastic_test_batch(100, True)
    num = 8
    test_num = 0
    correct = 0
    class_total = [0] * 102
    class_right = [0] * 102
    cur_num = 0
    for batch in test_batch:
        img_input, label = batch
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("select img_id, hash_code from transfer_hash_code")
        all_pics = cursor.fetchall()
        cursor = conn.cursor()
        features, hash_codes = generate_feature_vector_and_hash(img_input)
        for i, hash_code in enumerate(hash_codes):
            if cur_num<224:
                cur_num+=1
            else:
                break
            test_num += 1
            feature = features[i]
            topkids = []
            for data in all_pics:
                id = data[0]
                code = int(data[1])
                same_num = calculate_same_bit(hash_code, code)
                if len(topkids) < num or heapq.nsmallest(1, topkids)[0][0] == same_num:
                    heapq.heappush(topkids, (same_num, id))
                elif heapq.nsmallest(1, topkids)[0][0] < same_num:
                    heapq.heapreplace(topkids, (same_num, id))
            similar_ids = heapq.nlargest(num, topkids)
            result_img_id = 0
            result_classes = dict()
            for pic_id in similar_ids:
                sql = "SELECT class from transfer_img_info where img_id={}".format(pic_id[1])
                cursor.execute(sql)
                item_class = int(cursor.fetchone()[0])
                # candidate_feature = cursor.fetchone()[0]
                # candidate_feature = np.fromstring(candidate_feature, np.float32).reshape(1024)
                if item_class in result_classes.keys():
                    result_classes[item_class] += 1
                else:
                    result_classes[item_class] = 1

            max_times = 0
            result_class = -1
            for key, value in result_classes.items():
                if value > max_times:
                    result_class = key
                    max_times = value
            print(result_classes)
            print(label[i])
            print(result_class)

            if result_class == label[i]:
                correct += 1
                class_right[result_class] += 1
            class_total[label[i]] += 1
            print(correct / test_num)

    print(correct / test_num)
    with open("classify_result.csv", "w") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(["total", "right"])
        for r, t in zip(class_right, class_total):
            f_csv.writerow([t, r])

    print(class_right)
    print(class_total)

# img_preprocess()
# save_to_db()
transfer_test()
#img_preprocess()
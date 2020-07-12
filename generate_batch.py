import os
import json
import random
import numpy as np
import pickle
import cv2
import re
from db_connection import get_connection
def generate_name_label(data_type):
    img_names = []
    for d in os.listdir("processed_data/"+data_type):
        label = d
        for file in os.listdir(os.path.join("processed_data/"+data_type,d)):
            f_name = file
            img_names.append({"f_name": f_name, "label": label})
    with open(data_type+".json", 'w', encoding='utf-8') as f:
        json.dump(img_names, f)


def generate_stochastic_train_batch(size, israndom=False):
    cur_train_batch = np.zeros((size, 224, 224,3))
    cur_label = np.zeros(size,dtype=np.int8)
    cur_num = 0
    img_num = 0
    img_data = []
    for i in range(1, 12):
        file_name = "train_img_data_"+str(i)+".pkl"
        with open(file_name,'rb') as f:
            img_data+=pickle.load(f)
    if israndom:
        random.shuffle(img_data)

    for img in img_data:
        cur_train_batch[cur_num]=img['data']/256
        cur_label[cur_num] = img['label']
        img_num+=1
        cur_num += 1
        if cur_num==size:
            yield (cur_train_batch,cur_label)
            cur_num = 0
            cur_train_batch = np.zeros((size, 224, 224,3))
            cur_label = np.zeros(size, dtype=np.int8)
    yield (cur_train_batch, cur_label)


def generate_stochastic_valid_batch(size, israndom=False):
    cur_train_batch = np.zeros((size, 224, 224, 3))
    cur_label = np.zeros(size,dtype=np.int8)
    cur_num = 0
    file_name = "valid_img_data_1.pkl"
    with open(file_name, 'rb') as f:
        img_data = pickle.load(f)
    if israndom:
        random.shuffle(img_data)
    for img in img_data:
        cur_train_batch[cur_num]=img['data']/256
        cur_label[cur_num] = img['label']
        cur_num += 1
        if cur_num==size:
            yield (cur_train_batch,cur_label)
            cur_num = 0
            cur_train_batch = np.zeros((size, 224, 224, 3))
            cur_label = np.zeros(size, dtype=np.int8)
    yield (cur_train_batch, cur_label)

def generate_stochastic_test_batch(size, israndom=False):
    cur_train_batch = np.zeros((size, 224, 224, 3))
    cur_label = np.zeros(size,dtype=np.int8)
    cur_num = 0
    file_name = "test_img_data_1.pkl"
    with open(file_name, 'rb') as f:
        img_data = pickle.load(f)
    if israndom:
        random.shuffle(img_data)
    for img in img_data:
        cur_train_batch[cur_num]=img['data']/256
        cur_label[cur_num] = img['label']
        cur_num += 1
        if cur_num==size:
            yield (cur_train_batch,cur_label)
            cur_num = 0
            cur_train_batch = np.zeros((size, 224, 224, 3))
            cur_label = np.zeros(size, dtype=np.int8)
    yield (cur_train_batch, cur_label)

def generate_batch(size):
    types=["train","valid"]
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT img_id from hash_code_full;")
    saved = cursor.fetchall()
    saved = [s[0] for s in saved]

    cur_num = 0
    img_urls = []
    cur_train_batch = np.zeros((size, 224, 224, 3))
    cur_label = np.zeros(size, dtype=np.int8)
    processed_id = set()
    for data_type in types:
        for dir in os.listdir("processed_data/" + data_type):
            for filename in os.listdir(os.path.join("processed_data", data_type, dir)):
                img_id = re.search(r"\d_\d_image_(.*).jpg",filename).group(1)
                img_original_name = re.search(r"\d_\d_(.*)",filename).group(1)
                if int(img_id) in processed_id or int(img_id) in saved:
                    continue
                processed_id.add(int(img_id))
                img = cv2.imread(os.path.join("static", "jpg", img_original_name))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_size = (224, 224)
                img_resize = cv2.resize(img_gray, img_size, interpolation=cv2.INTER_AREA)
                img_data = img_resize/256
                cur_train_batch[cur_num] = img_data
                if cur_num == 99:
                    print(99)
                cur_label[cur_num] = dir
                img_urls.append(os.path.join("static", "jpg", img_original_name))
                cur_num += 1
                if cur_num == size:
                    yield (cur_train_batch, cur_label, img_urls)
                    cur_num = 0
                    cur_train_batch = np.zeros((size, 224, 224, 3))
                    cur_label = np.zeros(size, dtype=np.int8)
                    img_urls = []
    yield (cur_train_batch, cur_label, img_urls)

#
# a = generate_stochastic_train_batch(10)
# next(a)
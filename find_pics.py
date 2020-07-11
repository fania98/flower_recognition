from feature_hash import generate_feature_vector_and_hash
import cv2
import pymysql
import heapq
import numpy as np
from db_connection import get_connection

#修改点：model新加入了is_training, inception_v2内加入了keep_prob 使其稳定

def get_picture(url):
    #url = "processed_data/test/92/1_2_image_06038.jpg"
    img = cv2.imread(url,0)
    img = img/256
    img = img[np.newaxis,:,:]
    return img

def calculate_same_bit(code1, code2):
    num=0
    xor = code1^code2
    #print(bin(xor))
    if (xor==0):
        return 512
    while xor != 0:
        if xor%2 == 0:
            num += 1
        xor = xor//2
    return num

def find_picture_ids(hashcode, img_num=10):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("select img_id, hash_code from hash_code_full")
    result = cursor.fetchall()
    topkids = []
    for data in result:
        id = data[0]
        code = int(data[1])
        same_num = calculate_same_bit(hashcode, code)
        if len(topkids)<img_num or heapq.nsmallest(1,topkids)[0][0]==same_num:
            heapq.heappush(topkids,(same_num,id))
        elif heapq.nsmallest(1,topkids)[0][0]<same_num:
            heapq.heapreplace(topkids, (same_num,id))
    print(heapq.nlargest(img_num,topkids))
    return heapq.nlargest(img_num,topkids)


#find_picture_ids()
# img = get_picture("processed_data/test/0/1_3_image_06741.jpg")
# feature,hashcode = generate_feature_vector_and_hash(img)
# hashcode1 = hashcode[0]//2
# print(bin(hashcode1))
#
# img2 = get_picture("processed_data/test/3/1_2_image_05659.jpg")
# feature2,hashcode2 = generate_feature_vector_and_hash(img2)
# hashcode2 = hashcode2[0]//2
# print(bin(hashcode2))
# print(calculate_same_bit(hashcode1,hashcode2))


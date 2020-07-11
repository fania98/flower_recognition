import cv2
import numpy as np
from PIL import Image
import io
from find_pics import find_picture_ids
from feature_hash import generate_feature_vector_and_hash
import pymysql
import re
import numpy as np
from db_connection import get_connection

# def process(img):
#     num=10
#     img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     size = (224, 224)
#     img_resize = cv2.resize(img_gray, size, interpolation=cv2.INTER_AREA)
#     img_resize = img_resize/256
#     img_resize =img_resize[np.newaxis,:,:]
#     feature, hash_code = generate_feature_vector_and_hash(img_resize)
#     hash_code = hash_code[0]
#     feature = feature[0]
#     similar_ids = find_picture_ids(hash_code, num)
#     conn = pymysql.connect(
#         host="49.234.209.10",
#         user="root", password="980512",
#         database="flower", charset="utf8")
#     cursor = conn.cursor()
#     for pic_id in similar_ids:
#         sql = "SELECT img_url feature_vector from img_hash_code where id={}".format(pic_id[1])
#         cursor.execute(sql)
#         dd = cursor.fetchone()
#         print(dd)
#     print("SELECT img_url from img_hash_code where id = {}".format(similar_ids[0][1]))
#     cursor.execute("SELECT img_url,class_name from img_hash_code where id = {}".format(similar_ids[0][1]))
#     img_info = cursor.fetchone()
#     img_path = img_info[0]
#     class_name = img_info[1]
#     print(img_path)
#     img_name = re.search(r"\d_\d_(.*)",img_path).group(1)
#     img_path = "jpg/{}".format(img_name)
#     return img_path,class_name


def process(img):
    num=10
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = (224, 224)
    img_resize = cv2.resize(img_gray, size, interpolation=cv2.INTER_AREA)
    img_resize = img_resize/256
    img_resize =img_resize[np.newaxis,:,:]
    feature, hash_code = generate_feature_vector_and_hash(img_resize)
    hash_code = hash_code[0]
    similar_ids = find_picture_ids(hash_code, num)
    conn = get_connection()
    cursor = conn.cursor()
    result_img_id = 0
    result_classes = dict()
    for pic_id in similar_ids:
        sql = "SELECT class from img_info where img_id={}".format(pic_id[1])
        cursor.execute(sql)
        item_class = int(cursor.fetchone()[0])
        if item_class in result_classes.keys():
            result_classes[item_class].append(pic_id)
        else:
            result_classes[item_class] = [pic_id]

    max_times = 0
    result_class = -1
    for key, value in result_classes.items():
        if len(value) > max_times:
            result_class = key
            max_times = len(value)
    print("SELECT img_url,class_name from img_info where img_id = {}".format(result_classes[result_class][0][1]))
    cursor.execute("SELECT img_url,class_name from img_info where img_id = {}".format(result_classes[result_class][0][1]))
    img_info = cursor.fetchone()
    img_path = img_info[0].replace("\\", "/")
    class_name = img_info[1]
    # img_name = re.search(r"static(.*\.jpg)",img_path).group(1)
    # img_path = "jpg/{}".format(img_name)
    return img_path, class_name

# img = cv2.imread("static/jpg/image_00001.jpg")
# textprocess(img)
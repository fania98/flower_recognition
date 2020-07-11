from generate_batch import generate_stochastic_test_batch
from feature_hash import generate_feature_vector_and_hash
from find_pics import find_picture_ids, calculate_same_bit
import pymysql
import numpy as np
import csv
import heapq

test_batch = generate_stochastic_test_batch(100, True)
num = 15
test_num = 0
correct = 0
class_total = [0]*102
class_right = [0]*102
for batch in test_batch:
    img_input,label = batch
    conn = pymysql.connect(
        host="49.234.209.10",
        user="root", password="980512",
        database="flower", charset="utf8")
    cursor = conn.cursor()
    cursor.execute("select img_id, hash_code from hash_code_full")
    all_pics = cursor.fetchall()
    cursor = conn.cursor()
    features, hash_codes = generate_feature_vector_and_hash(img_input)
    for i, hash_code in enumerate(hash_codes):
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
        similar_ids = heapq.nlargest(num,topkids)
        result_img_id = 0
        result_classes = dict()
        for pic_id in similar_ids:
            sql = "SELECT class from img_info where img_id={}".format(pic_id[1])
            cursor.execute(sql)
            item_class = int(cursor.fetchone()[0])
            #candidate_feature = cursor.fetchone()[0]
            #candidate_feature = np.fromstring(candidate_feature, np.float32).reshape(1024)
            if item_class in result_classes.keys():
                result_classes[item_class] += 1
            else:
                result_classes[item_class] = 1

        max_times = 0
        result_class = -1
        for key,value in result_classes.items():
            if value>max_times:
                result_class = key
                max_times = value
        print(result_classes)
        print(label[i])
        print(result_class)

        if result_class == label[i]:
            correct += 1
            class_right[result_class] +=1
        class_total[label[i]] += 1
        print(correct/test_num)

print(correct/test_num)
with open("classify_result.csv","w") as f:
    f_csv = csv.writer(f)
    f_csv.writerow(["total","right"])
    for r,t in zip(class_right,class_total):
        f_csv.writerow([t,r])

print(class_right)
print(class_total)


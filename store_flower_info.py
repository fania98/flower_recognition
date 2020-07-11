from db_connection import get_connection
import json
import os
conn = get_connection()
cursor = conn.cursor()

def generate_sqls():
    with open("../flower_info.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

        for d in data:
            if d["class"]>=71 and d['class']<=90:
                img_directory = "static/raw_data/valid/"+str(d["class"])
                img_name = os.listdir(img_directory)[0]
                img_url = img_directory+"/"+img_name
                sql = "INSERT INTO flower_info(class, name, img_url, description, distribution, more_info) " \
                      "VALUES({},'{}','{}','{}','{}','{}')".format(d["class"],d["name"],img_url,d["description"],d["distribution"], d["more_info"])
                print(sql+";")
                cursor.execute(sql)
                conn.commit()

def modify_address():
    for i in range(0,11):
        cursor.execute("SELECT class, img_url from flower_info where class="+str(i));
        result = cursor.fetchone()
        img_url = "static/"+result[1].replace("\\", "/")
        cursor.execute("UPDATE flower_info SET img_url = '{}' WHERE class = {}".format(img_url, result[0]))
        conn.commit()

generate_sqls()

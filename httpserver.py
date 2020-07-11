from flask import Flask, request,jsonify,make_response
from process_img import process
import json
from db_connection import get_connection

app = Flask(__name__, static_url_path='')

@app.route("/findPic",methods=["POST","GET"])
def find_pic():
    if request.method=="POST":
        img = request.files["img"].stream.read()
        img_result = process(img)
        result = {"img_path":img_result[0],"class_name":img_result[1]}
        print(result)
        return json.dumps(result)

@app.route("/flower_info",methods = ["GET"])
def flower_info():
    class_name = request.values["name"]
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM flower_info WHERE name='{}'".format(class_name))
    result = cursor.fetchone()
    return {"title":result[1], "description":result[3], "distribution":result[4], "more_info":result[5], "img_url":result[2].replace("\\","/")}

if __name__=="__main__":
    app.run(debug=True)
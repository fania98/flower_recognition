import random
import re
import os
import cv2
import shutil
import json
import numpy as np
import pickle
import matplotlib.pyplot as plt


def get_name_list():
    """
    get the name of all flower classes
    """
    flower_names = []
    with open("flower_name.txt",'r',encoding='utf-8') as f:
        for line in f:
            match = re.match("\d+-(.+)",line)
            name = match.group(1)
            flower_names.append(name)
    return flower_names


def process_image_rotate(type):
    """
        read raw images of the specified "type", resize the images, make images more by fliping and rotating images
        :param type: "train" "valid" or "test"
        """
    for dir in os.listdir("static/raw_data/"+type):
        if not os.path.exists("processed_data_1/"+type+"/"+dir):
            os.mkdir("processed_data_1/"+type+"/"+dir)
        for img_name in os.listdir("static/raw_data/"+type+"/"+dir):
            img = cv2.imread("static/raw_data/"+type+"/"+dir+"/"+img_name)
            size = (224, 224)
            img_resize = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            img_flip1 = cv2.flip(img_resize, 1)
            img_flip2 = cv2.flip(img_resize, 0)
            img_rotate_1 = rotate(img_resize,45,True)
            img_rotate_1 = cv2.resize(img_rotate_1, size, interpolation=cv2.INTER_AREA)
            img_rotate_2 = rotate(img_resize,90,True)
            img_rotate_2 = cv2.resize(img_rotate_2, size, interpolation=cv2.INTER_AREA)
            img_rotate_3 = rotate(img_resize,315, True)
            img_rotate_3 = cv2.resize(img_rotate_3, size, interpolation=cv2.INTER_AREA)

            print("processed_data_1/"+type+"/"+dir+"/"+img_name)
            cv2.imwrite("processed_data/"+type+"/"+dir + "/1_"+img_name, img_resize)
            cv2.imwrite("processed_data/" + type + "/" + dir + "/2_" + img_name, img_flip1)
            cv2.imwrite("processed_data/" + type + "/" + dir + "/3_" + img_name, img_flip2)
            cv2.imwrite("processed_data/" + type + "/" + dir + "/4_" + img_name, img_rotate_1)
            cv2.imwrite("processed_data/" + type + "/" + dir + "/5_" + img_name, img_rotate_2)
            cv2.imwrite("processed_data/" + type + "/" + dir + "/6_" + img_name, img_rotate_3)


def process_image(type):
    """
    just resize the image, not rotating or fliping
    :param type: "train" "valid" or "test"
    :return:
    """
    for dir in os.listdir("static/raw_data/" + type):
        if not os.path.exists("processed_data/" + type + "/" + dir):
            os.mkdir("processed_data/" + type + "/" + dir)
        for img_name in os.listdir("static/raw_data/" + type + "/" + dir):
            img = cv2.imread("static/raw_data/" + type + "/" + dir + "/" + img_name)
            size = (224, 224)
            img_resize = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            cv2.imwrite("processed_data/"+type+"/"+dir + "/1_"+img_name, img_resize)


def rotate(image, angle, scale=1.0):
    crop_image = lambda img, x0, y0, w, h: img[x0:x0 + w,y0: y0 + h]

    height, width = image.shape[:2]#获取图像的高和宽
    center = (width / 2, height / 2) #取图像的中点

    M = cv2.getRotationMatrix2D(center, angle, scale)#获得图像绕着某一点的旋转矩阵
    rotated = cv2.warpAffine(image, M, (height, width))
    angle_crop = angle % 180
    if angle > 90:
        angle_crop = 180 - angle_crop
    # 转化角度为弧度
    theta = angle_crop * np.pi / 180
    # 计算高宽比
    hw_ratio = float(height) / float(width)
    # 计算裁剪边长系数的分子项
    tan_theta = np.tan(theta)
    numerator = np.cos(theta) + np.sin(theta) * np.tan(theta)

    # 计算分母中和高宽比相关的项
    r = hw_ratio if height > width else 1 / hw_ratio
    # 计算分母项
    denominator = r * tan_theta + 1
    # 最终的边长系数
    crop_mult = numerator / denominator

    # 得到裁剪区域
    w_crop = int(crop_mult * width)
    h_crop = int(crop_mult * height)
    x0 = int((width - w_crop) / 2)
    y0 = int((height - h_crop) / 2)

    rotated = crop_image(rotated, x0, y0, w_crop, h_crop)

    return rotated


def undersampling():
    """
    undersampling of the training examples
    :return:
    """
    for dir in os.listdir("processed_data/train"):
        pic_list = os.listdir("processed_data/train/"+dir)
        random.shuffle(pic_list)
        if len(pic_list) > 500:
            for i in range(len(pic_list)//2):
                os.remove("processed_data_1/train/"+dir+"/"+pic_list[i])


# def increase_train_data():
#     for dir in os.listdir("static/raw_data/train"):
#         test_imgs = os.listdir("static/raw_data/test/"+dir)
#         test_num = len(test_imgs)
#         for i in range(1, test_num//2):
#             shutil.move("static/raw_data/test/"+dir+"/"+test_imgs[i-1], "static/raw_data/train/"+dir+"/"+test_imgs[i-1])
#
#         valid_imgs = os.listdir("static/raw_data/valid/" + dir)
#         valid_num = len(valid_imgs)
#         for i in range(1, valid_num // 2):
#             shutil.move("static/raw_data/valid/" + dir+"/"+valid_imgs[i - 1], "static/raw_data/train/" + dir+ "/"+valid_imgs[i-1])

def open_new_file(data_type,num):
    f = open(data_type+"_img_data_"+str(num)+".pkl", 'wb')
    return f

def save_img_datas(data_type):
    """
    read in the processed images and save them in a record file(pickle)
    :param data_type: "train","valid" or "test"
    :return:
    """
    data_record=[]
    i = 0
    for dir in os.listdir("processed_data/"+data_type):
        for img in os.listdir(os.path.join("processed_data", data_type, dir)):
            img_data = cv2.imread(os.path.join("processed_data", data_type, dir, img))
            img_data = img_data.astype(np.uint8)
            if i % 2000 != 0 or i == 0:
                data_record.append({"data": img_data, "label": dir})
                i += 1
            else:
                i += 1
                f = open_new_file(data_type, i //2000)
                pickle.dump(data_record, f)
                f.close()
                data_record = []
                data_record.append({"data": img_data, "label": dir})
                
    f = open_new_file(data_type, (i // 2000)+1)
    pickle.dump(data_record, f)
    f.close()

def get_picture_num():
    img_num=0
    for dir in os.listdir("processed_data/train"):
         img_num += len(os.listdir("processed_data/train/"+str(dir)))
    print(img_num)

# def zoom_image(type):
#     for dir in os.listdir("processed_data_past/"+type):
#         if not os.path.exists("processed_data/"+type+"/"+dir):
#             os.mkdir("processed_data/"+type+"/"+dir)
#         for img_name in os.listdir("processed_data_past/"+type+"/"+dir):
#             img_gray = cv2.imread("processed_data_past/"+type+"/"+dir+"/"+img_name,0)
#             size = (224, 224)
#             img_resize = cv2.resize(img_gray, size, interpolation=cv2.INTER_AREA)
#             cv2.imwrite("processed_data/"+type+"/"+dir + "/1_"+img_name, img_resize)

# def get_picture_num_in_files():
#     img_num=0
#     for i in range(1,12):
#         with open("train_img_data_"+str(i)+".pkl",'rb') as f:
#             d= pickle.load(f)
#         img_num+=len(d)
#     print(img_num)


def get_train_num_of_class():
    train_num = []
    for dir in os.listdir("processed_data/train"):
        train_num.append(len(os.listdir("processed_data/train/"+dir)))
    print(train_num)
    plt.bar(range(len(train_num)), train_num)
    plt.show()


if __name__ == "__main__":
    #the entire img preprocessing operation
    process_image_rotate("train")
    process_image("valid")
    process_image("train")
    save_img_datas("train")
    save_img_datas("valid")
    save_img_datas("test")

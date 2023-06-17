import PIL.Image as Image
import numpy as np
import tifffile as tf
import cv2
import os
import skimage.io as io
IMAGES_FORMAT = ['.tif']  # 图片格式



IMAGES_PATH = r'../data/Vaihingen/test/'  # 图片集地址
IMAGE_SIZE = 128  # 每张小图片的大小
IMAGE_ROW = 20  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 15  # 图片间隔，也就是合并成一张图后，一共有几列
ID = '30'
IMAGE_SAVE_PATH = r'./test.tif'  # 图片转换后的地址

# 获取图片集地址下的所有图片名称
# filelist = os.listdir(IMAGES_PATH)
# image_names = []
# for file in filelist:
#     i=int(file[26:-4])
#     image_names.append(file)
# #image_names.sort(key=lambda x:int(x[49:-4]))
# image_names.sort(key=lambda x:int(x[26:-4]))

#path = os.path.expanduser("./")
# for f in os.listdir(path):
#     if 'area' + ID in f:
#         img = tf.imread(f)
#         width = img.shape[1]
#         height = img.shape[0]
#         break


cut_size={5:(1887,2557),15:(1919,2565),21:(1903,2546),30:(1934,2563),40:(6000,6000)}  #40是potsdam
def save_all(path, all_img,id=30):
    (width,height)=cut_size[id]
    IMAGE_ROW=height//IMAGE_SIZE +1
    IMAGE_COLUMN=width//IMAGE_SIZE +1
    to_image = np.zeros([height, width, 3]).astype(np.uint8) # 创建一个新图

    x_p = 0
    y_p = 0
    # tif
    for y in range(0, IMAGE_COLUMN):
        for x in range(0, IMAGE_ROW):
            if x == IMAGE_ROW - 1 and y == IMAGE_COLUMN - 1:
                it = IMAGE_COLUMN * x + y
                #img = io.imread(IMAGES_PATH + image_names[it])
                img=all_img[it]
                img_h, img_w, _ = img.shape
                x_lap = IMAGE_SIZE - (height - IMAGE_SIZE * (IMAGE_ROW - 1))
                y_lap = IMAGE_SIZE - (width - IMAGE_SIZE * (IMAGE_COLUMN - 1))
                to_image[x_p - x_lap:x_p - x_lap + img_h, y_p - y_lap:y_p - y_lap + img_w] = np.copy(img)

            elif x == IMAGE_ROW - 1:
                it = IMAGE_COLUMN * x + y
                #img = io.imread(IMAGES_PATH + image_names[it])
                img=all_img[it]
                img_h, img_w, _ = img.shape
                x_lap = IMAGE_SIZE - (height - IMAGE_SIZE * (IMAGE_ROW - 1))
                to_image[x_p - x_lap:x_p - x_lap + img_h, y_p:y_p + img_w] = np.copy(img)

            elif y == IMAGE_COLUMN - 1:
                it = IMAGE_COLUMN * x + y
                #img =io.imread(IMAGES_PATH + image_names[it])
                img=all_img[it]
                img_h, img_w, _ = img.shape
                y_lap = IMAGE_SIZE - (width - IMAGE_SIZE * (IMAGE_COLUMN - 1))
                to_image[x_p:x_p + img_h, y_p - y_lap:y_p - y_lap + img_w] = np.copy(img)

            else:
                it = IMAGE_COLUMN * x + y
                #img = tf.imread(IMAGES_PATH + image_names[it])
                img=all_img[it]
                img_h, img_w, _ = img.shape
                to_image[x_p:x_p + img_h, y_p:y_p + img_w] = np.copy(img)
            x_p = x_p + img_h
        x_p = 0
        y_p = y_p + img_w
    to_image = to_image[:, :, (2, 1, 0)]
    cv2.imwrite(path, to_image)

#save_all('./test.png',[])
# for y in range(0, IMAGE_COLUMN):
#     for x in range(0, IMAGE_ROW):
#         if x == IMAGE_ROW - 1 and y == IMAGE_COLUMN - 1:
#             it = IMAGE_COLUMN * x + y
#             img = cv2.imread(IMAGES_PATH + image_names[it])
#             img_h, img_w, _ = img.shape
#             x_lap = IMAGE_SIZE - (height - IMAGE_SIZE * (IMAGE_ROW - 1))
#             y_lap = IMAGE_SIZE - (width - IMAGE_SIZE * (IMAGE_COLUMN - 1))
#             to_image[x_p - x_lap:x_p - x_lap + img_h, y_p - y_lap:y_p - y_lap + img_w] = np.copy(img)
#
#         elif x == IMAGE_ROW - 1:
#             it = IMAGE_COLUMN * x + y
#             img = cv2.imread(IMAGES_PATH + image_names[it])
#             img_h, img_w, _ = img.shape
#             x_lap = IMAGE_SIZE - (height - IMAGE_SIZE * (IMAGE_ROW - 1))
#             to_image[x_p - x_lap:x_p - x_lap + img_h, y_p:y_p + img_w] = np.copy(img)
#
#         elif y == IMAGE_COLUMN - 1:
#             it = IMAGE_COLUMN * x + y
#             img = cv2.imread(IMAGES_PATH + image_names[it])
#             img_h, img_w, _ = img.shape
#             y_lap = IMAGE_SIZE - (width - IMAGE_SIZE * (IMAGE_COLUMN - 1))
#             to_image[x_p:x_p + img_h, y_p - y_lap:y_p - y_lap + img_w] = np.copy(img)
#
#         else:
#             it = IMAGE_COLUMN * x + y
#             img = cv2.imread(IMAGES_PATH + image_names[it])
#             img_h, img_w, _ = img.shape
#             to_image[x_p:x_p + img_h, y_p:y_p + img_w] = np.copy(img)
#         x_p = x_p + img_h
#     x_p = 0
#     y_p = y_p + img_w
# # to_image = to_image[:, :, (2, 1, 0)]
# cv2.imwrite(IMAGE_SAVE_PATH, to_image)
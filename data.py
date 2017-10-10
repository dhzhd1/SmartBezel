import mxnet as mx
import PIL
import os
import cv2
import numpy as np


def create_rec_file(notation_list, train_data_ratio=.8):
    # The coordinate of the bbox [left, top, right, bottom]
    print("There are {} images need to be converted to rec file...".format(len(notation_list)))
    train_rec_file_name = os.path.join('./datasets/', 'train')
    val_rec_file_name = os.path.join('./datasets/', 'val')
    train_rec = mx.recordio.MXIndexedRecordIO(train_rec_file_name + '.idx', train_rec_file_name + '.rec', 'w')
    # val_rec = mx.recordio.MXIndexedRecordIO(val_rec_file_name + '.idx', val_rec_file_name + '.rec', 'w')
    idx = 0
    for notation in notation_list:
        img = cv2.imread(notation[1])
        imgR, labelR = image_bbox_resize(img, notation[2], dsize=(224,224))
        header = mx.recordio.IRHeader(flag=0, label=labelR.flatten(), id=0, id2=0)
        s = mx.recordio.pack_img(header, imgR, quality=100, img_fmt='.jpg')
        train_rec.write_idx(idx, s)
        idx += 1
    print("JPG to REC is Done")
    train_rec.close()

def image_bbox_resize(image, bbox, dsize=(224, 224)):
    himg, wimg = image.shape[:2]
    imgR = cv2.resize(image, dsize)
    left, top, right, bottom = bbox
    top_r = 1.0 * top / himg
    bottom_r = 1.0 * bottom / himg
    left_r = 1.0 * left / wimg
    right_r = 1.0 * right / wimg
    return imgR, np.array([1, left_r, top_r, right_r, bottom_r])


if __name__ == "__main__":
    notation_file = [[1, './datasets/images/A.J._Buckley/00000001.jpg', [165.21, 105.50, 298.57, 238.86]],
                     [1, './datasets/images/A.J._Buckley/00000002.jpg', [56.05, 130.34, 222.86, 297.15]],
                     [1, './datasets/images/A.J._Buckley/00000003.jpg', [79.79, 138.89, 256.08, 315.17]],
                     [1, './datasets/images/A.J._Buckley/00000005.jpg', [28.19, 65.33, 111.43, 148.57]],
                     [1, './datasets/images/A.J._Buckley/00000006.jpg', [89.90, 70.00, 179.14, 159.24]],
                     [1, './datasets/images/A.J._Buckley/00000007.jpg', [60.05, 139.67, 238.86, 318.48]],
                     [1, './datasets/images/A.J._Buckley/00000008.jpg', [68.93, 137.52, 274.37, 342.97]],
                     [1, './datasets/images/A.J._Buckley/00000010.jpg', [146.51, 195.01, 363.77, 412.27]]]
    print(notation_file)
    create_rec_file(notation_file)


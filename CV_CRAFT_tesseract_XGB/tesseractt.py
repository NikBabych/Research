from integr import  *
import pytesseract
import pandas as pd
from pytesseract import Output
class BboxLabel:
    def __init__(self, label, x0, y0, w, h):
        self.h = h
        self.w = w
        self.y0 = y0
        self.x0 = x0
        self.label = label

    @classmethod
    def from_str(cls, label_str):
        split = label_str.strip().split(" ")
        label, x0, y0, w, h = split
        return cls(int(label), float(x0), float(y0), float(w), float(h))

    def bbox(self, h, w):
        x0 = int((self.x0 - self.w / 2) * w)
        y0 = int((self.y0 - self.h / 2) * h)
        x1 = int((self.x0 + self.w / 2) * w)
        y1 = int((self.y0 + self.h / 2) * h)
        return x0, y0, x1, y1


def get_grount_truth(path_label, path_image):
    with open(path_label) as f:
        labels = f.read()
    labels = labels.split('\n')
    img_label = cv2.imread(path_image)

    true_label = {}
    for ind, i in enumerate(labels[:-1]):
        q = BboxLabel.from_str(i)
        x, y, w, h = q.bbox(img_label.shape[0], img_label.shape[1])
        label = q.label

        true_label[ind] = (x, y, w - x, h - y, label)
    return true_label
net=create_craft()


net=create_craft()
lst_df=[]
for l in ['0_0']:
    pth_img='selenium/'+l+'.png'
    pth_txt='selenium/'+l+'.txt'
    labels = detect_save_text(pth_img, net)
    img = cv2.imread(pth_img)
    result = img.copy()
    without_text, mean_text = preproc_mean_text(labels, img)
    contours = get_contours(without_text)
    rect_dict, line_dict = get_rect_line(contours)
    for ind, line in line_dict.items():

        x, y, w, h = line.ravel()[:2][0], line.ravel()[:2][1], line.ravel()[-2:][0] - line.ravel()[:2][0], -25
        rect_dict[ind + 200] = (x, y, w, h)
    rect_dict = filter_rect(result, rect_dict, mean_text, threshold_inter=0.2, distance_text=300)

    x_3 = get_grount_truth(pth_txt, pth_img)
    dict_sel = get_df(pth_img, rect_dict, x_3)
    print(dict_sel)
    img_list=[]
    for k, v in dict_sel.items():

        dic_l = {}
        dic_l['area'] = v[-4]
        dic_l['in_text'] = v[-2]
        dic_l['len_text'] = v[-1]
        dic_l['image'] = v[-3]
        dic_l['label']=v[-6]
        img_list.append(dic_l)

    lst_df.append(img_list)
last=[]
for i in lst_df:
    last=last+i



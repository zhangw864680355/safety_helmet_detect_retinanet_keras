import xml.etree.ElementTree as ET
import os
import random

    
def file_is_img(img_path):
    postfix = img_path.strip().split('.')[-1]
    if postfix.lower() in ['jpg', 'jpeg', 'png', 'bmp']:
        if postfix in ['JPG', 'JPEG']:
            os.rename(img_path, img_path.replace(postfix, 'jpg'))
            print(img_path)
        return True
    else:
        return False

def get_image_idx(dir_path):
    img_idx_list = []
    if not os.path.exists(dir_path):
        return []
    for root, dirs, files in os.walk(dir_path):
        for img in files:
            img_path = os.path.join(root, img)
            if file_is_img(img_path):
                img_idx_list.append(img.split('.')[0])
    return img_idx_list


def convert_annotation(root_path, image_idx, classes, csv_file):
    annotation_path = root_path + 'Annotations/{}.xml'.format(image_idx) 
    if not os.path.exists(annotation_path):
        return None
    
    in_file = open(annotation_path)

    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        #cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        csv_file.write(root_path + 'JPEGImages/{}.jpg'.format(image_idx))
        csv_file.write("," + ",".join([str(a) for a in b]) + ',' + cls)
        csv_file.write('\n')
        

if __name__ == '__main__':
    classes = ["hat", "person"]
    root = '/opt/sdb/workspace/data/VOCdevkit/VOC-Safety-Helmet/'
    dir_path = root + 'JPEGImages'
    image_idx_list = get_image_idx(dir_path)
    random.shuffle(image_idx_list)

    val_split = 0.2
    image_idx_val_list = image_idx_list[:int(len(image_idx_list)*val_split)]
    image_idx_train_list = image_idx_list[int(len(image_idx_list)*val_split):]

    for data_type in ['train', 'val']:
        csv_path = 'voc_{}_annotation.csv'.format(data_type)
        if os.path.exists(csv_path):
            os.remove(csv_path)
        if data_type == 'train':
            image_idx_list = image_idx_train_list
        else:
            image_idx_list = image_idx_val_list
        
        with open(csv_path, 'w') as f:
            for image_idx in image_idx_list:
                convert_annotation(root, image_idx, classes, f)

    classes_csv_path = 'classes.csv' 
    if os.path.exists(classes_csv_path):
        os.remove(classes_csv_path)

    with open(classes_csv_path, 'w') as f:
        for i, cls in enumerate(classes):
            f.write(cls + ',' + str(i) + '\n')


    print('finished done ...')

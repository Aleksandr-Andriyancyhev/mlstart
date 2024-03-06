import xml.etree.ElementTree as ET
import csv
from PIL import Image
import os

def crop_object(image_path, coordinates, output_path):
    image = Image.open(image_path)
    cropped_image = image.crop(coordinates)
    cropped_image = cropped_image.convert('RGB')
    cropped_image.save(output_path)

def parsing(file_path:str):
    xml_list = []
    tree = ET.parse(file_path)
    for group1 in tree.findall('image'):
        xml_name = group1.get('name')
        xml_subset = group1.get("subset")
        for group2 in group1.findall('box'):
                xml_class = group2.get('label')
                xml_xtl = float(group2.get('xtl'))
                xml_ytl = float(group2.get('ytl'))
                xml_xbr = float(group2.get('xbr'))
                xml_ybr = float(group2.get('ybr'))
                dict = {'Name' : xml_name,
                    'Subset' : xml_subset,
                    'Class' : xml_class,
                    'xtl' : xml_xtl,
                    'ytl' : xml_ytl,
                    'xbr' : xml_xbr,
                    'ybr' : xml_ybr}
                xml_list.append(dict)
                
    return xml_list

def main() -> None:
    images_folder = 'C:\\Users\\profi\\Desktop\\ml\\photo'
    output_folder = 'C:\\Users\\profi\\Desktop\\ml\\ships_vs_planes\\'
    xml_file = 'C:\\Users\\profi\\Desktop\\ml\\objects\\annotations.xml'

    xml_list = parsing(xml_file)
    with open("dataset.csv", "w", newline="") as csvfile:
        fieldnames = ["Name", "Subset", "Class", 'xtl', 'ytl', 'xbr', 'ybr']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for dictionary in xml_list:
            writer.writerow(dictionary)


    with open('dataset.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image_filename = row['Name']
            x1 = float(row['xtl'])
            y1 = float(row['ytl'])
            x2 = float(row['xbr'])
            y2 = float(row['ybr'])
            subset = row['Subset']
            class_name = row['Class']
            image_path = os.path.join(images_folder, image_filename)
            output_folder_path = f'{output_folder}{subset}\\{class_name}'
            os.makedirs(output_folder_path, exist_ok=True)
            output_path = os.path.join(output_folder_path, f'{class_name}_{image_filename}')
            crop_object(image_path, (x1, y1, x2, y2), output_path)

if __name__ == "__main__":
    main()

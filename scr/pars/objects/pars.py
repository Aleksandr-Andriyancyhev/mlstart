import xml.etree.ElementTree as ET
import csv

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
                dict = {
                    'Name' : xml_name,
                    'Subset' : xml_subset,
                    'Class' : xml_class,
                    'xtl' : xml_xtl,
                    'ytl' : xml_ytl,
                    'xbr' : xml_xbr,
                    'ybr' : xml_ybr

                }
                xml_list.append(dict)
                
    return xml_list

xml_file = 'C:\\Users\\profi\\Desktop\\ml\\objects\\annotations.xml'

xml_list = parsing(xml_file)

with open("dataset.csv", "w", newline="") as csvfile:
    fieldnames = ["Name", "Subset", "Class", 'xtl', 'ytl', 'xbr', 'ybr']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for dictionary in xml_list:
        writer.writerow(dictionary)

import numpy as np
from pandas import read_csv
import os
import requests
from requests.exceptions import HTTPError
import glob
import argparse
import json


def download_image(url, dest_folder, image_name):
    """
    :param url: url link for the remote image by using VGG face dataset
    :param dest_folder: dedicate folder for each person's image
    :param image_name: image name for downloaded file (it will be a image id)
    :return: Bool value, True and False to indicate the results
    """
    try:
        r = requests.get(url, stream=True)
        with open(os.path.join(dest_folder, image_name + ".jpg"), 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("INFO: Image has been downloaded from '{}'".format(url))
        return  True
    except HTTPError:
        print("ERROR: Image cannot be downloaded from '{}'".format(url))
        return False

def get_notation_file_list(folder_name):
    """
    :param folder_name: The folder path where contains notation files
    :return: list of notation file name with path
    """
    search_pattern = os.path.join(folder_name, "*.txt")
    notation_file_list = glob.glob(search_pattern)
    return sorted(notation_file_list)

def load_notation_file(file_name):
    notation_content = read_csv(file_name, '\s+',
                                names=["id", "url", "left", "top", "right", "bottom", "pose", "score", "curation"])
    notation_content['width'] = notation_content['right'] - notation_content['left']
    notation_content['height']  = notation_content['bottom'] - notation_content['top']
    notation_content['center_x'] = notation_content['left'] + (notation_content['right'] - notation_content['left']) / 2
    notation_content['center_y'] = notation_content['top'] + (notation_content['bottom'] - notation_content['top']) /2
    return notation_content

def get_person_name(file_name):
    return file_name.split('/')[-1].split('.txt')[0]

def check_folder_name(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

def save_notation_db(notation_array, output_file):
    try:
        with open(output_file, 'w') as f:
            json.dump(notation_array, f)
        print("INFO: Notation DB has been saved on {}".format(output_file))
        return True
    except:
        print("ERROR: Failed on save notation db")
        return False


def main():
    parser = argparse.ArgumentParser('Download Dataset and Generate Notation files')
    parser.add_argument('--notation-file-path', help='File Path of Notations', required=True, type=str)
    parser.add_argument('--download-folder', help='Output folder for downloaded images', required=True, type=str)
    parser.add_argument('--notation-db', help='Numpy array image file for notations', required=True, type=str)
    args = parser.parse_args()
    if not os.path.exists(args.notation_file_path):
        print("ERROR: Notation file folder cannot be found or existed. ")
        print("ERROR: Program existed. ")
        return

    notation_file = []

    check_folder_name(args.download_folder)
    notation_file_list = get_notation_file_list(args.notation_file_path)
    for file_name in notation_file_list:
        person_name = get_person_name(file_name)
        person_image_folder = os.path.join(args.download_folder, person_name)
        check_folder_name(person_image_folder)
        notation_file_content = load_notation_file(file_name)
        for i in xrange(len(notation_file_content)):
            print(notation_file_content.url[i])
            if download_image(notation_file_content.url[i], person_image_folder,
                              '{0:08d}'.format(int(notation_file_content.id[i]))):
                img_note = [person_name,
                            os.path.join(person_image_folder, '{0:08d}'.format(int(notation_file_content.id[i]))),
                            [notation_file_content['left'], notation_file_content['top'],
                             notation_file_content['right'], notation_file_content['bottom']],
                            [notation_file_content['width'], notation_file_content['height']],
                            [notation_file_content['center_x'], notation_file_content['center_y']]]
                notation_file.append(img_note)
        print("Processed {}.".format(file_name))
    print("All notation files have been processed!")

    save_notation_db(notation_file, os.path.join(args.notation_db))


if __name__ == "__main__":
    def test_get_notation_file_list():
        file_list = get_notation_file_list("./datasets/vgg_face_dataset/files/")
        for i in xrange(5):
            print(file_list[i])

    def test_load_notation_file():
        result = load_notation_file("./datasets/vgg_face_dataset/files/A.J._Buckley.txt")
        print(result.head(5))
        print(result.url[:5])

    def test_get_person_name():
        name = get_person_name("./datasets/vgg_face_dataset/files/A.J._Buckley.txt")
        print(name)

    def test_download_image():
        notation_folder = "./datasets/vgg_face_dataset/files/"
        notation_file_list = get_notation_file_list(notation_folder)
        '''
        testing the first notation file and try to download the first 5 images
        '''
        person_name = get_person_name(notation_file_list[0])
        dest_folder = os.path.join("./datasets/temp", person_name)
        check_folder_name(dest_folder)
        content = load_notation_file(notation_file_list[0])
        for i in xrange(5):
            download_image(content.url[i], dest_folder, '{0:08d}'.format(i))

    main()



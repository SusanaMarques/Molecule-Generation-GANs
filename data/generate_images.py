#!/usr/bin/env python3
import os
import tarfile

import cv2

import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

molecules = pd.read_csv("train.csv")

OUTPUT_DIR = "images"

os.mkdir(OUTPUT_DIR)

def process_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    crop_image = gray_image[125:375, 0:500]
    cv2.imwrite(image_path, crop_image)

i = 0
for molecule in molecules["SMILES"]:
    image_path = f'images/{molecule}.png'
    Draw.MolToFile(Chem.MolFromSmiles(molecule), image_path, size=(500,500))
    process_image(image_path)
    i += 1
    if i == 60000:
        break

tar = tarfile.open("images.tar.gz", "w:gz")
tar.add(OUTPUT_DIR)
tar.close()

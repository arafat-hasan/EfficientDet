#!/bin/bash

# Use: ./dataset-create-symlink.sh "path/to/dhaka-ai/Final Train Dataset" "path/to/copy/ImageSet/Main"
# This script creates a dataset made of symlinks in Pascal VOC structure.
# Train-Val-Test split is also created.

set -e

if [ $# -eq 0 ]
  then
    echo "Supply data dir path"
    exit
fi

if [ -d datasets/dhaka-ai/voc/ ]; then
    rm -r datasets/dhaka-ai/voc/
fi


mkdir -p datasets/dhaka-ai/voc/JPEGImages
mkdir -p datasets/dhaka-ai/voc/Annotations
mkdir -p datasets/dhaka-ai/voc/ImageSets/Main

for jpg in "$1"/*.jpg
do
    ln -s "$jpg" datasets/dhaka-ai/voc/JPEGImages
done

for JPG in "$1"/*.JPG
do
    ln -s "$JPG" datasets/dhaka-ai/voc/JPEGImages
done


for jpeg in "$1"/*.jpeg
do
    ln -s "$jpeg" datasets/dhaka-ai/voc/JPEGImages
done

for png in "$1"/*.png
do
    ln -s "$png" datasets/dhaka-ai/voc/JPEGImages
done

for PNG in "$1"/*.PNG
do
    ln -s "$PNG" datasets/dhaka-ai/voc/JPEGImages
done


for xml in "$1"/*.xml
do
    ln -s "$xml"  datasets/dhaka-ai/voc/Annotations
done


rm datasets/dhaka-ai/voc/JPEGImages/231.jpg # Corrupted file
rm datasets/dhaka-ai/voc/Annotations/231.xml    # Corrupted file

python3 png2jpg.py  # Convert all png files to jpg


rename JPG jpg datasets/dhaka-ai/voc/JPEGImages/*.JPG || :  # Convert uppercase file ext to lowercase
rename jpeg jpg datasets/dhaka-ai/voc/JPEGImages/*.jpeg || :

rename 's/\.JPG$/.jpg/' datasets/dhaka-ai/voc/JPEGImages/*.JPG || :
rename 's/\.jpeg$/.jpg/' datasets/dhaka-ai/voc/JPEGImages/*.jpeg || :


if [ $# -eq 2 ]
  then
    cp -r "$2"/* datasets/dhaka-ai/voc/ImageSets/Main/
else
  python3 generateimagesets.py    # Create train-test-val split in ImageSets dir  
fi




echo "Done"
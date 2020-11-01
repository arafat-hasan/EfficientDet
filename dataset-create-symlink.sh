#!/bin/bash


if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    exit
fi

if [ -d datasets/ ]; then
    rm -r datasets/
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

rm datasets/dhaka-ai/voc/JPEGImages/231.jpg
rm datasets/dhaka-ai/voc/Annotations/231.xml

python3 png2jpg.py


rename 's/\.JPG$/.jpg/' datasets/dhaka-ai/voc/JPEGImages/*.JPG
rename 's/\.jpeg$/.jpg/' datasets/dhaka-ai/voc/JPEGImages/*.jpeg

cp ImageSets/Main* datasets/dhaka-ai/voc/ImageSets/Main/

# python3 generateimagesets.py
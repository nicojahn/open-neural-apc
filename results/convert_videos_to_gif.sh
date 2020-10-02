#!/bin/bash
source_folder="videos"
file_extension="mp4"
target_folder="gifs"
find $source_folder/ -name 'video*.'$file_extension -type f -exec bash -c 'i="$1"; echo $i | sed -e "s/$2/gif/g" -e "s/videos//g" -e "s/video//g" -e "s/\//gifs\//g" | xargs ffmpeg -y -i $i -vf "fps=10,scale=-1:320" -loop 0' _ {} $file_extension \;

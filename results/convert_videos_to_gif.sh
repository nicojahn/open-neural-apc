#!/bin/bash
source_folder="videos"
target_folder="gifs"
find $source_folder/ -name 'video*.avi' -type f -exec bash -c 'i="$1"; echo $i | sed -e "s/avi/gif/g" -e "s/videos//g" -e "s/video//g" -e "s/\//gifs\//g" | xargs ffmpeg -y -i $i -vf "fps=10" -loop 0' _ {} \;

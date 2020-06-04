#/bin/bash
source_folder="videos"
target_folder="gifs"
find $source_folder/ -name 'video*.mp4' -type f -exec bash -c "echo {} | sed -e 's/mp4/gif/g' -e 's/videos//g' -e 's/video//g' -e 's/\//gifs\//g' | xargs ffmpeg -y -i {} -vf "fps=10" -loop 0 " \;

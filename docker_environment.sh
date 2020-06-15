#!/bin/bash

docker stop open-neural-apc
docker rm open-neural-apc

# --privileged=true : is needed for tensorboard profiling. Usually not recommended.
# -p 6006:6006 : is needed for tensorboard investigation
docker run -d --name open-neural-apc -p 8888:8888 \
	--gpus all -v $(pwd)/:/tf/ tensorflow/tensorflow:2.2.0-gpu-jupyter

sleep 3
docker logs open-neural-apc

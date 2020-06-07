#!/bin/bash

docker stop open-neural-apc
docker rm open-neural-apc

docker run -d --name open-neural-apc -p 8888:8888 \
	--gpus all -v $(pwd)/:/tf/ tensorflow/tensorflow2.2.0-gpu-jupyter
sleep 3
docker logs open-neural-apc

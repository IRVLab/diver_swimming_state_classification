WORKSPACE_DIR="/home/dtkutzke/workspace/anomaly_detection"
IMAGE_NAME=anomaly_detection
IMAGE_VERSION=fromrepo

docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 --network="host" \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v $WORKSPACE_DIR:/workspace \
    -v /mnt:/mnt \
    $IMAGE_NAME:$IMAGE_VERSION

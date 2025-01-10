# Use a base image with Python
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

# Set the working directory in the container
WORKDIR /app

# Copy the necessary files into the container
COPY requirements.txt requirements.txt
COPY src/segment_cacs_predict.py segment_cacs_predict.py
COPY src/model.py model.py

# Install the required Python libraries
RUN pip install -r requirements.txt

# Run the Python script when the container starts
ENTRYPOINT ["python", "segment_cacs_predict.py", "--device","cpu", "--model_dir" , "/app/data/model/SegmentCACS_0001619_unet.pt", "--data_dir", "/app/data/images", "--prediction_dir", "/app/data/predictions", "--filetype", "mhd"]

# --- Linux ---
# Build docker
# docker build -t segmentcacs .
# Run docker cpu
# docker run --rm --mount type=bind,src=/mnt/HHD/data/SegmentCACSSeg/docker/code/src/data,dst=/app/data --user "$(id -u):$(id -g)" -it segmentcacs
# Run docker gpu
# docker run --rm --gpus 1 --mount type=bind,src=/mnt/HHD/data/SegmentCACSSeg/docker/code/src/data,dst=/app/data --user "$(id -u):$(id -g)" -it segmentcacs --gpus device=0 nvidia/cuda

# --- Windows ---
# Builddocker
# docker build -t segmentcacs .
# Run docker cpu
# docker run --rm --gpus 1 --mount type=bind,src=C:/docker_test/code/data,dst=/app/data -it segmentcacs
# Run docker gpu
# docker run --rm --gpus 1 --mount type=bind,src=C:/docker_test/code/data,dst=/app/data -it segmentcacs --device gpu


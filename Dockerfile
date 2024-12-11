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
CMD ["python", "segment_cacs_predict.py", "--device","cpu", "--model_dir" , "/app/data/model/SegmentCACS_0001619_unet.pt", "--data_dir", "/app/data/images", "--prediction_dir", "/app/data/predictions"]

# Build docker:
# docker build -t segmentcacs .

# Run docker:
# docker run --mount type=bind,src=/mnt/HHD/data/SegmentCACSSeg/docker/code/src/data,dst=/app/data --user "$(id -u):$(id -g)" -it segmentcacs
# docker run --mount type=bind,src=/mnt/HHD/data/SegmentCACSSeg/docker/code/src/data,dst=/app/data --user "$(id -u):$(id -g)" -it segmentcacs --gpus device=0 nvidia/cuda
# --gpus device=0 nvidia/cuda
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install conda
RUN apt-get update && apt-get install -y wget bzip2 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda init

# Create the conda environment
COPY environment.yml .
RUN /opt/conda/bin/conda env create -n inferencing-env -f environment.yml

# Activate the conda environment and install MLflow
RUN /opt/conda/bin/conda run -n inferencing-env pip install mlflow tensorflow flask

# Set the MLflow tracking URI as an environment variable
ENV MLFLOW_TRACKING_URI="http://127.0.0.1:8899"

# Make port 5001 available to the world outside this container
EXPOSE 5001

# Run the application
CMD ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "inferencing-env", "python", "app.py"]

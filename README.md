# Homogenised Finite Elements (HFE)

## Building Dependencies

This project uses Docker to manage its dependencies. To build the Docker image, follow these steps:

1. Install Docker on your machine if you haven't already. You can download it from [here](https://www.docker.com/products/docker-desktop).

2. Navigate to the project directory that contains the Dockerfile. In this case, it's the `02_CODE` directory.

```sh
cd 02_CODE
```

1. Build the Docker image using the Dockerfile.ubuntu24.04 file. Replace your_image_name with the name you want to give to your Docker image.

```sh
docker build -t your_image_name -f Dockerfile.ubuntu24.04 .
```

## Running the Docker Image

After building the Docker image, you can run it using the following command:

```sh
docker run -it your_image_name
```

This will start a Docker container with the built image and open an interactive shell in the container. The Docker container has all the dependencies installed and the environment set up as specified in the Dockerfile.

## Running the Project

Once you're inside the Docker container, you can run the project. The exact command depends on how your project is structured, but it will generally look something like this:

```sh
conda init
source .bashrc
conda activate hfe-essentials
cd 02_CODE
python src/pipeline_runner.py
```

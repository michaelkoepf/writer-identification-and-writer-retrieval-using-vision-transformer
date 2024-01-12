FROM python:3.8-slim-buster

RUN apt-get update -y &&  \
    apt-get install -y --no-install-recommends curl python3-dev python3-pip libsm6 libxext6 libxrender-dev libatlas-base-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libopencv-dev build-essential pkg-config libjpeg-dev libpng-dev libgtk-3-dev  \
    && rm -rf /var/lib/apt/lists/*

# GID of the host user (use --build-arg HOST_GID=$(id -g) when building the image)
ARG HOST_GID
# UID of the host user (use --build-arg HOST_UID=$(id -u) when building the image)
ARG HOST_UID
# non-root user
ARG USER=user
# constraints file for PyTorch installation
ARG CONSTRAINTS_FILE

# add a new group and a new non-root user
# use the passed GID and UID of the host user to avoid permissions problems with mounted files
RUN addgroup --system --gid $HOST_GID $USER && adduser --system --uid $HOST_UID --gid $HOST_GID --disabled-password $USER

# working directory
WORKDIR /app
RUN chown -R $USER:$USER /app

# leverage docker cache and re-install dependencies only if requirements have changed
COPY --chown=$USER:$USER requirements ./requirements

# disable pip version check and cache
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PIP_NO_CACHE_DIR 1

# install dependencies
RUN pip3 install -r requirements/requirements.txt -c requirements/$CONSTRAINTS_FILE

# copy all other files
COPY --chown=$USER:$USER . .

# run entrypoint as non-root user
USER $USER

CMD ["bash"]
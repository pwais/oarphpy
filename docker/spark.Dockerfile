# Copyright 2023 Maintainers of OarphPy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG UBUNTU_VERSION=20.04

FROM ubuntu:${UBUNTU_VERSION} as base

# We don't care for __pycache__ and .pyc files; sometimes VSCode doesn't clean
# up properly when deleting things and the cache gets stale.
ENV PYTHONDONTWRITEBYTECODE 1

RUN \
  apt-get update && \
  apt-get install -y \
    python3 \
    python3-pip

# Choose a spark to install
RUN \
  DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-11-jdk && \
  echo JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 >> /etc/environment
RUN pip3 install pyspark==3.3.1
RUN pip3 install numpy pandas>=1.0.0
ENV PYSPARK_PYTHON python3
ENV PYSPARK_DRIVER_PYTHON python3

COPY . /opt/oarphpy
WORKDIR /opt/oarphpy

RUN pip3 install -e ".[spark]"

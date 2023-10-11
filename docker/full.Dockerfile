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

# syntax=docker/dockerfile:1


## Spark

FROM curlimages/curl:8.00.1 AS spark-fetch
ARG SPARK_VERSION=3.5.0
RUN curl -L --retry 3 \
  "https://dlcdn.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz" \
  | tar xz -C /tmp/ \
  && mv /tmp/spark-${SPARK_VERSION}-bin-hadoop3 /tmp/spark


## Firefox (Selenium)

# Firefox moved to be snap-package-only, which is dumb, so we have to install from tarball
# NB see: https://github.com/googlecolab/colabtools/issues/3861#issuecomment-1644329617
FROM curlimages/curl:8.00.1 AS selenium-firefox-fetch

RUN curl -L --retry 3 \
  "https://download.mozilla.org/?product=firefox-latest&os=linux64&lang=en-US" \
  | tar xj -C /tmp/ \
  && echo "Copy from /tmp/firefox"


## Geckodriver (Selenium)

# Firefox moved to be snap-package-only, which is dumb, so we have to install from tarball
# NB see: https://github.com/googlecolab/colabtools/issues/3861#issuecomment-1644329617
FROM curlimages/curl:8.00.1 AS selenium-geckodriver-fetch

RUN curl -L --retry 3 \
  "https://github.com/mozilla/geckodriver/releases/download/v0.33.0/geckodriver-v0.33.0-linux64.tar.gz" \
  | tar xz -C /tmp/ \
  && echo "Copy from /tmp/geckodriver"


## GCS-Spark connector.  S3 is built-in

FROM curlimages/curl:8.00.1 AS gcs-connector-fetch
RUN \
  cd /tmp && \
  curl -L --retry 3 https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar \
    -o gcs-connector-hadoop3-latest.jar


## Main

FROM oarphpy/lambda-stack:22.04

# We don't care for __pycache__ and .pyc files; sometimes VSCode doesn't clean
# up properly when deleting things and the cache gets stale.
ENV PYTHONDONTWRITEBYTECODE 1


### Core
RUN \
  --mount=type=cache,target=/var/cache/apt \
  apt-get update && \
  echo "Required for installing and testing things:" && \
  apt-get install -y \
    curl \
    git \
    python-dev-is-python3 \
    python3-pip \
    python3-dev \
    wget \
  && \
  echo "Base dev tools:" && \
  apt-get install -y \
    dnsutils \
    iputils-ping \
    less \
    net-tools \
    screen \
    ssh \
    sudo \
    tree \
    unzip \
    vim

## Java
RUN \
  --mount=type=cache,target=/var/cache/apt \
  apt-get update && \
  apt-get install -y openjdk-11-jdk && \
  ls -lhat /usr/lib/jvm/java-11-openjdk-amd64 && \
  echo JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 >> /etc/environment
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64


### Dev tools
COPY docker/.vimrc /root/.vimrc
COPY docker/.screenrc /root/.screenrc
RUN \
  pip3 install --upgrade pip setuptools wheel && \
  pip3 install ipdb pytest && \
  pip3 install sphinx==6.1.3 recommonmark m2r m2r2==0.2.5 sphinx-rtd-theme

## Imageio + ffmpeg. Note that newest imageio dropped the ffmpeg download util :(
RUN \
  pip3 install imageio==2.22.4 && \
  pip3 install --upgrade imageio-ffmpeg

## Jupyter & friends
RUN \
  --mount=type=cache,target=/var/cache/apt \
  apt-get install -y nodejs && \
  pip3 install \
    jupyterlab==3.5.2 \
    ipykernel \
    ipywidgets \
    jupyter_http_over_ws \
    jupyterlab_widgets \
    matplotlib \
    nbformat \
    scipy \
    widgetsnbextension \
  && \
  jupyter serverextension enable --py jupyter_http_over_ws && \
  ln -s /usr/bin/ipython3 /usr/bin/ipython

## Selenium
## Used for **testing** oarhpy.plotting / bokeh
COPY --from=selenium-firefox-fetch /tmp/firefox /opt/firefox
RUN ln -s /opt/firefox/firefox /usr/local/bin/firefox
COPY --from=selenium-geckodriver-fetch /tmp/geckodriver /usr/local/bin/geckodriver
RUN \
  --mount=type=cache,target=/var/cache/apt \
  apt-get install -y libxtst6 libdbus-glib-1-2 && \
  pip3 install selenium==4.7.2 webdriver_manager



## Spark, now with built-in Hadoop
ENV SPARK_HOME /opt/spark

# Keep in sync with spark-fetch, docker doesn't yet support sub-args (?)
ENV SPARK_VERSION 3.5.0
COPY --from=spark-fetch /tmp/spark $SPARK_HOME

ENV PYSPARK_PYTHON python3
ENV PATH $PATH:${SPARK_HOME}/bin
RUN cd $SPARK_HOME/python && python3 setup.py install


# Spark install above appears to default to INFO for logs, perhaps via Hadoop.
# Let's make the WARN instead (the prior standard):
# https://github.com/apache/spark/blob/13fd9eee164604485e12bc03c73357a55572a630/conf/log4j.properties.template#L19
RUN \
  cp -v /opt/spark/conf/log4j2.properties.template /opt/spark/conf/log4j2.properties && \
  sed -i -e 's/rootLogger.level = info/rootLogger.level = warn/g' /opt/spark/conf/log4j2.properties && \
  echo "log4j.rootCategory=WARN,console" >> /opt/spark/conf/log4j.properties && \
  echo "log4j.appender.console=org.apache.log4j.ConsoleAppender" >> /opt/spark/conf/log4j.properties && \
  echo "log4j.appender.console.target=System.err" >> /opt/spark/conf/log4j.properties && \
  echo "log4j.appender.console.layout=org.apache.log4j.PatternLayout" >> /opt/spark/conf/log4j.properties && \
  echo "log4j.appender.console.layout.ConversionPattern=%d{yy/MM/dd HH:mm:ss} %p %c{1}: %m%n" >> /opt/spark/conf/log4j.properties


# GCloud with Spark Support
RUN \
  curl https://sdk.cloud.google.com | bash && \
  pip3 install -U crcmod && \
  pip3 install --upgrade google-api-python-client && \
  pip3 install --upgrade oauth2client 
COPY --from=gcs-connector-fetch /tmp/gcs-connector-hadoop3-latest.jar $SPARK_HOME/jars/gcs-connector-hadoop3-latest.jar
ENV PATH $PATH:/root/google-cloud-sdk/bin/


## SparkMonitor from https://github.com/swan-cern/sparkmonitor
RUN \
  pip3 install sparkmonitor==2.1.1 && \
  ipython profile create && \
  echo "c.InteractiveShellApp.extensions.append('sparkmonitor.kernelextension')" >> \
    $(ipython profile locate default)/ipython_kernel_config.py && \
  jupyter nbextension install sparkmonitor --py && \
  jupyter nbextension enable  sparkmonitor --py

# Finally, make sure jupyter lab is built
RUN jupyter lab build --debug


## Include oarphpy
COPY docker/bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

# FIXME Somebody installs "flatbuffers-1.12.1-git20200711.33e2d80-dfsg1-0.6" which breaks setuptools
RUN pip3 install "flatbuffers==2.0"

# FIXME https://github.com/pypa/setuptools/issues/3772#issuecomment-1384671296
RUN pip3 install "setuptools==65.7.0"

# Shallow copy install stuff for faster iteration
COPY ./setup.py /tmp/install-op/setup.py
COPY ./oarphpy/__init__.py /tmp/install-op/oarphpy/__init__.py
RUN cd /tmp/install-op && pip3 install -v -e ".[all]" && rm -rf /tmp/install-op

COPY . /opt/oarphpy
WORKDIR /opt/oarphpy
ENV PYTHONPATH $PYTHONPATH:/opt/oarphpy

RUN pip3 install -v -e ".[all]"

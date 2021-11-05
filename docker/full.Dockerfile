# Copyright 2020 Maintainers of OarphPy
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

FROM oarphpy/lambda-stack:20.04

# We don't care for __pycache__ and .pyc files; sometimes VSCode doesn't clean
# up properly when deleting things and the cache gets stale.
ENV PYTHONDONTWRITEBYTECODE 1


### Core
### Required for installing and testing things
RUN \
  apt-get update && \
  apt-get install -y \
    curl \
    git \
    python-dev \
    python3-pip \
    python3-dev \
    wget


# Tensorflow 1.15.x support for nvidia stubs is now broken, thus if we try to use
# tensorflow-gpu with a cpu-only host, we'll run into a showstopping bug. (Upon importing
# tensorflow, we see an infinite hang with python waiting on a pipe with fd number 15).
# The lambda stack base includes tensorflow-gpu.  Uninstal that now and pin to CPU-only
# tensorflow
RUN \
  pip3 uninstall -y tensorflow-gpu && \
  pip3 install --upgrade --ignore-installed tensorflow-cpu==2.3.0 && \
  echo "Test tensorflow" && python3 -c 'import tensorflow as tf; print(tf.config.list_physical_devices())'

### Spark (& Hadoop)
### Use a binary distro for:
###  * Spark LZ4 support through Hadoop
###  * Spark env file hacking (e.g. debug / profiling)
ENV HADOOP_VERSION 3.2.1
ENV HADOOP_HOME /opt/hadoop
ENV HADOOP_CONF_DIR $HADOOP_HOME/etc/hadoop
ENV PATH $PATH:$HADOOP_HOME/bin
ENV LD_LIBRARY_PATH "$HADOOP_HOME/lib/native/:$LD_LIBRARY_PATH"
RUN curl -L --retry 3 \
  "https://www.apache.org/dyn/mirrors/mirrors.cgi?action=download&filename=hadoop/common/hadoop-$HADOOP_VERSION/hadoop-$HADOOP_VERSION.tar.gz" \
  | gunzip \
  | tar -x -C /opt/ \
 && mv /opt/hadoop-$HADOOP_VERSION $HADOOP_HOME \
 && rm -rf $HADOOP_HOME/share/doc

ENV SPARK_VERSION 3.0.1
ENV SPARK_PACKAGE spark-${SPARK_VERSION}-bin-without-hadoop
ENV SPARK_HOME /opt/spark
ENV PYSPARK_PYTHON=python3
ENV SPARK_DIST_CLASSPATH "$HADOOP_HOME/etc/hadoop/*:$HADOOP_HOME/share/hadoop/common/lib/*:$HADOOP_HOME/share/hadoop/common/*:$HADOOP_HOME/share/hadoop/hdfs/*:$HADOOP_HOME/share/hadoop/hdfs/lib/*:$HADOOP_HOME/share/hadoop/hdfs/*:$HADOOP_HOME/share/hadoop/yarn/lib/*:$HADOOP_HOME/share/hadoop/yarn/*:$HADOOP_HOME/share/hadoop/mapreduce/lib/*:$HADOOP_HOME/share/hadoop/mapreduce/*:$HADOOP_HOME/share/hadoop/tools/lib/*"
ENV PATH $PATH:${SPARK_HOME}/bin
RUN curl -L --retry 3 \
  "https://www.apache.org/dyn/mirrors/mirrors.cgi?action=download&filename=spark/spark-${SPARK_VERSION}/${SPARK_PACKAGE}.tgz" \
  | gunzip \
  | tar x -C /opt/ \
 && mv /opt/$SPARK_PACKAGE $SPARK_HOME
RUN cd /opt/spark/python && python3 setup.py install

## Java
RUN \
  apt-get update && \
  apt-get install -y openjdk-11-jdk && \
  ls -lhat /usr/lib/jvm/java-11-openjdk-amd64 && \
  echo JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 >> /etc/environment
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64

# GCloud with Spark / Hadoop support
RUN \
  curl https://sdk.cloud.google.com | bash && \
  pip3 install -U crcmod && \
  pip3 install --upgrade google-api-python-client && \
  pip3 install --upgrade oauth2client && \
    cd /opt/hadoop/share/hadoop/common/lib && \
    wget https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar && \
    cd -
ENV PATH $PATH:/root/google-cloud-sdk/bin/


### Dev tools
COPY docker/.vimrc /root/.vimrc
COPY docker/.screenrc /root/.screenrc
RUN \
  apt-get update && \
  apt-get install -y \
    curl \
    dnsutils \
    git \
    iputils-ping \
    less \
    net-tools \
    screen \
    ssh \
    sudo \
    tree \
    unzip \
    vim \
    wget \
        && \
  pip3 install --upgrade pip setuptools wheel && \
  pip3 install ipdb pytest && \
  pip3 install sphinx==2.3.1 recommonmark==0.6.0 m2r2==0.2.5 sphinx-rtd-theme==0.5.0 && \
  curl -LO https://github.com/BurntSushi/ripgrep/releases/download/0.10.0/ripgrep_0.10.0_amd64.deb && \
  dpkg -i ripgrep_0.10.0_amd64.deb

# Imageio + ffmpeg. Note that newest imageio dropped the ffmpeg download util :(
RUN \
  pip3 install imageio==2.4.1 && \
  python3 -c 'import imageio; imageio.plugins.ffmpeg.download()'

# Jupyter & friends
RUN pip3 install \
      jupyterlab 'jupyter-client>=6.1.7' matplotlib \
      jupyter_http_over_ws ipykernel nbformat \
      scipy \
      sklearn && \
    jupyter serverextension enable --py jupyter_http_over_ws && \
    ln -s /usr/bin/ipython3 /usr/bin/ipython

# SparkMonitor from https://github.com/swan-cern/jupyter-extensions
# NB: need nodejs v10, but only v8 available in Ubuntu 18.x
 RUN \
   curl -sL https://deb.nodesource.com/setup_10.x | bash - && \
   apt-get install -y nodejs && \
   pip3 install sparkmonitor>=1.1.0 && \
   jupyter nbextension install sparkmonitor --py && \
   jupyter nbextension enable  sparkmonitor --py && \
   jupyter serverextension enable --py --system sparkmonitor  && \
   jupyter lab build && \
   ipython profile create && echo "c.InteractiveShellApp.extensions.append('sparkmonitor.kernelextension')" >>  $(ipython profile locate default)/ipython_kernel_config.py

## Phantomjs and Selenium
## Used for **testing** oarhpy.plotting / bokeh
RUN \
  pip3 install selenium==3.8.0 && \
  cd /tmp && \
  wget https://bitbucket.org/ariya/phantomjs/downloads/phantomjs-2.1.1-linux-x86_64.tar.bz2 && \
  tar xvjf phantomjs-2.1.1-linux-x86_64.tar.bz2 && \
  cp phantomjs-2.1.1-linux-x86_64/bin/phantomjs /usr/bin/ && \
  rm -rf phantomjs-2.1.1-linux-x86_64 && \
  cd -

## Include oarphpy
COPY docker/bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

# FIXME pip3-install-editable isn't giving us the desired version of pandas
RUN pip3 install --upgrade --force-reinstall pandas>=1.1.2

COPY . /opt/oarphpy
WORKDIR /opt/oarphpy

RUN pip3 install -e ".[all]"

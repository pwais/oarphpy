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

FROM tensorflow/tensorflow:1.15.0-jupyter

# We don't care for __pycache__ and .pyc files; sometimes VSCode doesn't clean
# up properly when deleting things and the cache gets stale.
ENV PYTHONDONTWRITEBYTECODE 1


# FIXME(https://github.com/tensorflow/tensorflow/issues/18480)
# Tensorflow is broken: it includes enum34 improperly.  This in turn
# breaks the Python `re` module when Spark tries to use it during pyspark
# worker start-up.
RUN pip uninstall -y enum34


### Core
### Required for installing and testing things
RUN \
  apt-get update && \
  apt-get install -y \
    curl \
    git \
    python-dev \
    python-pip \
    python3-dev \
    wget


### Spark (& Hadoop)
### Use a binary distro for:
###  * Spark LZ4 support through Hadoop
###  * Spark env file hacking (e.g. debug / profiling)
ENV HADOOP_VERSION 3.1.3
ENV HADOOP_HOME /opt/hadoop
ENV HADOOP_CONF_DIR $HADOOP_HOME/etc/hadoop
ENV PATH $PATH:$HADOOP_HOME/bin
ENV LD_LIBRARY_PATH "$HADOOP_HOME/lib/native/:$LD_LIBRARY_PATH"
RUN curl -L --retry 3 \
  "http://mirrors.ibiblio.org/apache/hadoop/common/hadoop-$HADOOP_VERSION/hadoop-$HADOOP_VERSION.tar.gz" \
  | gunzip \
  | tar -x -C /opt/ \
 && mv /opt/hadoop-$HADOOP_VERSION $HADOOP_HOME \
 && rm -rf $HADOOP_HOME/share/doc

ENV SPARK_VERSION 2.4.5
ENV SPARK_PACKAGE spark-${SPARK_VERSION}-bin-without-hadoop
ENV SPARK_HOME /opt/spark
ENV SPARK_DIST_CLASSPATH "$HADOOP_HOME/etc/hadoop/*:$HADOOP_HOME/share/hadoop/common/lib/*:$HADOOP_HOME/share/hadoop/common/*:$HADOOP_HOME/share/hadoop/hdfs/*:$HADOOP_HOME/share/hadoop/hdfs/lib/*:$HADOOP_HOME/share/hadoop/hdfs/*:$HADOOP_HOME/share/hadoop/yarn/lib/*:$HADOOP_HOME/share/hadoop/yarn/*:$HADOOP_HOME/share/hadoop/mapreduce/lib/*:$HADOOP_HOME/share/hadoop/mapreduce/*:$HADOOP_HOME/share/hadoop/tools/lib/*"
ENV PATH $PATH:${SPARK_HOME}/bin
RUN curl -L --retry 3 \
  "https://www.apache.org/dyn/mirrors/mirrors.cgi?action=download&filename=spark/spark-${SPARK_VERSION}/${SPARK_PACKAGE}.tgz" \
  | gunzip \
  | tar x -C /opt/ \
 && mv /opt/$SPARK_PACKAGE $SPARK_HOME

## Java 8.  NB: can't use Java 11 yet for Spark
RUN \
  apt-get update && \
  apt-get install -y openjdk-8-jdk && \
  ls -lhat /usr/lib/jvm/java-8-openjdk-amd64 && \
  echo JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 >> /etc/environment
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64

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
    net-tools \
    screen \
    ssh \
    sudo \
    tree \
    unzip \
    vim \
    wget \
        && \
  pip3 install --upgrade setuptools wheel && \
  pip3 install ipdb pytest && \
  pip3 install sphinx recommonmark m2r sphinx-rtd-theme && \
  curl -LO https://github.com/BurntSushi/ripgrep/releases/download/0.10.0/ripgrep_0.10.0_amd64.deb && \
  dpkg -i ripgrep_0.10.0_amd64.deb
RUN \
  pip3 install imageio==2.4.1 && \
  python3 -c 'import imageio; imageio.plugins.ffmpeg.download()'
RUN \
  pip3 install sparkmonitor-s && \
  jupyter nbextension install sparkmonitor --py --user --symlink && \
  jupyter nbextension enable sparkmonitor --py --user && \
  jupyter serverextension enable --py --user sparkmonitor && \
  ipython profile create && echo "c.InteractiveShellApp.extensions.append('sparkmonitor.kernelextension')" >>  $(ipython profile locate default)/ipython_kernel_config.py
    # FIXME(https://github.com/krishnan-r/sparkmonitor/issues/18)
    # Use sparkmonitor official package when original author fixes it


## Phantomjs and Selenium
## Used for **testing** oarhpy.plotting / bokeh
RUN \
  pip3 install selenium && \
  cd /tmp && \
  wget https://bitbucket.org/ariya/phantomjs/downloads/phantomjs-2.1.1-linux-x86_64.tar.bz2 && \
  tar xvjf phantomjs-2.1.1-linux-x86_64.tar.bz2 && \
  cp phantomjs-2.1.1-linux-x86_64/bin/phantomjs /usr/bin/ && \
  rm -rf phantomjs-2.1.1-linux-x86_64 && \
  cd -


## Include oarphpy
COPY docker/bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

COPY . /opt/oarphpy
WORKDIR /opt/oarphpy

RUN pip3 install -e ".[all]"

```
                                          _________________________
                                         < OarphPy!! Oarph! Oarph! >
                                         <   OarphKit for Python!! >
                                          -------------------------
                                                        \
                                                         \
                         ____                __   ___       -~~~~-
                        / __ \___ ________  / /  / _ \__ __|O __ O|     
                       / /_/ / _ `/ __/ _ \/ _ \/ ___/ // /|_\__/_|__-  
                       \____/\_,_/_/ / .__/_//_/_/   \_,---(__/\__)---  
                                 .--/_/             /___/ /  ~--~  \    
                            ,__;`  o __`'.          _,..-/  | \/ |  \   
                            '  `'---'  `'.'.      .'.'` |   | /\ |   |
                                          .'-...-`.'  _/ /\__    __/\ \_
                                            -...-`  ~~~~~    ~~~~    ~~~~~
```

[![License](http://img.shields.io/:license-apache-orange.svg)](http://www.apache.org/licenses/LICENSE-2.0) 
[![Build Status](https://circleci.com/gh/pwais/oarphpy.png?style=shield)](https://circleci.com/gh/pwais/oarphpy/tree/master)

`OarphPy` is a collection of Python utilities for Data Science with
[PySpark](https://spark.apache.org/docs/latest/api/python/) and Tensorflow. 
Related (but orthogonal) to [OarphKit](https://github.com/pwais/oarphkit).

## Quickstart

Use the dockerized environment hosted on [DockerHub](https://hub.docker.com/u/oarphpy):
```
  $ ./oarphcli --shell
  -- or --
  $ docker run -it --net=host oarphpy/full bash
```

## Dockerized Development Environments

OarphPy is built and tested in a variety of environments to ensure the library
works with and without [optional dependencies](setup.py).  These environments
are shared on [DockerHub](https://hub.docker.com/u/oarphpy):
 
 * [`oarphpy/full`](docker/full.Dockerfile) -- Includes Tensorflow, Jupyter,
  a binary install of [Spark](https://spark.apache.org/), and other tools like
  [Bokeh](https://bokeh.org/). Use this environment for adhoc data science or
  as a starter for other projects.

 * [`oarphpy/base-py2`](docker/base-py2.Dockerfile) -- Tests `oarphpy` in a
  vanilla Python 2.7 environment to ensure clean interop with other projects.

 * [`oarphpy/base-py3`](docker/base-py3.Dockerfile) -- Tests `oarphpy` in a 
  vanilla Python 3 environment to ensure clean interop with other projects.

 * [`oarphpy/spark`](docker/spark.Dockerfile) -- Tests `oarphpy` with a vanilla
  install of [`pyspark`](https://spark.apache.org/) to ensure basic
  compatibility.

 * [`oarphpy/tensorflow`](docker/tensorflow.Dockerfile) -- Tests `oarphpy` with
  Tensorflow 1.x to ensure basic compatibility (e.g. of 
  [`oarphpy.util.tfutil`](oarphpy/util/tfutil.py)]).


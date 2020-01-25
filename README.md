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
[![PyPI version](https://badge.fury.io/py/oarphpy.svg)](https://badge.fury.io/py/oarphpy)

OarphPy is a collection of Python utilities for Data Science with
[PySpark](https://spark.apache.org/docs/latest/api/python/) and Tensorflow. 
Related (but orthogonal) to [OarphKit](https://github.com/pwais/oarphkit).

## Quickstart

Install from PyPI: `pip install oarphpy`.  We test OarphPy in a variet of 
environments (see below), so it should play well with your Jupyter/Colab
notebook or project environment.  To include all extras, use
`pip install oarphpy[all]`.

Or use the dockerized environment hosted on [DockerHub](https://hub.docker.com/u/oarphpy):
```
  $ ./oarphcli --shell
  -- or --
  $ docker run -it --net=host oarphpy/full bash
```

See also [API documentation](https://pwais.github.io/oarphpy/).

## Dockerized Development Environments

OarphPy is built and tested in a variety of environments to ensure the
library works with and without [optional dependencies](setup.py#L18).  These
environments are shared on [DockerHub](https://hub.docker.com/u/oarphpy) and 
defined in the [docker](docker) subdirectory of this repo:
 
 * `oarphpy/full` -- Includes Tensorflow, Jupyter, a binary install of
  [Spark](https://spark.apache.org/), and other tools like
  [Bokeh](https://bokeh.org/). Use this environment for adhoc data science or
  as a starter for other projects.

 * `oarphpy/base-py2` -- Tests `oarphpy` in a vanilla Python 2.7 environment
  to ensure clean interop with other projects.

 * `oarphpy/base-py3` -- Tests `oarphpy` in a vanilla Python 3 environment 
  to ensure clean interop with other projects.

 * `oarphpy/spark` -- Tests `oarphpy` with a vanilla install of
  [PySpark](https://spark.apache.org/) to ensure basic compatibility.

 * `oarphpy/tensorflow` -- Tests `oarphpy` with Tensorflow 1.x to ensure basic
  compatibility (e.g. of `oarphpy.util.tfutil`).

## Development

See `./oarphcli --help` for the development and release workflow.

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

[![Build Status](https://circleci.com/gh/pwais/oarphpy.png)](https://circleci.com/gh/pwais/oarphpy/tree/master)
[![License](http://img.shields.io/:license-apache-orange.svg)](http://www.apache.org/licenses/LICENSE-2.0) 

`OarphPy` is a collection of Python utilities for Data Science with
[PySpark](https://spark.apache.org/docs/latest/api/python/) and
Tensorflow.  Related (but orthogonal) to
[OarphKit](https://github.com/pwais/oarphkit).


# Easy Install

Vanilla, using your own dependencies:
`pip install oarphpy`

TODO: `pip3 install -e ".[utils]"`

Run in a container:
```
docker run -it --net=host tensorflow/tensorflow:1.15.0-py3-jupyter bash
$ apt-get update && apt-get install -y openjdk-8-jdk
$ pip3 install oarphpy
$ oarphpy-cli --test-local
```


# Highlights & Problems Solved

## Flavors of OarphPy

FIXME

* basic python env  
    --> ensure import is safe and basic no-dep tools work.  proves that 
    oarphpy could be at worst benign in user environment
* just spark
    --> enough to make the spark suite of tools run on a vanilla spark
    install.  proves oarphpy could work with use spark
* just tensorflow 
    --> enough to make the tensorflow tools run with a vanilla tensorflow
    install.  proves oarphpy could work with user tensorflow
* full race
    --> all our deps + tensorflow + our own au-based spark (plus alluxio?).
    this environment not only tests all functionality, but also
    can be a useful docker devenv or base devenv.

skip tests if deps not there

FIXME

## Spark

### Automagically ship local code with your PySpark job

TODO demo

### Tensorflow

#### `TFSummaryReader` demo!

### Other

#### Union DataFrames

Unions both the rows and columns of two or more `DataFrames`.  (You'd think
PySpark would include this utility, but they don't, likely arguing that the
user needs to be responsible for schema mismatches).  

```
>>> df1 = spark.createDataFrame([Row(a=1, b=2.0)])
>>> df2 = spark.createDataFrame([Row(a=3, c='foo')])
>>> unioned = Spark.union_dfs(df1, df2)
>>> unioned.show()
+---+----+----+
|  a|   b|   c|
+---+----+----+
|  1| 2.0|null|
|  3|null| foo|
+---+----+----+
```

An improvement from [https://stackoverflow.com/a/40404249](StackOverflow).


## Extras

### `ThruputObserver` for runtime metrics; a less glittery `tqdm`

In many cases, one wants as simple tool to measure the runtime and/or byte
throughput of a key block of code.  Logging and/or printing that information
can facilitate fast debugging, simple batch job monitoring, and discovery
of unknown unkown system constraints, etc.  `tqdm` is great for notebooks;
`ThruputObserver` is meant for longer-running batch jobs.

TODO examples


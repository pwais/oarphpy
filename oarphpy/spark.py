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

import os
import sys
from collections import Counter
from contextlib import contextmanager

from oarphpy import util

################################################################################
### Import Spark
### We're going to expose any import and setup errors HERE, at the time of
### importing `oarphpy.spark`, because:
###  * `pyspark` requires java even to use as a client for a remote cluster;
###       if you simply import pyspark but don't have java, your job will
###       crash with a unhelpful error message.
###  * importing `pyspark`  module may require first monkeying with
###      `sys.environ` (e.g. through `findspark`) or else you get a
###      broken imported pyspark
### So below we try to find Spark / Java, and produce a helpful error
### message otherwise

try:

  # In python3, we filter:
  #  "py4j-0.10.7-src.zip/py4j/java_gateway.py:2020: 
  #      DeprecationWarning: invalid escape sequence \*"
  if sys.version_info.major >= 3:
    import warnings
    warnings.filterwarnings(
      action='ignore',
      message=r'invalid escape sequence')

  # Is pyspark on the PYTHONPATH?
  try:
    import pyspark
  except ModuleNotFoundError as e:
    
    # OK, can findspark find a local install of Spark / Hadoop, e.g.
    # at $SPARK_HOME or /opt/spark ?

    try:
      import findspark
      findspark.init()
    except ImportError:
      # Don't make findspark a hard requirement
      pass

    import pyspark

except Exception as e:
  msg = """
      This portion of OarphPy requires Spark, which in turn requires
      Java 8 or higher.  Mebbe try installing using:
        $ pip install pyspark
      That will fix import errors.  To get Java, try:
        $ apt-get install -y openjdk-11-jdk && echo JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 >> /etc/environment
      If you have spark installed locally (e.g. from source), set $SPARK_HOME
      *** Original error: %s
  """ % (e,)
  raise ImportError(msg)


################################################################################
### General Spark Utils
### 


def num_executors(spark):
  # NB: Not a public API! But likely stable.
  # https://stackoverflow.com/a/42064557
  return spark.sparkContext._jsc.sc().getExecutorMemoryStatus().size()


def for_each_executor(spark, thunk):
  """While Spark does not officially provide an API since the number
  of executors can change through the course of a job, this approach
  leverages the best practices to evaluate `thunk` at most once per
  executor."""
  class LazyFunc(object):
    def __init__(self, th):
      self._thunk = th
    def __call__(self, _):
      if not hasattr(self, '_result'):
        # call body will be executed at most once per executor python process
        import uuid
        self._result = self._thunk()
        self._id = str(uuid.uuid4())
      return (self._id, self._result)
      
  sc = spark.sparkContext
  N = max(1, num_executors(spark))
  rdd = sc.parallelize(list(range(N)), numSlices=N)
  
  id_to_res = dict(rdd.map(LazyFunc(thunk)).collect())
  results = list(id_to_res.values())
    # Return results from only distinct executors
  
  assert len(results) == N
  return results


def cluster_cpu_count(spark):
  """Like `os.cpu_count()` but for an entire Spark cluster.  Useful for
  scalling memory-intensive jobs that use the RDD api."""
  def get_cpu_count():
    import multiprocessing
    return multiprocessing.cpu_count()
  res = for_each_executor(spark, get_cpu_count)
  return sum(res)


def run_callables(spark, callables, parallel=-1):
  import cloudpickle
    # Spark uses regular pickle for RDD data; here we need cloudpickle in
    # order to serialize and run code.
  callable_bytess = [cloudpickle.dumps(c) for c in callables]
  if parallel <= 0:
    parallel = len(callable_bytess)

  rdd = spark.sparkContext.parallelize(callable_bytess, numSlices=parallel)
  def invoke(callable_bytes):
    import cloudpickle
    c = cloudpickle.loads(callable_bytes)
    res = c()
    return callable_bytes, cloudpickle.dumps(res)

  rdd = rdd.map(invoke)
  all_results = [
    (cloudpickle.loads(callable_bytes), cloudpickle.loads(res))
    for callable_bytes, res in rdd.collect()
  ]
  return all_results


def union_dfs(*dfs):
  """Return the union of a sequence DataFrames and attempt to merge
  the schemas of each (i.e. union of all columns).
  Based upon https://stackoverflow.com/a/40404249
  """
  if not dfs:
    return dfs
  
  df = dfs[0]
  for df_other in dfs[1:]:
    left_types = {f.name: f.dataType for f in df.schema}
    right_types = {f.name: f.dataType for f in df_other.schema}
    left_fields = set(
      (f.name, f.dataType, f.nullable) for f in df.schema)
    right_fields = set(
      (f.name, f.dataType, f.nullable) for f in df_other.schema)

    from pyspark.sql.functions import lit

    # First go over `df`-unique fields
    for l_name, l_type, l_nullable in left_fields.difference(right_fields):
      if l_name in right_types:
        r_type = right_types[l_name]
        if l_type != r_type:
          raise TypeError(
            "Union failed. Type conflict on field %s. left type %s, right type %s" % (l_name, l_type, r_type))
        else:
          raise TypeError(
            "Union failed. Nullability conflict on field %s. left nullable %s, right nullable %s"  % (l_name, l_nullable, not(l_nullable)))
      df_other = df_other.withColumn(l_name, lit(None).cast(l_type))

    # Now go over `df_other`-unique fields
    for r_name, r_type, r_nullable in right_fields.difference(left_fields):
      if r_name in left_types:
        l_type = right_types[r_name]
        if r_type != l_type:
          raise TypeError(
            "Union failed. Type conflict on field %s. right type %s, left type %s" % (r_name, r_type, l_type))
        else:
          raise TypeError(
            "Union failed. Nullability conflict on field %s. right nullable %s, left nullable %s" % (r_name, r_nullable, not(r_nullable)))
      df = df.withColumn(r_name, lit(None).cast(r_type))
    df = df.unionByName(df_other)
  return df


def get_balanced_sample(spark_df, col, n_per_category=None, seed=1337):
  """Given a column `col` in `spark_df`, return a *balanced* sample
  (countering class imbalances in `spark_df[col]`).  Optionally limit the
  sample to having up to `n_per_category` examples for every distinct
  categorical value of `spark_df[col]`."""
  from pyspark.sql import functions as F
  category_to_count_df = spark_df.groupBy(col).agg(F.count('*'))
  category_to_count = category_to_count_df.rdd.collectAsMap()
  assert category_to_count

  # We will only sample as many as the rarest category
  numerator = min(category_to_count.values())
  if n_per_category is not None:
    numerator = min(numerator, n_per_category)
  fractions = dict(
    (category, float(numerator) / count)
    for category, count in category_to_count.items()
  )
  return spark_df.sampleBy(col, fractions=fractions, seed=seed)


### Test Utilities (for unit tests and other debugging)

def cluster_get_info(spark):
  """Return a text report showing various details about each worker in the
  cluster."""

  infos = for_each_executor(spark, lambda: util.get_sys_info())
  def format_info(info):
    s = """
      Host: {hostname} {host}
      Egg: {filepath}
      Internet connectivity: {have_internet}
      Num CPUs: {n_cpus}
      Memory:
      {memory}

      PYTHONPATH:
      {PYTHONPATH}

      nvidia-smi:
      {nvidia_smi}

      Disk:
      {disk_free}
      """.format(**info)
    return '\n'.join(l.lstrip() for l in s.split('\n'))
  info_str = '\n\n'.join(format_info(info) for info in infos)
  return info_str


def test_pi(spark):
  """Run the "textbook" Monte Carlo Pi Sampling demo and assert correctness.
  When run on a cluster, this test can help suss out networking issues
  between the master and worker machines."""
  util.log.info("Running PI ...")
  sc = spark.sparkContext
  num_samples = 1000000
  def inside(p):
    import random
    x, y = random.random(), random.random()
    return x*x + y*y < 1
  count = sc.parallelize(list(range(0, num_samples))).filter(inside).count()
  pi = 4 * float(count) / num_samples
  util.log.info("Pi estimate: %s" % pi)
  assert abs(pi - 3.14) < 0.1, "Spark program had an error?"


def _op_test():
  from oarphpy import util
  res = list(util.ichunked([1, 2, 3], 1))
  assert res == [(1,), (2,), (3,)], res


def test_egg(spark, modname='oarphpy', test_egg_contents=_op_test):
  """Test the egg that `SessFactory` (below) dynamically builds and includes
  in spark jobs.  We'll run the function `test_egg_contents` on each worker
  to actually execute code included in the egg.
  """

  # SessFactory below always creates eggs with this name
  EXPECTED_EGG_NAME = modname + '-0.0.0' + _egg_py_suffix()

  def run_test_on_worker():
    # Normally, pytest puts the local source tree on the PYTHONPATH.  That
    # setting gets inherited when Spark forks a python subprocess to run
    # this function.  Remove the source tree from the PYTHONPATH here
    # in order to force pyspark to read from the egg file / SparkFiles.
    # We may safely edit the PYTHONPATH here because this code is run in a
    # child python process that will soon exit.
    import sys
    if '/opt/oparhpy' in sys.path:
      sys.path.remove('/opt/oparhpy')
    if '' in sys.path:
      sys.path.remove('')

    ## Check for the egg, which Spark puts on the PYTHONPATH
    egg_path = ''
    for p in sys.path:
      if EXPECTED_EGG_NAME in p:
        egg_path = p
    assert egg_path, "Egg not found in {}".format(sys.path)

    ## Is the egg any good?
    import zipfile
    f = zipfile.ZipFile(egg_path)
    egg_contents = f.namelist()
    assert any(modname in fname for fname in egg_contents), egg_contents

    ## Use the egg!
    test_egg_contents()
    
    return util.get_sys_info()

  util.log.info("Testing egg ...")
  infos = for_each_executor(spark, run_test_on_worker)
  host_to_syspath = dict((info['host'], info['PYTHONPATH']) for info in infos)
  assert all(EXPECTED_EGG_NAME in p for p in host_to_syspath.values()), \
    "One or more workers missing egg %s: %s" % (
      EXPECTED_EGG_NAME, host_to_syspath)


def test_tensorflow(spark):
  """Remotely test Tensorflow support in the given `spark` cluster"""

  def test_and_get_info():
    import random
    import tensorflow as tf

    tf.compat.v1.disable_v2_behavior()
      # NB: this only impacts the executor Python process
    
    x = int(10 * random.random())
    a = tf.constant(x)
    b = tf.constant(3)

    from oarphpy import util
    sess = util.tf_create_session()
    res = sess.run(a * b)

    assert res == 3 * x
    
    import socket
    info = {
      'hostname': socket.gethostname(),
      'gpu': tf.test.gpu_device_name(),
    }
    return [info]

  util.log.info("Testing Tensorflow ...")
  res = for_each_executor(spark, test_and_get_info)
  util.log.info("... Tensorflow success!  Info:")
  import pprint
  util.log.info('\n\n' + pprint.pformat(res) + '\n\n')



### Counters

try:
  from pyspark.accumulators import AccumulatorParam
  AccumulatorParam_BASE = AccumulatorParam
except:
  class AccumulatorParam_BASE(object):
    # A non-functional dummy when pyspark is not available
    pass

class CounterAccumulator(AccumulatorParam_BASE):
  def zero(self, value):
    return Counter({})
  def addInPlace(self, value1, value2):
    return value1 + value2

def create_counter_accumulator(spark):
  sc = spark.sparkContext
  acc = sc.accumulator(Counter(), CounterAccumulator())
  return acc

class CounterCollection(object):
  def __init__(self, spark, name=''):
    self._acc = create_counter_accumulator(spark)
    self._name = name
  
  def __getitem__(self, key):
    return self._acc.value[key]
  
  def __setitem__(self, key, value):
    self.tally(key, value)
  
  def tally(self, key, value):
    c = Counter()
    c[key] = value
    self._acc += c

  def kv_tally(self, tag, key='', value=0):
    self.tally('__psegs_kv.' + tag + '.key=' + key, value)
  
  def get_kv_tally(self, tag):
    key_prefix = '__psegs_kv.' + tag + '.key='
    return dict(
      (k[len(key_prefix):], v)
      for k, v in self._acc.value.items()
      if k.startswith(key_prefix)
    )

  def __str__(self):
    import pprint

    HEADER = (
      "CounterCollection({name})\n"
      "Spark Accumulator: {acc}\n"
      "Counters:\n"
    ).format(
      name=self._name,
      acc=self._acc.aid,
    )
    kv_tags = set()
    kvs = []
    kv_prefix = '__psegs_kv.'
    for k, v in self._acc.value.items():
      if k.startswith(kv_prefix):
        k = k[len(kv_prefix):]
        tag = k.split('.key=')[0]
        kv_tags.add(tag)
      else:
        kvs.append((k, v))
    for tag in kv_tags:
      kvs.append((tag, self.get_kv_tally(tag)))
    
    BODY = '\n'.join("%s: %s" % (k, pprint.pformat(v)) for k, v in sorted(kvs))
    return "%s\n%s\n" % (HEADER, BODY)
  
  def __repr__(self):
    # For things like pprint that use __repr__ instead of __str__
    return str(self)
  
  @contextmanager
  def log_progress(self, log_func=None, log_freq_sec=10):
    log_func = log_func or util.log.info
    
    import threading
    exit_event = threading.Event()
    def spin_log():
      REPORT_EVERY_SEC = 10
      import time
      start_wait = time.time()
      while not exit_event.is_set():
        if time.time() - start_wait >= log_freq_sec:
          log_func(self)
          start_wait = time.time()
        time.sleep(0.5)
    bkg_th = threading.Thread(target=spin_log, args=())
    bkg_th.daemon = True
    bkg_th.start()

    yield self

    exit_event.set()
    bkg_th.join()

### OarphPy-specific Extras

def archive_rdd(spark, path):
  fws = util.ArchiveFileFlyweight.fws_from(path)
  return spark.sparkContext.parallelize(fws)


################################################################################
### Spark Session Factories
### 

def _egg_py_suffix():
  py_vers = (sys.version_info.major, sys.version_info.minor)
  py_suffix = '-py{}.{}.egg'.format(*py_vers)
  return py_suffix


class SessionFactory(object):
  """This class is a factory for pre-configured `SparkSession` instances.  A
  primary feaure of this factory is that it automagically includes the caller's
  python module as a Spark PyFile, thus effectively shipping the user's python
  project with the Spark job.  This class also helps centralize project-wide
  Spark configuration.  This class is designed as programmatic replacement
  for the `spark-submit` shell script.

  To create and use a session:

  >>> from oarphpy import spark as S
  >>> spark = S.SessionFactory.getOrCreate()
  >>> S.num_executors(spark)
  1

  Or using as a context manager:

  >>> with S.SessionFactory.sess() as spark:
  ...     print(S.num_executors(spark))
  1

  See `LocalK8SSpark` below for an example of how to subclass this factory for
  your own project.  See also `NBSpark` below for an example subclass
  that enables interop with the `sparkmonitor` package for jupyter.
  """

  # Default to local Spark master
  MASTER = None
  
  # Default `SparkConf` instance to use for any new session
  CONF = None
  
  # Default set of `SparkConf` key-value settings to use for any new sesion,
  # e.g. {
  #   'spark.port.maxRetries': '96',
  #       # For local instances with many CPUs, let Spark use tons of ports
  #
  #   'spark.driver.memory': '8g',
  #   'spark.sql.files.maxPartitionBytes': int(8 * 1e6),
  #       # Helps aid in reading larger Parqet datasets
  #
  #   'spark.jars.packages': 'io.delta:delta-core_2.11:0.4.0',
  #       # To enable Delta
  #
  #   'spark.pyspark.python': 'python3',
  #       # To force python3
  #
  #   'spark.python.worker.reuse': False,
  #       # Useful when using Tensorflow on Spark, because TF leaks :P
  # }
  CONF_KV = {}

  # Force all sessions to package and use code in this source root directory.
  # Specify a path to the library dir, i.e. the given path should be a 
  # directory containing an __init__.py file (but a setup.py is not needed).
  # If False-y, we'll try to auto-deduce this path from the calling code.
  SRC_ROOT = None

  # If you have more than one python module in `SRC_ROOT`, provide a list
  # of their names, or the list ['*'] to match all eligible modules (i.e.
  # every subdirectory with an __init__.py).
  SRC_ROOT_MODULES = []
  

  ### Core Features


  #### create_egg() and Support

  @classmethod
  def _resolve_src_root(cls):
    src_root = cls.SRC_ROOT
    if src_root is None:
      try:
        import inspect
        frames = inspect.stack()#[2][0]
        for frame in frames:
          # Ignore frames associated with this class
          if ('oarphpy/spark.py' in frame.filename and 
                hasattr(cls, frame.function)):
            continue
          
          # Ignore user using ::sess() context manager
          if ('contextlib.py' in frame.filename and 
                frame.function == '__enter__'):
            continue
          
          # Ok we might have found the calling user program!
          candidate = os.path.abspath(frame.filename)
          i_am_in_a_module = os.path.exists(
            os.path.join(os.path.dirname(candidate), '__init__.py'))
          if i_am_in_a_module:
            src_root = os.path.abspath(os.path.join(candidate, os.pardir))
            break
        if not src_root:
          raise ValueError("Ran out of candidate stack frames")
      except Exception as e:
        util.log.info(
          "Failed to auto-resolve src root (error: %s) "
          "falling back to %s" % (e, cls.SRC_ROOT))
        src_root = cls.SRC_ROOT
    
    if src_root and src_root.endswith(os.sep):
      src_root = src_root[:-1]
    return src_root

  @classmethod
  def _create_tmp_workdir(cls):
    # Create a working directory for the build and the egg. The spark context
    # dies upon process exit, so we'll keep the temp directory alive for
    # the same duration.
    import atexit
    import shutil
    import tempfile
    path = tempfile.mkdtemp(suffix='_oarphpy_eggbuild')
    atexit.register(lambda: shutil.rmtree(path))
    return path

  @classmethod
  def _create_new_egg(cls, src_root, out_dir):
    assert os.path.exists(src_root)
    assert os.path.exists(out_dir)

    MODNAME = os.path.basename(src_root)
    if sys.version_info.major >= 3:
      # For whatever reason,
      # In py 2.7.x, setuptools wants the path of the python module
      # In py 3.x, setuptools wants the directory containing the python module
      src_root = os.path.dirname(src_root)
    
    util.log.info("Using source root %s " % src_root)

    # Below is a programmatic way to run something like:
    # $ cd /opt/au && python setup.py clean bdist_egg
    # But we don't actually need a setup.py (!)
    # Based upon https://github.com/pypa/setuptools/blob/a94ccbf404a79d56f9b171024dee361de9a948da/setuptools/tests/test_bdist_egg.py#L30
    # See also: 
    # * https://github.com/pypa/setuptools/blob/f52b3b1c976e54df7a70db42bf59ca283412b461/setuptools/dist.py
    # * https://github.com/pypa/setuptools/blob/46af765c49f548523b8212f6e08e1edb12f22ab6/setuptools/tests/test_sdist.py#L123
    # * https://github.com/pypa/setuptools/blob/566f3aadfa112b8d6b9a1ecf5178552f6e0f8c6c/setuptools/__init__.py#L51
    from setuptools.dist import Distribution
    from setuptools import PackageFinder
    MODNAME = MODNAME.replace('-', '_') # setuptools will do it anyways
    
    # By default we only want MODNAME in the egg, but we'll support
    # multiple modules (e.g. both oarphpy and oaprhpy_test).
    include = [MODNAME + '*']
    if cls.SRC_ROOT_MODULES == ['*']:
      include = cls.SRC_ROOT_MODULES
    elif cls.SRC_ROOT_MODULES:
      include = [m + '*' for m in cls.SRC_ROOT_MODULES]

    # We want to confine setuptools to a clean directory because it'll create
    # stateful files and directories like `build/`
    setuptools_workdir = os.path.join(out_dir, 'workdir')
    util.cleandir(setuptools_workdir)
    dist = Distribution(attrs=dict(
        script_name='setup.py',
        script_args=[
          'clean',
          'bdist_egg', 
            '--dist-dir', out_dir,
            '--bdist-dir', setuptools_workdir,
        ],
        name=MODNAME,
        src_root=src_root,
        packages=PackageFinder.find(where=src_root, include=include),
    ))
    util.log.info("Generating egg to %s ..." % out_dir)
    with util.with_cwd(setuptools_workdir):
      with util.quiet():
        dist.parse_command_line()
        dist.run_commands()

    # NB: This approach didn't work so well:
    # Typically we want to give spark the egg from:
    #  $ python setup.py bdist_egg
    # from setuptools.command import bdist_egg
    # cmd = bdist_egg.bdist_egg(
    #                 bdist_dir=os.path.dirname(setup_py_path), editable=True)
    # cmd.run()

    egg_path = os.path.join(out_dir, MODNAME + '-0.0.0' + _egg_py_suffix())
    assert os.path.exists(egg_path), "Can't find {}".format(egg_path)
    util.log.info("... done.  Egg at %s" % egg_path)
    return egg_path

  @classmethod
  def create_egg(cls, force_new=False):
    """Build a Python Egg from the current project and return a path
    to the artifact.  The path may be to a cached, pre-computed egg only
    if not `force_new`.  The 'current project' is either class-defaulted or
    auto-deduced.

    Why an Egg?  `pyspark` supports zipfiles and egg files as Python artifacts.
    One might wish to use a wheel instead of an egg.  See this excellent
    article and repo:
     * https://bytes.grubhub.com/managing-dependencies-and-artifacts-in-pyspark-7641aa89ddb7
     * https://github.com/alekseyig/spark-submit-deps
    
    The drawbacks to using a wheel include:
     * wheels often require native libraries to be installed (e.g. via
        `apt-get`), and those deps are typically best baked into the Spark
        Worker environment (versus installed every job run).
     * The `BdistSpark` example above is actually rather slow, especially
        when Tensorflow is a dependency, and `BdistSpark` must run before
        every job is submitted.
     * Spark treats wheels as zip files and unzips them on every run; this
        unzip operation can be very expensive if the zipfile contains large
        binaries (e.g. tensorflow)
     * Wheels are not yet officially supported:
        https://issues.apache.org/jira/browse/SPARK-6764
    
    In comparison, an Egg provides the main benefits we want (to ship project
    code, often pre-committed code, to workers).
    """

    if force_new or not hasattr(cls, '_cached_egg_path'):
      cls._cached_egg_path = ''
    
    if not cls._cached_egg_path:
      # Lazyily egg-ify `src_root`. NB: We don't delete any previous eggs
      # if `force_new` because some Spark session might still try to read
      # the old file.
      out_dir = cls._create_tmp_workdir()
    
      # Now decide the source root that we'll egg-ify.
      src_root = cls._resolve_src_root()
      if src_root:
        util.log.info("Using source root %s " % src_root)
        cls._cached_egg_path = cls._create_new_egg(src_root, out_dir)
    return cls._cached_egg_path


  ## Primary Public Interface    

  @classmethod
  def getOrCreate(cls):
    """Spark sessions are typically instantiated using the
    `Builder.getOrCreate()` or `SparkSession.getOrCreate()` methods.  This
    method is a drop-in replacement that uses class-specified defaults,
    includes the local python module as a PyFile, etc.

    Returns:
      A `pyspark.sql.session.SparkSession` instance.
    """

    # TODO: take a SparkConf like SparkSession does.

    # Warm the cache; this call surfaces build errors *before* trying
    # to start spark.
    _ = cls.create_egg()

    from pyspark import sql
    builder = sql.SparkSession.builder
    if cls.MASTER is not None:
      builder = builder.master(cls.MASTER)
    elif 'SPARK_MASTER' in os.environ:
      # spark-submit honors this env var
      builder = builder.master(os.environ['SPARK_MASTER'])
    if cls.CONF is not None:
      builder = builder.config(conf=cls.CONF)
    if cls.CONF_KV is not None:
      for k, v in cls.CONF_KV.items():
        builder = builder.config(k, v)

    # if cls.HIVE:
    #   # FIXME see mebbe https://creativedata.atlassian.net/wiki/spaces/SAP/pages/82255289/Pyspark+-+Read+Write+files+from+Hive
    #   # builder = builder.config("hive.metastore.warehouse.dir", '/tmp') 
    #   # builder = builder.config("spark.sql.warehouse.dir", '/tmp')
    #   builder = builder.enableHiveSupport()
    
    try:
      spark = builder.getOrCreate()

    except Exception as e:
      # If the user just did `pip install pyspark`, then they might not
      # have java. Try to provide a helpful error message.

      if hasattr(e, 'args') and e.args:
        if 'Java gateway process exited' in e.args[0]:
          msg = """
            This portion of OarphPy requires Spark, which in turn requires
            Java 8 or higher. Looks like you might be missing Java. To get it,
            try:
              $ apt-get update && apt-get install -y openjdk-8-jdk && echo JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 >> /etc/environment
            Original error: %s
          """ % (e,)
          raise Exception(msg)
      raise

    # spark.sparkContext.setLogLevel('INFO')

    egg_path = cls.create_egg()
    if egg_path:
      spark.sparkContext.addPyFile(egg_path)
    else:
      util.log.info(
          "Could not resolve a source root, skipping auto-egg inclusion.")
    
    return spark
  
  @classmethod
  @contextmanager
  def sess(cls, *args):
    if args and args[0]:
      spark = args[0]
      yield spark
    else:
      spark = cls.getOrCreate()
      yield spark

  @classmethod
  def selftest(cls, modname=None):
    with cls.sess() as spark:
      test_pi(spark)
      if modname:
        test_egg(spark, modname=modname)
    return True

class LocalK8SSpark(SessionFactory):
  """Example of how to subclass the Spark factory above for use with K8S"""
  MASTER = 'k8s://http://127.0.0.1:8001'
  CONF_KV = {
    'spark.kubernetes.container.image': 'my-docker-image', 
  }

  @classmethod
  def getOrCreate(cls):
    if 'spark.driver.host' not in cls.CONF_KV:
      # In practice, we need to set `host` explicitly in order to get the
      # proper driver IP address to the workers.  This choice may break
      # cluster mode where the driver process will run in the cluster
      # instead of locally.  This choice may also break in certain networking
      # setups.  Spark networking is a pain :(
      cls.CONF_KV['spark.driver.host'] = (
        os.environ.get('SPARK_LOCAL_IP', util.get_non_loopback_iface()))

    return super(NBSpark, cls).getOrCreate()
  


# NBSpark is a session builder for local Jupyter notebooks. NBSpark also serves
# as an example of how to subclass the Spark factory above for use with the
# `sparkmonitor` jupyter package.
# (FMI see demo https://krishnan-r.github.io/sparkmonitor/)

# Try to find sparkmonitor's Spark interop jar
try:
  HAVE_SPARK_2 = pyspark.__version__.startswith('2')
  import sparkmonitor
  SPARKMONITOR_HOME = os.path.dirname(sparkmonitor.__file__)
  SPARKMONITOR_JAR_PATH = os.path.join(
        SPARKMONITOR_HOME,
        'listener_2.11.jar' if HAVE_SPARK_2 else 'listener_2.12.jar')
    # E.g. /usr/local/lib/python3.6/dist-packages/sparkmonitor/listener_x.xx.jar
except Exception:
  SPARKMONITOR_JAR_PATH = ''

def _dict_concat(*dicts):
  out = {}
  for d in dicts:
    out.update(**d)
  return out


class NBSpark(SessionFactory):
  """NBSpark is a session builder for local Jupyter notebooks.  Also includes
  support for the `sparkmonitor` jupyter package, see 
  https://krishnan-r.github.io/sparkmonitor/
  """
  
  # Enable support for shipping local modifications to library code without
  # needing to restart the notebook kernel.  This feature will attempt to
  # re-build the user code Egg if there are new local changes and update the
  # Egg in the current Spark session.  Enabling this feature leads to small
  # potential performance degredation; see below.  
  # NB: If you have an RDD or DataFrame that uses Egg data structures or code,
  # you'll need to re-compute that RDD or DataFrame for updated code to take
  # effect; updating the egg does not invalidate any `cache()ed` or
  # `persist()ed` Spark data.
  MAYBE_REBUILD_EGG_EVERY_CELL_RUN = True
  
  # Options to support dynamic Egg updating.  Firstly, `spark.files.overwrite`
  # is needed to accomodate updates to SparkFiles at all; `pyspark` will error
  # on update otherwise.  Secondly, while updating SparkFiles works as
  # expected, updating *Python* modules can lead to 'zipimport malformed zip
  # header' crashes because:
  #  * zipimport caches loaded modules: 
  #      https://github.com/python/cpython/blob/83d3202b92fb4c2fc6df5b035d57f3a1cf715f20/Lib/zipimport.py#L37
  #        Clearing this private global can fix the error, but it must be
  #        cleared immediately before import, which is an undesirable
  #        constraint to impress upon the user.
  #  * pyspark clears importlib caches, but only in the driver:
  #      https://spark.apache.org/docs/2.4.4/api/python/_modules/pyspark/context.html#SparkContext.addPyFile
  # A fix we've found reliable is to disable Python process re-use via
  # `spark.python.worker.reuse` so that the Spark Python worker always has
  # an empty zipimport cache prior to execution.
  REBUILD_EGG_OPTS = {
    'spark.files.overwrite': 'true',
    'spark.python.worker.reuse': 'false',
  }

  # These settings are required as part of standard `sparkmonitor` setup
  # https://github.com/krishnan-r/sparkmonitor/blob/b023845a245010fb7dd9c4be73747f0e5a8c93bd/extension/sparkmonitor/kernelextension.py#L203
  SPARKMONITOR_OPTS = {
    'spark.extraListeners': 'sparkmonitor.listener.JupyterSparkMonitorListener',
    'spark.driver.extraClassPath': SPARKMONITOR_JAR_PATH,
  }

  CONF_KV = _dict_concat(
              REBUILD_EGG_OPTS if MAYBE_REBUILD_EGG_EVERY_CELL_RUN else {},
              SPARKMONITOR_OPTS if SPARKMONITOR_JAR_PATH else {}
  )

  @classmethod
  def getOrCreate(cls):
    spark = super(NBSpark, cls).getOrCreate()
    
    if cls.MAYBE_REBUILD_EGG_EVERY_CELL_RUN:
      def maybe_rebuild_egg(*args):
        egg_path = cls.create_egg()
        egg_time = os.path.getmtime(egg_path)

        src_root = cls._resolve_src_root()
        latest_src_time = os.path.getmtime(src_root)
        for path in util.all_files_recursive(src_root):
          latest_src_time = max(latest_src_time, os.path.getmtime(path))

        if latest_src_time > egg_time:
          util.log.info("Source has changed! Rebuilding Egg ...")
          egg_path = cls.create_egg(force_new=True)
          spark.sparkContext.addPyFile(egg_path) # Updated!

      # Patch into IPython / Jupyter / Google Colab to maybe rebuild the egg
      # every cell run.  NB:
      # * `get_ipython()` is a global provided in any IPython-esque session
      # * Using pre_run_code_hook triggers a UserWarning / deprecation
      #     warning, but the suggested events don't exist ...
      import warnings
      warnings.filterwarnings(
        action='ignore',
        message=r'Hook pre_run_code_hook is deprecated')
      get_ipython().set_hook('pre_run_code_hook', maybe_rebuild_egg)

    return spark



################################################################################
### Spark SQL Type Adaption Utils
###

TENSOR_AUTO_PACK_MIN_KBYTES = 2

class Tensor(object):
  """An ndarray-like object designed to store numpy arrays in Parquet / 
  Spark SQL format.  Spark's DenseVector and Matrix unfortunately don't 
  support arbitrary tensor shape.  Furthermore, `Tensor` stores data in
  an explicit order accessible to external readers such as Eigen in C++
  or nd4j / BLAS wrappers in Java.
  """
  __slots__ = ('shape', 'dtype', 'order', 'values', 'values_packed')

  @staticmethod
  def from_numpy(arr):
    t = Tensor()
    t.shape = list(arr.shape)
    t.dtype = arr.dtype.name
    t.order = 'C' # C-style row-major

    if arr.nbytes >= TENSOR_AUTO_PACK_MIN_KBYTES * (2**10):
      t.values = [arr.dtype.type(0)]
        # Need a non-empty array for type deduction
      t.values_packed = bytearray(arr.tobytes(order='C'))
    else:
      t.values = arr.flatten(order='C').tolist()
      t.values_packed = bytearray()
    return t
  
  @staticmethod
  def to_numpy(t):
    import numpy as np
    if t.values_packed:
      return np.reshape(
        np.frombuffer(t.values_packed, dtype=np.dtype(t.dtype)),
        t.shape)
    else:
      return np.array(
              np.reshape(t.values, t.shape, order=t.order),
              dtype=np.dtype(t.dtype))


class CloudpickeledCallableData(object):
    __slots__ = ('func_bytes', 'func_pyclass')
    def __init__(self, **kwargs):
      for k in self.__slots__:
        setattr(self, k, kwargs.get(k))

class CloudpickeledCallable(object):
  """Wraps callable objects (e.g. functions, including lambdas) and 
  uses `cloudpickle` for serialization.  Spark uses `cloudpickle` for 
  serializing _tasks_ (e.g. map functions) but uses `pickle` for 
  serializing _data_.  In particular, data in a Spark RDD or DataFrame
  must be pickleable.  `CloudpickeledCallable` provides a wrapper
  so that you can embed Python functions as *data* in RDDs, DataFrames,
  and other forms of data at rest (e.g. pickle files or Parquet data).

  Note that `cloudpickle` can be selective about how much of the object tree
  that it serializes; some imports and globals may get ignored.  When you
  deserialize and invoke a `CloudpickeledCallable`, your interpreter should
  have the same (or similar) code as that used when serializing the callable,
  otherwise behavior may be difficult to predict.

  Note that `cloudpickle` can't handle non-serializable data like thread local
  variables, mutices, etc.  `CloudpickeledCallable` won't work for all code.

  `CloudpickeledCallable` is useful for embedding flyweights in your dataset.
  (FMI see <https://en.wikipedia.org/wiki/Flyweight_pattern> )
  For example:

  >>> def load_matrix(path):
  >>>   import numpy as np
  >>>   return np.loadtxt(path)
  >>> my_db_row = {
  >>>     'path': 'path/to/data.txt',
  >>>     'factory': 
  >>>        CloudpickeledCallable(lambda: load_matrix('path/to/data.txt'))
  >>> }
  >>> import pickle
  >>> pickle.dump(my_db_row, open('dumped.pkl', 'wb'))

  Now if you deserialize `my_db_row` from disk and run `my_db_row['factory']()`,
  your `load_matrix()` helper will get invoked with the embedded path.  Thus
  your `my_db_row` is a flyweight for the data in 'path/to/data.txt'.
  """

  __slots__ = ('_func', '_func_pyclass')

  @staticmethod
  def _get_func_name(func):
    module = '<unknown_module>'
    if hasattr(func, '__module__'):
      module = func.__module__
    if hasattr(func, '__name__'):
      return '%s.%s' % (module, func.__name__)
    else:
      import inspect
      src = inspect.getsource(func)
      lambda_varname = inspect.getsource(func).split('=')[0].strip()
      return '%s.%s' % (module, lambda_varname)

  def __init__(self, func=None):
    self._func = func
    if func is None:
      self._func_pyclass = '(empty CloudpickeledCallable)'
    else:
      self._func_pyclass = CloudpickeledCallable._get_func_name(func)
  
  @classmethod
  def empty(cls):
    return cls()
  
  def __call__(self, *args, **kwargs):
    assert self._func is not None, \
      "This CloudpickeledCallable is the null CloudpickeledCallable"
    return self._func(*args, **kwargs)

  @classmethod
  def to_cc_data(cls, cc):
    if cc == cls.empty():
      func_bytes = bytearray()
    else:
      import cloudpickle
      func_bytes = bytearray(cloudpickle.dumps(cc._func))
    
    return CloudpickeledCallableData(
              func_bytes=func_bytes,
              func_pyclass=cc._func_pyclass)

  @classmethod
  def from_cc_data(cls, ccd):
    if len(ccd.func_bytes) > 0:
      import cloudpickle
      func = cloudpickle.loads(ccd.func_bytes)
    else:
      func = None
    cc = cls(func=func)
    cc._func_pyclass = ccd.func_pyclass
    return cc

  def __getstate__(self):
    return (self.to_cc_data(self),)

  def __setstate__(self, d):
    cc = self.from_cc_data(d[0])
    self._func = cc._func
    self._func_pyclass = cc._func_pyclass

  def __eq__(self, other):
    return self._func == other._func

  def __repr__(self):
    return "CloudpickeledCallable(_func_pyclass=%s)" % self._func_pyclass
  

class RowAdapter(object):
  """Transforms between custom objects and `pyspark.sql.Row`s used in Spark SQL
  or Parquet files. Use to encode numpy arrays and standard Python objects
  with a transparent Parquet schema that is accessible to other readers.

  Usage:
   * Use `RowAdapter.to_row()` in place of the `pyspark.sql.Row` constructor.
   * Call `RowAdapter.from_row()` on any `pyspark.sql.Row` instance, e.g.
       within an `RDD.map()` call or after a `DataFrame.collect()` call.
   * Decoding requires Python objects to have an available zero-arg __init__()
  
  Unfortunately, we can't use Spark's UDT API to embed this adapter (and 
  obviate user calls) because UDTs require schema definitions.  Furthermore,
  Spark <=2.x could not handle UDTs nested in maps or lists; i.e.
  [UDT()] (i.e. a list of UDTs) and {'foo': UDT()} (i.e. a map with UDT values)
  would cause Spark to crash.  Moreover, for ndarray data, Spark's ml.linalg
  package coerces all data to floats.

  Benefits of RowAdapter:
    * Transparently handles numpy arrays and numpy boxed scalar types
        (e.g. np.float32).
    * Deep type adaptation; supports nested types.
    * At the decode stage, supports evolution of object types independent of
        the schema of data at rest:
          - Added object fields don't get set unless there's a recorded value
          - Removed object fields will get ignored
          - NB: Fields that change type will get set with the data at rest;
              if you need to change type, consider adding a new field.
    * Handles slotted Python objects (which Spark currently does not support),
        as well as un-slotted objects (where Spark supports automatic encoding
        but not decoding).
    * Enables saving objects and numpy arrays to Parquet in a format accessible
        to external systems (no additional SERDES library required)
    * Uses cloudpickle to serialize `CloudpickeledCallable`-wrapped functions.
  """

  IGNORE_PROTECTED = False
  IGNORE_PRIVATE = True

  @staticmethod
  def _get_classname_from_obj(o):
    # Based upon https://stackoverflow.com/a/2020083
    module = o.__class__.__module__
    # NB: __module__ might be null
    if module is None or module == str.__class__.__module__:
      return o.__class__.__name__  # skip "__builtin__"
    else:
      return module + '.' + o.__class__.__name__

  @staticmethod
  def _get_class_from_path(path):
    # Pydoc is a bit safer and more robust than anything we can write
    import pydoc
    obj_cls, obj_name = pydoc.resolve(path)
    assert obj_cls
    return obj_cls

  @classmethod
  def to_schema(cls, obj):
    row = cls.to_row(obj)
    import pyspark.sql.types as pst
    return pst._infer_schema(row)

  @classmethod
  def to_row(cls, obj):
    from pyspark.sql import Row
    import numpy as np
    if isinstance(obj, Row):
      # Row is immutable, so we have to recreate
      row_dict = obj.asDict()
      return Row(**cls.to_row(row_dict))
    elif isinstance(obj, np.ndarray):
      return cls.to_row(Tensor.from_numpy(obj))
    elif isinstance(obj, CloudpickeledCallable):
      return cls.to_row(CloudpickeledCallable.to_cc_data(obj))
    elif isinstance(obj, np.generic):
      # Those pesky boxed scalars like np.float32
      return obj.item()
    elif hasattr(obj, '__slots__') or hasattr(obj, '__dict__'):
      def is_hidden(fname):
        # Check private first to disambiguate `_` vs `__` prefixes
        if cls.IGNORE_PRIVATE and fname.startswith('__'):
          return True
        if cls.IGNORE_PROTECTED and fname.startswith('_'):
          return True
        return False
      tag = ('__pyclass__', RowAdapter._get_classname_from_obj(obj))
      if hasattr(obj, '__slots__'):
        obj_attrs = [
          (k, cls.to_row(getattr(obj, k)))
          for k in obj.__slots__
          if not is_hidden(k)
        ]
      else:
        obj_attrs = [
          (k, cls.to_row(v))
          for k, v in obj.__dict__.items()
          if not is_hidden(k)
        ]
      return Row(**dict([tag] + obj_attrs))
    elif isinstance(obj, list):
      return [cls.to_row(x) for x in obj]
    elif isinstance(obj, tuple):
      return tuple(cls.to_row(x) for x in obj)
        # Spark will typically transform to list
    elif isinstance(obj, dict):
      return dict((k, cls.to_row(v)) for k, v in obj.items())
    else:
      return obj
  
  @classmethod
  def from_row(cls, row):
    if hasattr(row, '__fields__'):
      # Probably a pyspark Row instance; try to convert it to an object
      if '__pyclass__' in row.__fields__:
        obj_cls_name = row['__pyclass__']
        obj_cls = RowAdapter._get_class_from_path(obj_cls_name)
        obj = obj_cls.__new__(obj_cls)

        attr_to_default = {}
        if hasattr(obj, '__attrs_attrs__'):
          # In the case that an attrs-based class has now added an attribute
          # since being row-ified, the row will be missing a value for that
          # new attribute.  Use the attrs-specified default if available.
          attr_to_default = dict((a.name, a.default) for a in obj.__attrs_attrs__)
        
        if hasattr(obj, '__slots__'):
          for k in obj.__slots__:
            if k in row:
              setattr(obj, k, cls.from_row(row[k]))
            elif k in attr_to_default:
              setattr(obj, k, attr_to_default[k])
        elif hasattr(obj, '__dict__'):
          for k, v in row.asDict().items():
            if k == '__pyclass__':
              continue
            obj.__dict__[k] = cls.from_row(v)
        else:
          raise ValueError(
              "Object %s no longer has __slots__ nor __dict__" % obj_cls_name)

        if hasattr(obj, '__attrs_init__'):
          obj.__attrs_post_init__()
          
        if isinstance(obj, Tensor):
          obj = Tensor.to_numpy(obj)
        elif isinstance(obj, CloudpickeledCallableData):
          obj = CloudpickeledCallable.from_cc_data(obj)
        return obj
      
      else:
        # No known __pyclass__, so fall back to generic
        from pyspark.sql import Row
        attrs = dict((k, cls.from_row(v)) for k, v in row.asDict().items())
        return Row(**attrs)
    elif isinstance(row, list):
      return [cls.from_row(x) for x in row]
    elif isinstance(row, dict):
      return dict((k, cls.from_row(v)) for k, v in row.items())
    return row

  
################################################################################
### Spark-Tensorflow Utils
###

def spark_df_to_tf_dataset(
      spark_df,
      shard_col,
      spark_row_to_tf_element, # E.g. lambda r: (np.array[0],),
      tf_element_types, # E.g. [tf.int64]
      tf_output_shapes=None,
      non_deterministic_element_order=True,
      num_reader_threads=-1,
      logging_name='spark_tf_dataset'):
    """Create a tf.data.Dataset that reads from the Spark Dataframe
    `spark_df`.  Executes parallel reads using the Tensorflow's internal
    (native code) threadpool.  Each thread reads a single Spark partition
    at a time.

    This utility is similar to Petastorm's `make_reader()` but is far simpler
    and leverages Tensorflow's build-in threadpool (so we let Tensorflow
    do the read scheduling).  Status: alpha-quality; some perf quirks.

    Args:
      spark_df (pyspark.sql.DataFrame): Read from this Dataframe.
      shard_col (str): Implicly shard the dataset using this column; read
        one shard per reader thread at a time to conserve memory.
      spark_row_to_tf_element (func): 
        Use this function to map each pyspark.sql.Row in `spark_df`
        to a tuple that represents a single element of the
        induced TF Dataset.
      tf_element_types (tuple):
        The types of the elements that `spark_row_to_tf_element` returns;
        e.g. (tf.float32, tf.string).
      tf_output_shapes (tuple):
        Optionally specify the shape of the output of `spark_row_to_tf_element`;
        e.g. (tf.TensorShape([]), tf.TensorShape([None])) (where the former
        return element is a single scalar and the latter is a list)
      non_deterministic_element_order (bool):
        Allow the resulting tf.data.Dataset to have elements in
        non-deterministic order for speed gains.
      num_reader_threads (int):
        Tell Tensorflow to use this many reader threads, or use -1
        to provision one reader thread per CPU core.
      logging_name (str):
        Log progress under this name.
    
    Returns:
      tf.data.Dataset: The induced TF Datset with one element per
        row in `spark_df`.
    """
    if num_reader_threads < 1:
      import multiprocessing
      num_reader_threads = max(1, int(0.5 * multiprocessing.cpu_count()))
        # NB: Tensorflow prefetch appears to launch twice as many threads
        # as you ask for :P
    df = spark_df

    # Each Tensorflow reader thread will read a single Spark partition
    # Breadcrumbs: We tried using the built-in spark partition id, but
    # Spark appears to want to re-compute this for every partition read,
    # leading to an O(n^2) overhead cost (for `n` partitions).
    # df = spark_df.withColumn('shard_col', spark_partition_id())
    
    util.log.info("Getting shards ...")
    pids = df.select(shard_col).distinct().rdd.flatMap(lambda x: x).collect()
    util.log.info("... found %s shards." % (len(pids),))


    import threading
    import tensorflow as tf
    pid_ds = tf.data.Dataset.from_tensor_slices(pids)
    
    class PartitionToRows(object):
      def __init__(self):
        self.overall_thruput = util.ThruputObserver(
                                    name=logging_name,
                                    log_on_del=True,
                                    n_total=df.count())
                                       # TODO: count() can be slow
        self.overall_thruput.start_block()
        self.lock = threading.Lock()
      
      def __call__(self, pid):
        # Convert pesky numpy boxed numeric types if needed
        import numpy as np
        if isinstance(pid, np.generic):
          pid = pid.item()

        part_df = df.filter(df[shard_col] == pid)
        part_rdd = part_df.rdd.repartition(100)
        rows = part_rdd.map(spark_row_to_tf_element).toLocalIterator()
        util.log.info("Reading partition %s " % pid)
        t = util.ThruputObserver(name='Partition %s' % pid, log_on_del=True)
        t.start_block()
        for row in rows:
          yield row
          t.update_tallies(n=1, num_bytes=util.get_size_of_deep(row))
        t.stop_block()
        util.log.info("Done reading partition %s, stats:\n %s" % (pid, t))
        with self.lock:
          # Since partitions are read in parallel, we need to maintain
          # independent timing stats for the main thread
          self.overall_thruput.stop_block(n=t.n, num_bytes=t.num_bytes)
          self.overall_thruput.maybe_log_progress(every_n=1)
          self.overall_thruput.start_block()

    ds = pid_ds.interleave(
       lambda pid_t: \
         tf.data.Dataset.from_generator(
           PartitionToRows(), 
           args=(pid_t,),
           output_types=tf_element_types,
           output_shapes=tf_output_shapes),
       cycle_length=num_reader_threads,
       num_parallel_calls=num_reader_threads)
    
    # Breadcrumbs: alternative TF API
    # ds = pid_ds.apply(
    #  tf.compat.v2.data.experimental.parallel_interleave(
    #    lambda pid_t: 
    #      tf.data.Dataset.from_generator(
    #        get_rows, 
    #        args=(pid_t,),
    #        output_types=tf_element_types),
    #  cycle_length=num_reader_threads,
    #  sloppy=non_deterministic_element_order))
    
    return ds





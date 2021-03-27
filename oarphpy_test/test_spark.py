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
import unittest
import sys

import pytest

from oarphpy_test import testutil
from oarphpy_test.testutil import LocalSpark
from oarphpy_test.testutil import skip_if_no_spark


@skip_if_no_spark
def test_union_dfs():
  from oarphpy import spark as S
  with testutil.LocalSpark.sess() as spark:
    from pyspark.sql import Row
    df1 = spark.createDataFrame([Row(a=1, b=2.0)])
    df2 = spark.createDataFrame([Row(a=3, c='foo')])

    unioned = S.union_dfs(df1, df2)
    
    EXPECTED = """
         a    b     c
      0  1  2.0  None
      1  3  NaN   foo
    """
    actual = str(unioned.toPandas())
    def to_lines(str):
      lines = [line.strip() for line in str.split('\n')]
      return [l for l in lines if l]
    assert to_lines(EXPECTED) == to_lines(actual)


@skip_if_no_spark
def test_spark_selftest():
  assert testutil.LocalSpark.selftest(modname='oarphpy_test')


@skip_if_no_spark
def test_spark_with_custom_library_script():
  # We must run this test in a subprocess in order for it to have an isolated
  # Spark session
  SCRIPT_PATH = testutil.get_fixture_path('test_spark_with_custom_library.py')
  
  from oarphpy import util
  out = util.run_cmd(sys.executable + ' ' + SCRIPT_PATH, collect=True)
  assert b'test_spark_with_custom_library success!' in out


@skip_if_no_spark
def test_spark_script_in_standalone_lib():
  # We must run this test in a subprocess in order for it to have an isolated
  # Spark session
  SCRIPT_PATH = testutil.get_fixture_path(
    'test_spark_script_in_standalone_lib.py')
  
  from oarphpy import util
  out = util.run_cmd(sys.executable + ' ' + SCRIPT_PATH, collect=True)
  assert b'test_spark_script_in_standalone_lib success!' in out


@skip_if_no_spark
def test_spark_with_custom_library_in_notebook():
  pytest.importorskip("jupyter")

  import re
  from oarphpy import util
  
  NB_PATH = testutil.get_fixture_path(
    'test_spark_ships_custom_library_in_notebook.ipynb')

  # We need to fork off nbconvert to run the test
  TEST_CMD = """
    cd /tmp && 
    jupyter-nbconvert --ExecutePreprocessor.timeout=3600 \
      --to notebook --execute --output /tmp/out \
      {notebook_path}
  """.format(notebook_path=NB_PATH)
  out = util.run_cmd(TEST_CMD, collect=True)
  assert re.search('Writing .* bytes to /tmp/out.ipynb', out.decode())


@skip_if_no_spark
def test_cluster_get_info_smoke():
  with testutil.LocalSpark.sess() as spark:
    from oarphpy import spark as S
    S.cluster_get_info(spark)


@skip_if_no_spark
def test_pi():
  with testutil.LocalSpark.sess() as spark:
    from oarphpy import spark as S
    S.test_pi(spark)


@skip_if_no_spark
def test_spark_tensorflow():
  tf = pytest.importorskip("tensorflow")
  from oarphpy import spark as S
  with testutil.LocalSpark.sess() as spark:
    S.test_tensorflow(spark)


@skip_if_no_spark
def test_spark_cpu_count():
  from oarphpy import spark as S
  import multiprocessing
  with testutil.LocalSpark.sess() as spark:
    assert multiprocessing.cpu_count() == S.cluster_cpu_count(spark)


@skip_if_no_spark
def test_spark_counters():
  from oarphpy import spark as S
  with testutil.LocalSpark.sess() as spark:
    counters = S.CounterCollection(spark)

    def run_counter(x, counters):
      
      # counters['iadd'] += 1
      counters['itally'] = 1
      counters.tally('my_tally', 2)
      counters.kv_tally('my_hist', key=str(x), value=1)

      return x
    
    task_rdd = spark.sparkContext.parallelize(range(10))

    with counters.log_progress():
      f = lambda x: run_counter(x, counters)
      final = task_rdd.map(f).sum()
      assert final == sum(range(10))
    
    assert counters['itally'] == 10
    assert "itally: 10" in str(counters)
    assert counters['my_tally'] == 20
    assert "my_tally: 20" in str(counters)
    assert \
      counters.get_kv_tally('my_hist') == dict((str(k), 1) for k in range(10))
    assert "my_hist: {" in str(counters)


@skip_if_no_spark
class TestArchiveRDD(unittest.TestCase):

  def _check_rdd(self, fw_rdd, fixture_strs):
    names = fw_rdd.map(lambda fw: fw.name).collect()
    expected_names = [s.decode('utf-8') for s in fixture_strs]
    assert sorted(names) == sorted(expected_names)

    values = fw_rdd.map(lambda fw: fw.data).collect()
    assert sorted(values) == sorted(fixture_strs)

  def test_archive_rdd_zip(self):
    TEST_TEMPDIR = testutil.test_tempdir('test_archive_rdd_zip')
    fixture_path = os.path.join(TEST_TEMPDIR, 'test.zip')
      
    # Create the fixture:
    # test.zip
    #  |- foo: "foo"
    #  |- bar: "bar"
    # ... an archive with a few files, where each file contains just a string
    #   that matches the name of the file in the archive
    ss = [b'foo', b'bar', b'baz']
      
    import zipfile
    with zipfile.ZipFile(fixture_path, mode='w') as z:
      for s in ss:
        z.writestr(s.decode('utf-8'), s)
      
    # Test Reading!
    with testutil.LocalSpark.sess() as spark:
      from oarphpy import spark as S
      fw_rdd = S.archive_rdd(spark, fixture_path)
      self._check_rdd(fw_rdd, ss)

  def test_archive_rdd_tar(self):
    TEST_TEMPDIR = testutil.test_tempdir('test_archive_rdd_tar')
    fixture_path = os.path.join(TEST_TEMPDIR, 'test.tar')

    # Create the fixture:
    # test.tar
    #  |- foo: "foo"
    #  |- bar: "bar"
    # ... an archive with a few files, where each file contains just a string
    #   that matches the name of the file in the archive
    ss = [b'foo', b'bar', b'baz']

    import tarfile
    with tarfile.open(fixture_path, mode='w') as t:
      for s in ss:
        from io import BytesIO
        buf = BytesIO()
        buf.write(s)
        buf.seek(0)

        buf.seek(0, os.SEEK_END)
        buf_len = buf.tell()
        buf.seek(0)
        
        info = tarfile.TarInfo(name=s.decode('utf-8'))
        info.size = buf_len
        
        t.addfile(tarinfo=info, fileobj=buf)
      
    # Test Reading!
    with testutil.LocalSpark.sess() as spark:
      from oarphpy import spark as S
      fw_rdd = S.archive_rdd(spark, fixture_path)
      self._check_rdd(fw_rdd, ss)


@skip_if_no_spark
def test_get_balanced_sample():
  from oarphpy import spark as S
  from pyspark.sql import Row
  
  VAL_TO_COUNT = {
    'a': 10,
    'b': 100,
    'c': 1000,
  }
  rows = []
  for val, count in VAL_TO_COUNT.items():
    for _ in range(count):
      i = len(rows)
      rows.append(Row(id=i, val=val))
  
  def _get_category_to_count(df):
    from collections import defaultdict
    rows = df.collect()

    category_to_count = defaultdict(int)
    for row in rows:
      category_to_count[row.val] += 1
    return category_to_count

  def check_sample_in_expectation(df, n_per_category, expected):
    import numpy as np
    import pandas as pd
    
    # NB: even with a fixed seed below, this test's behavior is dependent
    # on the number of system CPUs since Spark draws random numbers
    # concurrently.  To simulate a low-cpu system, use a Spark master local[1].
    # We do more trials to accomodate low-cpu systems (e.g. CircleCI).
    N_TRIALS = 30
    rows = [
      _get_category_to_count(
        S.get_balanced_sample(
          df, 'val',
          n_per_category=n_per_category,
          seed=100*s))
      for s in range(N_TRIALS)
    ]
    pdf = pd.DataFrame(rows)
    pdf = pdf.fillna(0)

    ks = sorted(expected.keys())
    mu = pdf.mean()
    actual_arr = np.array([mu[k] for k in ks])
    expected_arr = np.array([expected[k] for k in ks])
    
    import numpy.testing as npt
    npt.assert_allclose(actual_arr, expected_arr, rtol=0.3)
      # NB: We can only test to about 30% accuracy with this few samples

  with testutil.LocalSpark.sess() as spark:
    df = spark.createDataFrame(rows)

    check_sample_in_expectation(
      df, n_per_category=1, expected={'a': 1, 'b': 1, 'c': 1})

    check_sample_in_expectation(
      df, n_per_category=10, expected={'a': 10, 'b': 10, 'c': 10})

    check_sample_in_expectation(
      df, n_per_category=20, expected={'a': 10, 'b': 10, 'c': 10})

    check_sample_in_expectation(
      df, n_per_category=None, expected={'a': 10, 'b': 10, 'c': 10})


@skip_if_no_spark
def test_spark_df_to_tf_dataset():
  pytest.importorskip("tensorflow")

  from oarphpy.spark import spark_df_to_tf_dataset
  with testutil.LocalSpark.sess() as spark:

    import numpy as np
    import tensorflow as tf
    from pyspark.sql import Row

    def tf_dataset_to_list(ds):
      from oarphpy import util
      with util.tf_data_session(ds) as (sess, iter_dataset):
        return list(iter_dataset())

    df = spark.createDataFrame([
      Row(id='r1', x=1, y=[3., 4., 5.]),
      Row(id='r2', x=2, y=[6.]),
      Row(id='r3', x=3, y=[7., 8., 9.]),
    ])

    # Test empty
    ds = spark_df_to_tf_dataset(
            df.filter('x == False'), # Empty!,
            'id',
            spark_row_to_tf_element=lambda r: ('test',),
            tf_element_types=(tf.string,))
    assert tf_dataset_to_list(ds) == []

    # Test simple
    ds = spark_df_to_tf_dataset(
            df,
            'id',
            spark_row_to_tf_element=lambda r: (r.x,),
            tf_element_types=(tf.int64,))
    assert sorted(tf_dataset_to_list(ds)) == [(1,), (2,), (3,)]

    # Test Complex
    ds = spark_df_to_tf_dataset(
            df,
            'id',
            spark_row_to_tf_element=lambda r: (r.x, r.id, r.y),
            tf_element_types=(tf.int64, tf.string, tf.float64))
    expected = [
      (1, b'r1', np.array([3., 4., 5.])),
      (2, b'r2', np.array([6.])),
      (3, b'r3', np.array([7., 8., 9.])),
    ]
    items = list(zip(sorted(tf_dataset_to_list(ds)), sorted(expected)))
    for actual, exp in items:
      print('actual', actual,'exp', exp)
      assert len(actual) == len(exp)
      for i in range(len(actual)):
        np.testing.assert_array_equal(actual[i], exp[i])
    


################################################################################
### Test CloudpickeledCallable

def expensive_func():
  return "expensive_func"

def ExpensiveCallable(object):
  def __call__(self, x, y=5):
    return "ExpensiveCallable(%s, y=%s)" % (x, y)

@skip_if_no_spark
class TestCloudpickeledCallable(unittest.TestCase):

  def test_yay(self):
    from oarphpy.spark import CloudpickeledCallable
    f = CloudpickeledCallable(expensive_func)
    assert f() == expensive_func()
    assert str(f) == 'CloudpickeledCallable(_func_pyclass=oarphpy_test.test_spark.expensive_func)'

    cl = CloudpickeledCallable(lambda x: x * x)
    assert cl(2) == 4
    assert str(cl) == 'CloudpickeledCallable(_func_pyclass=oarphpy_test.test_spark.<lambda>)'

    import pickle
    b = pickle.dumps(f)

    decoded = pickle.loads(b)
    assert decoded() == expensive_func()
    assert str(decoded) == 'CloudpickeledCallable(_func_pyclass=oarphpy_test.test_spark.expensive_func)'
    assert decoded == f


################################################################################
### Test RowAdapter

## NB: these classes must be declared package-level (rather than test-scoped)
## for cloudpickle / Spark / RowAdapter to discover them properly

class Slotted(object):
  __slots__ = ('foo', 'bar', '_not_hidden')
  
  def __init__(self, **kwargs):
    # NB: ctor for convenience; SparkTypeAdapter does not require it
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k))
  
  def __repr__(self):
    # For unit tests below
    return "Slotted(%s)" % ([(k, getattr(self, k)) for k in self.__slots__],)


class Unslotted(object):
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)
  
  def __repr__(self):
    # For unit tests below
    # Hide any 'hidden' attributes since `RowAdapter` will elide them
    attrs = dict(
              (k, v) for k, v in self.__dict__.items()
              if not k.startswith('__')
    )
    return "Unslotted(%s)" % str(sorted(attrs.items()))
  
  def __eq__(self, other):
    # Just for testing
    return repr(self) == repr(other)

# Tests for attrs-based classes are optional
try:
  import attr
  import numpy as np
  
  # NB: Need these classes defined package-level
  
  @attr.s(eq=True)
  class AttrsUnslotted(object):
    foo = attr.ib(default="moof")
    bar = attr.ib(default=5)
  
  @attr.s(slots=True, eq=True)
  class AttrsSlotted(object):
    foo = attr.ib(default="moof")
    bar = attr.ib(default=5)

except ImportError:
  pass

# Tests for dataclasses are optional
if sys.version_info[0] >= 3:
  from oarphpy_test.dataclass_obj import *

# We use Row a lot in the tests below; tests are skipped w/out Spark
try:
  from pyspark.sql import Row
except Exception:
  pass


def _select_distinct(df, col):
  return list(set(r[0] for r in df.select(col).collect()))


@skip_if_no_spark
class TestRowAdapter(unittest.TestCase):

  ## Tests

  def test_nonadapted_input(self):
    from oarphpy.spark import RowAdapter

    # RowAdapter leaves bare data input unchanged
    BARE_VALUES = True, 1, 1.0, "moof", bytes(b"moof")
    for datum in BARE_VALUES:
      assert RowAdapter.to_row(datum) == datum
      assert RowAdapter.from_row(datum) == datum


  def test_pesky_numpy(self):
    import numpy as np
    from oarphpy.spark import RowAdapter

    # RowAdapter translates pesky numpy-boxed numbers ...
    assert RowAdapter.to_row(Row(x=np.float32(1.))) == Row(x=1.)

    # ... but only one way! In practice, just don't save boxed numbers in rows.
    assert RowAdapter.from_row(Row(x=np.float32(1.))) == Row(x=np.float32(1.))


  def test_python_basic_values(self):

    ## Columns with basic python types get translated as-is; FMI see Spary's
    ## core type mappings: https://github.com/apache/spark/blob/master/python/pyspark/sql/types.py#L875
    row_expected_schema = [
      (Row(x=True),                 [('x', 'boolean')]),
      (Row(x=1),                    [('x', 'bigint')]),
      (Row(x=1.0),                  [('x', 'double')]),
      (Row(x="moof"),               [('x', 'string')]),
      (Row(x=bytearray(b"moof")),   [('x', 'binary')]),
      (Row(x=None),                 [('x', 'null')]),
    ]
    
    self._check_raw_adaption([(r, r) for r, s in row_expected_schema])
    
    for row, expected_schema in row_expected_schema:
      schema = self._check_schema([row], expected_schema)
      if all(t != 'null' for colname, t in expected_schema):
        # In most cases, data can be written as expected
        self._check_serialization([row])
      else:
        # NB: None / null can't be written
        with pytest.raises(Exception) as excinfo:
          self._check_serialization([row], schema=schema)
        assert ("Parquet data source does not support null data type" 
          in str(excinfo.value))


  def test_python_basic_containers_adaption(self):
    
    ## RowAdapters translates basic python containers recursively 
    rows = [
      Row(x=[]),
      Row(x=tuple()),
      Row(x={}),
      
      Row(x=[1, 2, 3]),
      Row(x=(True, False)),
      Row(x={'a': 1, 'b': 2}),

      Row(x=[{'a': 1}, {'c': 2}]),

      Row(x=[Row(a={'b': Row(c=None)})]),
        # NB: as we'll see below, RowAdapter translates
        # `dict` as "map" and `Row` as "struct"
    ]
    self._check_raw_adaption([(r, r) for r in rows])


  def test_python_basic_empty_containers(self):

    # NOTE!!  Empty containers can't have their types auto-deduced!!
    rows = [
      Row(id=0, l=[]),
      Row(id=1, l=[]),
    ]
    with pytest.raises(ValueError) as excinfo:
      self._check_serialization(rows)
    assert "Some of types cannot be determined" in str(excinfo.value)
    
    # RowAdapter can deduce the type, but `array<null>` can't be serialized
    # (same with map<null, null>)
    schema = self._check_schema(
                      rows,
                      [('id', 'bigint'), ('l', 'array<null>')])
    with pytest.raises(Exception) as excinfo:
      self._check_serialization(rows, schema=schema)
    assert ("Parquet data source does not support array<null> data type" 
      in str(excinfo.value))

    # WORKAROUND: If you compute a schema based upon a prototype row, then you
    # can use that schema to write empty containers
    PROTOTYPE_ROW = Row(id=0, l=[0])
    schema = self._check_schema(
                      [PROTOTYPE_ROW],
                      [('id', 'bigint'), ('l', 'array<bigint>')])
    self._check_serialization(rows, schema=schema)


  def test_python_basic_nonempty_containers(self):
    
    # Data with non-empty containers enjoys auto schema deduction from Spark
    rows = [
      Row(id=0, l=[1],    d={'a': 'foo'}),
      Row(id=1, l=[2, 3], d={'b': 'bar'}),
      Row(id=2, l=[],     d={}),    # Some rows with empty containers are OK
      Row(id=3, l=None,   d=None)   # Null data is OK too
    ]
    self._check_serialization(rows)
    self._check_schema(
            rows,
            [('id', 'bigint'),
             ('l', 'array<bigint>'),
             ('d', 'map<string,string>')
            ])


  def test_basic_structs(self):

    # Spark treats `Row` like a 'struct' rather than a map; here's how you can
    # use raw `Row` instances:
    rows = [
      Row(id=0, x=Row(shape='square', area=1)),
      Row(id=1, x=Row(shape='circle', area=2)),
    ]
    self._check_serialization(rows)
    self._check_schema(
            rows,
            [('id', 'bigint'),
             ('x',  'struct<shape:string,area:bigint>'),
            ])


  def test_built_in_unslotted(self):

    # Spark actually has built-in support for adapting plain Python objects,
    # but ...
    rows = [
      Row(id=0, x=Unslotted(v=4)),
      Row(id=1, x=Unslotted(v=5)),
    ]
    df = self._check_serialization(rows, do_adaption=False)
    
    # ... Spark deserializes 'objects' as generic `Row` instances.
    df_rows = df.collect()
    assert sorted(df_rows) == sorted([
      Row(id=0, v=Row(v=4)), Row(id=1, x=Row(v=5))
    ])


  def test_built_in_dataclasses(self):
    pytest.importorskip('dataclasses')

    # Spark also has built-in support for Python dataclasses, but ...
    rows = [
      Row(id=0, x=DataclassObj(x='foo', y=5.)),
      Row(id=1, x=DataclassObj(x='bar', y=6.)),
    ]
    df = self._check_serialization(rows, do_adaption=False)
    
    # ... Spark deserializes 'dataclasses' as generic `Row` instances.
    df_rows = df.collect()
    assert sorted(df_rows) == sorted([
      Row(id=0, v=Row(x='foo', y=5.)), Row(id=1, x=Row(x='bar', y=6.))
    ])


  def test_built_in_attrs(self):
    pytest.importorskip('attr')

    # Spark also has built-in support for attrs-based objects, but ...
    rows = [
      Row(id=0, x=AttrsUnslotted()),
      Row(id=1, x=AttrsUnslotted(foo="foom")),
    ]
    df = self._check_serialization(rows, do_adaption=False)
    
    # ... Spark deserializes 'attrs' objects as generic `Row` instances.
    df_rows = df.collect()
    assert sorted(df_rows) == sorted([
      Row(id=0, x=Row(bar=5, foo='moof')), Row(id=1, x=Row(bar=5, foo='foom'))
    ])


  def test_built_in_numpy(self):
    import numpy as np
    
    # Spark can NOT handle complex objects like numpy arrays
    rows = [
      Row(id=0, x=np.array([1, 2, 3])),
      Row(id=1, x=np.array([4, 5, 6])),
    ]
    with pytest.raises(TypeError) as excinfo:
      self._check_serialization(rows, do_adaption=False)
    assert "not supported type: <class 'numpy.ndarray'>" in str(excinfo.value)


  def test_built_in_slotted(self):
    
    # Spark does NOT have built-in support for *slotted* objects
    rows = [
      Row(id=0, x=Slotted(foo=5, bar="abc", _not_hidden=1)),
      Row(id=1, x=Slotted(foo=7, bar="cba", _not_hidden=3)),
    ]
    with pytest.raises(TypeError) as excinfo:
      self._check_serialization(rows, do_adaption=False)
    assert ("not supported type: <class 'oarphpy_test.test_spark.Slotted'>"
      in str(excinfo.value))


  def test_built_in_contained_udt(self):
    from pyspark.ml.linalg import DenseVector
    from pyspark.sql.types import StructType

    # As of Spark 3.x, Spark DOES support lists, maps, and structs of UDTs ...
    assert hasattr(DenseVector, '__UDT__')
    rows = [
      Row(id=0, x=[DenseVector([-1.0, -1.0])]),
      Row(id=0, x=[DenseVector([-2.0, -2.0])]),
    ]
    df = self._check_serialization(rows, do_adaption=False)
    
    # ... and will reconstruct python objects when querying data ...
    assert df.collect() == rows

    # But there are several drawbacks:
    #  1) You must manually write the schema for your UDT using Spark's type
    #       objects:
    assert type(DenseVector.__UDT__.sqlType()) == StructType
    #  2) For ndarrays, Spark ML only supports double-valued structures and
    #        offers no packed encoding for large arrays; compare with
    #        RowAdapter's Tensor below and see also:
    #          https://github.com/apache/spark/blob/6c805470a7e8d1f44747dc64c2e49ebd302f9ba4/python/pyspark/ml/linalg/__init__.py#L144
    values_field = [
      f for f in DenseVector.__UDT__.sqlType().fields if f.name == 'values'
    ][0]
    assert str(values_field.dataType) == 'ArrayType(DoubleType,false)'


  def test_rowadapter_unslotted(self):
    
    # RowAdapter will deserialize and re-create Python objects, using the
    # `Unslotted` class defined at runtime.
    rows = [
      Row(id=0, x=Unslotted(v=4)),
      Row(id=1, x=Unslotted(v=5)),
    ]
    df = self._check_serialization(rows)
    
    # NB: The check above also checks that `Unslotted` instances are created
    # and checks equality via __eq__ or pprint.pformat().  But just to make
    # things visible in this test:
    from oarphpy.spark import RowAdapter
    decoded = sorted(RowAdapter.from_row(r) for r in df.collect())
    assert [r.x for r in decoded] == [Unslotted(v=4), Unslotted(v=5)]

    # RowAdapter records the (full) class name in the table as the
    # `__pyclass__` attribute of each value of the `x` column.
    assert _select_distinct(df, 'x.__pyclass__') == [
      'oarphpy_test.test_spark.Unslotted']

    # RowAdapter encodes objects as structs (even though in Python objects are
    # very dict-like).
    self._check_schema(
            rows,
            [('id', 'bigint'),
             ('x',  'struct<__pyclass__:string,v:bigint>'),
            ])
    
  
  def test_rowadapter_dataclasses(self):
    pytest.importorskip('dataclasses')

    rows = [
      Row(id=0, x=DataclassObj(x='foo', y=5.)),
      Row(id=1, x=DataclassObj(x='bar', y=6.)),
    ]
    df = self._check_serialization(rows)
    
    assert _select_distinct(df, 'x.__pyclass__') == [
      'oarphpy_test.dataclass_obj.DataclassObj']

    self._check_schema(
            rows,
            [('id', 'bigint'),
             ('x',  'struct<__pyclass__:string,x:string,y:double>'),
            ])
  
  
  def test_rowadapter_attrs(self):
    pytest.importorskip('attr')

    rows = [
      Row(id=0, x=AttrsUnslotted()),
      Row(id=1, x=AttrsUnslotted(foo="foom")),
    ]
    df = self._check_serialization(rows)
    assert _select_distinct(df, 'x.__pyclass__') == [
      'oarphpy_test.test_spark.AttrsUnslotted']
    self._check_schema(
            rows,
            [('id', 'bigint'),
             ('x',  'struct<__pyclass__:string,foo:string,bar:bigint>'),
            ])
    
    rows = [
      Row(id=0, x=AttrsSlotted()),
      Row(id=1, x=AttrsSlotted(foo="foom")),
    ]
    df = self._check_serialization(rows)
    assert _select_distinct(df, 'x.__pyclass__') == [
      'oarphpy_test.test_spark.AttrsSlotted']
    
    self._check_schema(
            rows,
            [('id', 'bigint'),
             ('x',  'struct<__pyclass__:string,foo:string,bar:bigint>'),
            ])


  def test_rowadapter_numpy_unpacked(self):
    import numpy as np
    
    # RowAdapter translates Numpy arrays to a oarphy Tensor object that affords
    # SQL-based inspection for small arrays (and uses a more efficient row- or
    # column-major packed encoding for large arrays; see next test)
    rows = [
      Row(id=0, x=np.array([1, 2, 3])),
      Row(id=1, x=np.array([4, 5, 6])),
    ]
    df = self._check_serialization(rows)
    assert _select_distinct(df, 'x.__pyclass__') == [
              'oarphpy.spark.Tensor']
    EXPECTED = """
        id                                                     x
    0   0  (oarphpy.spark.Tensor, [3], int64, C, [1, 2, 3], [])
    1   1  (oarphpy.spark.Tensor, [3], int64, C, [4, 5, 6], [])
    """
    self._pandas_compare_str(df.orderBy('id').toPandas(), EXPECTED)

    # _check_serialization() verifies that the data gets decoded as numpy
    # arrays, but just to make things visible in this test:
    from oarphpy.spark import RowAdapter
    decoded = [RowAdapter.from_row(r) for r in df.orderBy('id').collect()]
    np.testing.assert_equal(decoded[0].x, np.array([1, 2, 3]))
    np.testing.assert_equal(decoded[1].x, np.array([4, 5, 6]))
  

  def test_rowadapter_numpy_packed(self):
    import numpy as np
    from oarphpy.spark import TENSOR_AUTO_PACK_MIN_KBYTES

    N = int(TENSOR_AUTO_PACK_MIN_KBYTES * (2**10) / np.dtype(int).itemsize) + 1
    expect_packed = np.reshape(np.array(range(2 * N)), (2, N))

    rows = [
      Row(id=0, x=expect_packed),
      Row(id=1, x=(expect_packed + 1)),
    ]
    df = self._check_serialization(rows)
    assert _select_distinct(df, 'x.__pyclass__') == [
              'oarphpy.spark.Tensor']
    
    # Verify that we actually have a column of packed values
    bin_data = df.select('*').first().x.values_packed
    assert len(bin_data) == expect_packed.size * expect_packed.dtype.itemsize
      # For ints, usually 8 bytes per int * 2 * N
    
    # _check_serialization() verifies that the data gets decoded as numpy
    # arrays, but just to make things visible in this test:
    from oarphpy.spark import RowAdapter
    decoded = [RowAdapter.from_row(r) for r in df.orderBy('id').collect()]
    np.testing.assert_equal(decoded[0].x, expect_packed)
    np.testing.assert_equal(decoded[1].x, expect_packed + 1)


  def test_rowadapter_cloudpickled_callable(self):
    from oarphpy.spark import CloudpickeledCallable
    
    def moof():
      return 'moof'
    cc_moof = CloudpickeledCallable(moof)
    assert cc_moof() == moof()

    cc_empty = CloudpickeledCallable()
    assert cc_empty == CloudpickeledCallable.empty()
    with pytest.raises(Exception):
      cc_empty()

    rows = [
      Row(id=0, f=cc_empty),
      Row(id=1, f=cc_moof),
    ]
    df = self._check_serialization(rows)
    assert _select_distinct(df, 'f.__pyclass__') == [
              'oarphpy.spark.CloudpickeledCallableData']
    
    # Ensuring invoking the functions still works
    from oarphpy.spark import RowAdapter
    decoded = [RowAdapter.from_row(r) for r in df.orderBy('id').collect()]
    with pytest.raises(Exception):
      decoded[0].f()
    assert decoded[1].f() == moof()


  def test_rowadapter_slotted(self):
    
    # RowAdapter will treat slotted objects like normal (unslotted) objects. 
    # The main difference is that if the slotted type changes and _loses_
    # slots, RowAdapter won't try to setattr() on missing slots.
    rows = [
      Row(id=0, x=Slotted(foo=5, bar="abc", _not_hidden=1)),
      Row(id=1, x=Slotted(foo=7, bar="cba", _not_hidden=3)),
    ]
    df = self._check_serialization(rows)
    assert _select_distinct(df, 'x.__pyclass__') == [
      'oarphpy_test.test_spark.Slotted']

    # RowAdapter encodes objects as structs (even though in Python objects are
    # very dict-like).
    self._check_schema(
            rows,
            [('id', 'bigint'),
             ('x',  'struct<__pyclass__:string,foo:bigint,bar:string,_not_hidden:bigint>'),
            ])


  def test_rowadapter_contained_objects(self):
    
      # RowAdapter will process containers of translatable objects recursively
      s1 = Slotted(foo=5, bar="abc", _not_hidden=1)
      s2 = Slotted(foo=7, bar="cba", _not_hidden=3)
      rows = [
        Row(id=0, x={'a': [{'b': [s1]}]}),
        Row(id=1, x={'a': [{'b': [s2]}]}),
      ]
      df = self._check_serialization(rows)
      
      # The resulting type is very complicated ...
      self._check_schema(
              rows,
              [('id', 'bigint'),
              ('x',  'map<string,array<map<string,array<struct<__pyclass__:string,foo:bigint,bar:string,_not_hidden:bigint>>>>>'),
              ])
      
      # ... but not hard to query!
      assert 0 ==  df.select('*').where("x.a[0].b[0].foo = 5").first().id
      assert 1 ==  df.select('*').where("x.a[0].b[0].bar = 'cba'").first().id


  def test_rowadapter_complex(self):
    from oarphpy.spark import RowAdapter

    # A large-ish example that covers the above cases in aggregate
    rows = [
      Row(
        id=1,
        np_number=np.float32(1.),
        a=np.array([1]), 
        b={
          'foo': np.array( [ [1] ], dtype=np.uint8)
        },
        c=[
          np.array([[[1.]], [[2.]], [[3.]]])
        ],
        d=Slotted(foo=5, bar="abc", _not_hidden=1),
        e=[Slotted(foo=6, bar="def", _not_hidden=1)],
        f=Unslotted(meow=4, _not_hidden=1, __hidden=2),
        g=Unslotted(), # Intentionally empty; adapter should set nothing
        h=Row(i=1, j=2),
      ),

      # Include a mostly empty row below to exercise Spark type validation.
      # Spark will ensure the row below and row above have the same schema;
      # note that `None` (or 'null') is only allowed for Struct / Row types.
      Row(
        id=2,
        np_number=np.float32(2.),
        a=np.array([]),
        b={},
        c=[],
        d=None,
        e=[],
        f=None,
        g=None,
        h=Row(i=3, j=3),
      ),
    ]

    df = self._check_serialization(rows)
    EXPECTED_ALL = """
                                                                                  0                                                1
    id                                                                            1                                                2
    np_number                                                                   1.0                                              2.0
    a                                (oarphpy.spark.Tensor, [1], int64, C, [1], [])  (oarphpy.spark.Tensor, [0], float64, C, [], [])
    b              {'foo': ('oarphpy.spark.Tensor', [1, 1], 'uint8', 'C', [1], [])}                                               {}
    c          [(oarphpy.spark.Tensor, [3, 1, 1], float64, C, [1.0, 2.0, 3.0], [])]                                               []
    d                                  (oarphpy_test.test_spark.Slotted, 5, abc, 1)                                             None
    e                                [(oarphpy_test.test_spark.Slotted, 6, def, 1)]                                               []
    f                                     (oarphpy_test.test_spark.Unslotted, 4, 1)                                             None
    g                                          (oarphpy_test.test_spark.Unslotted,)                                             None
    h                                                                        (1, 2)                                           (3, 3)
    """
    self._pandas_compare_str(df.orderBy('id').toPandas().T, EXPECTED_ALL)

    # Test Schema Deduction
    mostly_empty = Row(
      id=2,
      np_number=None,
      a=None,
      b={},
      c=[],
      d=None,
      e=[],
      f=None,
      g=None,
      h=None,
    )
    mostly_empty_adapted = RowAdapter.to_row(mostly_empty)

    # Spark can't deduce schema from the empty-ish row ...
    with pytest.raises(ValueError) as excinfo:
      self._check_serialization([mostly_empty_adapted], do_adaption=False)
    assert "Some of types cannot be determined" in str(excinfo.value)
    
    # ... but this works if we tell it the schema!
    schema = RowAdapter.to_schema(rows[0])
    self._check_serialization([mostly_empty_adapted], schema=schema)

    # Let's check that RowAdapter schema deduction works as expected
    EXPECTED_SCHEMA = [
      ('id', 'bigint'),
      ('np_number', 'double'),
      ('a', 'struct<__pyclass__:string,shape:array<bigint>,dtype:string,order:string,values:array<bigint>,values_packed:binary>'),
      ('b', 'map<string,struct<__pyclass__:string,shape:array<bigint>,dtype:string,order:string,values:array<bigint>,values_packed:binary>>'),
      ('c', 'array<struct<__pyclass__:string,shape:array<bigint>,dtype:string,order:string,values:array<double>,values_packed:binary>>'),
      ('d', 'struct<__pyclass__:string,foo:bigint,bar:string,_not_hidden:bigint>'),
      ('e', 'array<struct<__pyclass__:string,foo:bigint,bar:string,_not_hidden:bigint>>'),
      ('f', 'struct<__pyclass__:string,meow:bigint,_not_hidden:bigint>'),
      ('g', 'struct<__pyclass__:string>'),
      ('h', 'struct<i:bigint,j:bigint>'),
    ]
    self._check_schema(rows, EXPECTED_SCHEMA)
   

  ## Test Support

  def _is_spark_2x(self):
    if not hasattr(self, '_is_spark_2x_cache'):
      import pyspark
      self._is_spark_2x_cache = pyspark.__version__.startswith('2.') 
    return self._is_spark_2x_cache

  def _check_raw_adaption(self, raw_expected):
    from oarphpy.spark import RowAdapter
    
    for raw_data, expected_row in raw_expected:
      actual_row = RowAdapter.to_row(raw_data)
      assert actual_row == expected_row

      actual_data = RowAdapter.from_row(expected_row)
      assert actual_data == raw_data

  def _check_schema(self, rows, expected_schema):
    from oarphpy.spark import RowAdapter
    schema = RowAdapter.to_schema(rows[0])
    adapted_rows = [RowAdapter.to_row(r) for r in rows]
    with testutil.LocalSpark.sess() as spark:
      df = spark.createDataFrame(
        adapted_rows, schema=schema, verifySchema=False)
        # verifySchema is expensive and improperly errors on mostly empty rows
      
      if self._is_spark_2x():
        # Spark 2.x returns schema values in a different order, so we do a more
        # flexible test
        def tokenize(s):
          import re
          return sorted(re.split('[<>,]+', s))
        actual = dict((col, tokenize(s)) for col, s in df.dtypes)
        expected = dict((col, tokenize(s)) for col, s in expected_schema)
        assert actual == expected
      else:
        # Tests are written for Spark 3.x
        assert df.dtypes == expected_schema
    
    return schema

  def _check_serialization(self, rows, schema=None, do_adaption=True):
    import inspect
    from oarphpy import util
    from oarphpy.spark import RowAdapter 

    test_name = inspect.stack()[1][3]

    TEST_TEMPDIR = testutil.test_tempdir('TestRowAdapter.' + test_name)

    if do_adaption:
      adapted_rows = [RowAdapter.to_row(r) for r in rows]
    else:
      adapted_rows = rows

    with testutil.LocalSpark.sess() as spark:
      if schema:
        df = spark.createDataFrame(
          adapted_rows, schema=schema, verifySchema=False)
            # verifySchema is expensive and improperly errors on mostly
            # empty rows
      else:
        df = spark.createDataFrame(adapted_rows)
          # Automatically samples rows to get schema
      outpath = os.path.join(TEST_TEMPDIR, 'rowdata_%s' % test_name)
      df.write.parquet(outpath)

      df2 = spark.read.parquet(outpath)
      decoded_wrapped_rows = df2.collect()
      
      if do_adaption:
        decoded_rows = [
          RowAdapter.from_row(row)
          for row in decoded_wrapped_rows
        ]
        # We can't do assert sorted(rows) == sorted(decoded_rows)
        # because numpy syntatic sugar breaks __eq__, so use pprint,
        # which is safe for our tests
        import pprint
        def sorted_row_str(rowz):
          if self._is_spark_2x():
            # Spark 2.x has non-stable sorting semantics for Row
            if len(rowz) > 1:
              rowz = sorted(rowz, key=lambda r: r.id)
            return pprint.pformat(rowz)
          else:
            return pprint.pformat(sorted(rowz))
        assert sorted_row_str(rows) == sorted_row_str(decoded_rows)
      
      return df
    
  def _pandas_compare_str(self, pdf, expected):
    def cleaned(s):
      lines = [l.strip() for l in s.split('\n')]
      lines = [l for l in lines if l]
      if self._is_spark_2x():
        # Spark 2.x returns table rows and array values in different orders,
        # so we do a more flexible test
        def tokenize(s):
          import re
          return sorted(re.split('[\(\),]+', s))
        lines = sorted(' '.join(tokenize(l)) for l in lines)
      return '\n'.join(lines)
    actual = pdf.to_string()
    assert cleaned(actual) == cleaned(expected), \
      "\n\nExpected %s \n\nActual %s\n\n" % (expected, actual)

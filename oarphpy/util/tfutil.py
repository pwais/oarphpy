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

from contextlib import contextmanager

from oarphpy.util.misc import GPUS_UNRESTRICTED


def tf_create_session_config(restrict_gpus=GPUS_UNRESTRICTED, extra_opts=None):
  extra_opts = extra_opts or {}
  
  import tensorflow as tf
  config = tf.compat.v1.ConfigProto()

  tf_session_config_restrict_gpus(config, restrict_gpus=restrict_gpus)
  config.log_device_placement = False
  
  # # Enable CPU XLA!
  # config.graph_options.optimizer_options.global_jit_level = \
  #   tf.OptimizerOptions.ON_1

  for k, v in extra_opts.items():
    setattr(config, k, v)
  return config


def tf_session_config_restrict_gpus(config, restrict_gpus=GPUS_UNRESTRICTED):
  if restrict_gpus is GPUS_UNRESTRICTED:
    config.allow_soft_placement = True
  else:
    config.device_count['GPU'] = len(restrict_gpus)
    config.gpu_options.visible_device_list = (
      ','.join(str(g) for g in restrict_gpus))
  config.gpu_options.allow_growth = True


def tf_create_session(config=None):
  config = config or tf_create_session_config()

  import tensorflow as tf
  sess = tf.compat.v1.Session(config=config)
  return sess


def tf_cpu_session_config():
  return tf_create_session_config(restrict_gpus=[])


def tf_cpu_session(config=None):
  if not config:
    config = tf_cpu_session_config()
  else:
    tf_session_config_restrict_gpus(config, restrict_gpus=[])
  return tf_create_session(config=config)


@contextmanager
def tf_data_session(dataset, sess=None, config=None):
  import tensorflow as tf

  # Must declare these before the graph gets finalized below
  iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
  next_element = iterator.get_next()
  
  # Silly way to iterate over a tf.Dataset
  # https://stackoverflow.com/a/47917849
  sess = sess or tf_cpu_session()
  with sess as sess:
    def iter_dataset():
      # see MonitoredTrainingSession.StepContext
      while True:
        try:
          # with loop_until_data_exausted():
          yield sess.run(next_element)
        except (tf.errors.OutOfRangeError, StopIteration):
          break
    yield sess, iter_dataset


# TPUS don't support strings :P  but they do support lists of integers
def to_padded_codepoint_list(s, max_len=5000):
  s_enc = s.encode('utf-8')
  return list(s_enc[:max_len]) + ([0] * (max_len - len(s_enc)))


def from_padded_codepoint_list(l, null_value=0):
  return ''.join(
              chr(el) for el in l # May only work in python3?
              if (null_value is None or el != null_value))


def give_me_frozen_graph(
          checkpoint,
          nodes=None,
          blacklist=None,
          base_graph=None,
          sess=None,
          saver=None):
  """
  Tensorflow has several ways to load checkpoints / graph artifacts.
  It's impossible to know if some API is stable or if tomorrow somebody
  will invent something new and break everything becaus PyTorch is shiny
  (e.g. TF Eager).  Sam Abrahams wrote a book on Tensorflow
  ( https://www.amazon.com/TensorFlow-Machine-Intelligence-hands--introduction-ebook/dp/B01IZ43JV4/ )
  and one time couldn't tell me definitively which API to use.  What's more is
  that freeze_graph.py is an optional script instead of a library module in
  Tensorflow.  Chaos!!

  So, based upon spark-dl's `strip_and_freeze_until()`
  ( https://github.com/databricks/spark-deep-learning/blob/4daa1179f498df4627310afea291133539ce7001/python/sparkdl/graph/utils.py#L199 ),
  here's a utility for getting a frozen, serializable, pyspark-friendly
  graph from a checkpoint artifact metagraph thingy I have no idea.
  """

  def op_name(v):
    name = v
    if hasattr(v, 'name'):
      name = v.name
    if ':' not in name:
      return name
    toks = name.split(':')
    assert len(toks) <= 2, (toks, v, name)
    return toks[0]

  import tensorflow as tf
  graph = base_graph or tf.Graph()
  if nodes:
    ops = [graph.get_operation_by_name(op_name(n)) for n in nodes]
  else:
    ops = graph.get_operations()
  # if blacklist:
  #   for n in blacklist:
  #     ops.remove(graph.get_operation_by_name(op_name(n)))

  with graph.as_default():
    with (sess or tf_cpu_session()) as sess:
      saver = saver or tf.train.Saver()
      log.info("Reading from checkpoint %s ..." % checkpoint)
      saver.restore(sess, checkpoint)
      log.info("... done.")

      gdef_frozen = tf.graph_util.convert_variables_to_constants(
        sess,
        graph.as_graph_def(add_shapes=True),
        [op.name for op in ops])
        # variable_names_blacklist=blacklist)
  return gdef_frozen


def tf_variable_summaries(var, prefix=''):
  """Create Tensorboard summaries showing basic stats of the
  variable `var`."""
  import tensorflow as tf

  if prefix:
    prefix = prefix + '/'
  else:
    def var_name(v):
      """Magic: get the name of the variable that the caller passed to 
      `tf_variable_summaries()`"""
      import inspect
      lcls = inspect.stack()[2][0].f_locals
      for name in lcls:
        if id(v) == id(lcls[name]):
          return name
      return None
    prefix = var_name(var)
    if not prefix:
      prefix = str(var.name)
      idx = prefix.find('/')
      if idx >= 0:
        prefix = prefix[:prefix.find('/')] # Exclude slashes in var name
      idx = prefix.find(':')
      if idx >= 0:
        prefix = prefix[:prefix.find(':')] # Exclude : too
  
  with tf.variable_scope(prefix):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


class TFSummaryRow(object):
  __slots__ = (
    'path',
    'split',

    'step',
    'wall_time',
    'tag',

    'simple_value',
    'image',
    'tensor',
  )

  def __init__(self):
    self.path = ''
    self.split = ''
    self.step = -1
    self.wall_time = 0
    self.tag = ''
    self.simple_value = float('nan')
    self.image = None
    self.tensor = None

  @staticmethod
  def fill_simple_value(row, summary):
    if summary.HasField('simple_value'):
      row.simple_value = summary.simple_value
  
  @staticmethod
  def fill_image(row, summary):
    if summary.HasField('image'):
      import imageio
      row.image = imageio.imread(summary.image.encoded_image_string)
  
  @staticmethod
  def fill_tensor(row, summary):
    if summary.HasField('tensor'):
      import tensorflow as tf
      row.tensor = tf.make_ndarray(summary.tensor)
  
  def as_dict(self):
    return dict((k, getattr(self, k)) for k in self.__slots__)
  
  def as_row(self, extra=None):
    from pyspark.sql import Row
    from au.spark import NumpyArray
    d = self.as_dict()
    d['image'] = NumpyArray(d['image'])
    d['tensor'] = NumpyArray(d['tensor'])
    d.update(**(extra or {}))
    return Row(**d)
    

class TFSummaryReader(object):

  # Subclass and use this attribute to elide / ignore some summary messages
  FILLERS = (
    TFSummaryRow.fill_simple_value,
    TFSummaryRow.fill_image,
    TFSummaryRow.fill_tensor,
  )

  def __init__(self, paths=None, glob_events_from_dir=None):
    self._paths = paths or []
    if glob_events_from_dir and os.path.exists(glob_events_from_dir):
      self._paths.extend(
        pathlib.Path(glob_events_from_dir).rglob('**/events.out*'))

  def __iter__(self):
    import tensorflow as tf
    for path in self._paths:
      path = str(path)
      log.info("Reading summaries from path %s ..." % path)
      
      split = ''
      # TF estimators puts eval summaries in the 'eval' subdir
      eval_str = os.pathsep + 'eval' + os.pathsep
      if eval_str in path:
        split = 'eval'

      def iter_events_verbose(path):
        # When there's an error in the file, e.g. truncated record, Tensorflow
        # doesn't print the path :(
        try:
          for tf_event in tf.train.summary_iterator(path):
            yield tf_event
        except Exception as e:
          raise Exception(("Error reading file %s" % path, e))
      
      for tf_event in iter_events_verbose(path):
        for tf_summary in tf_event.summary.value:
          row = TFSummaryRow()
          row.path = path
          row.split = split

          row.wall_time = tf_event.wall_time
          row.step = tf_event.step
          row.tag = tf_summary.tag

          for filler in self.FILLERS:
            filler(row, tf_summary)
          
          yield row


class TFRecordsFileAsListOfStrings(object):
  """
  Friends Don't Let Friends Use TFRecords.

  This utility provides a Tensorflow-free, minimal-dependency solution
  for reading TFRecords from a *file stream* (e.g. a buffered reader) and
  exposes random access over the stream's records.

  Based upon:
    * https://github.com/apache/beam/blob/master/sdks/python/apache_beam/io/tfrecordio.py#L67
    * https://www.tensorflow.org/versions/r1.11/api_guides/python/python_io#TFRecords_Format_Details
    * https://github.com/gdlg/simple-waymo-open-dataset-reader/blob/master/simple_waymo_open_dataset_reader/__init__.py#L19
    * https://gist.github.com/ed-alertedh/9f49bfc6216585f520c7c7723d20d951
  """

  ## Public API

  def __init__(self, fileobj):
    self.fileobj = fileobj
    self._offset_length = None

  def __len__(self):
    self._maybe_build_index()
    return len(self._offset_length)
  
  def __getitem__(self, idx):
    if idx >= len(self):
      raise IndexError
    else:
      start, length = self._offset_length[idx]
      return self._get_data(start, length)
  
  def __iter__(self):
    self.fileobj.seek(0)
    for offset, length in self._iter_offset_length():
      yield self._get_data(offset, length)

  ## Utils

  @classmethod
  def _masked_crc32c(cls, value):
    """Compute a masked crc32c checksum for a value.  FMI see
    https://www.tensorflow.org/versions/r1.11/api_guides/python/python_io#TFRecords_Format_Details
    """

    if not hasattr(cls, '_crc32c_fn'):
      import crcmod
      cls._crc32c_fn = crcmod.predefined.mkPredefinedCrcFun('crc-32c')

    crc = cls._crc32c_fn(value)
    return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xffffffff

  @classmethod
  def _check_crc(cls, value, expected_crc):
    crc_actual = cls._masked_crc32c(value)
    if crc_actual != expected_crc:
      import codecs
      raise ValueError(
        'Invalid TFRecord, mistmatch: %s %s %s' % (
          codecs.encode(value[:10], 'hex'), crc_actual, expected_crc))

  def _iter_offset_length(self):
    self.fileobj.seek(0)
    while True:
      header = self.fileobj.read(12)
      if header == b'':
        break
      assert len(header) == 12

      import struct
      length, lengthcrc = struct.unpack("<QI", header)
      self._check_crc(header[:8], lengthcrc)

      base = self.fileobj.tell()
      yield (base, length)

      # Skip over payload and payload CRC
      self.fileobj.seek(base + length + 4)

  def _maybe_build_index(self):
    if self._offset_length is None:
      self._offset_length = list(self._iter_offset_length())
      self.fileobj.seek(0)

  def _get_data(self, start, length):
    self.fileobj.seek(start)
    payload = self.fileobj.read(length + 4)
    assert len(payload) == (length + 4)

    import struct
    data, datacrc = struct.unpack("<%dsI" % length, payload)
    self._check_crc(data, datacrc)

    return data

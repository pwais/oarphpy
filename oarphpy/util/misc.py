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

import itertools
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from contextlib import contextmanager


def np_truthy(v):
  import numpy as np
  if isinstance(v, np.ndarray):
    return bool(v.size)
  else:
    return bool(v)


def get_size_of_deep(v):
  """(Hacky) Get size of the value `v` in bytes.  Does not rely on a more
  precise library like guppy or pympler.  Intended for values `v` that 
  contain large binary blobs."""
  import six
  INTEGRAL_TYPES = tuple(itertools.chain.from_iterable(
        (six.string_types, six.integer_types, six.class_types,
        (bytes, bytearray))))
    # The above types can trigger expensive recursion unless we base case them
  if isinstance(v, INTEGRAL_TYPES):
    return sys.getsizeof(v)
  if hasattr(v, 'nbytes'):
    return v.nbytes
  elif hasattr(v, 'items'):
    # Typically a dict
    return sum(
      get_size_of_deep(key) + get_size_of_deep(value)
      for key, value in v.items())
  elif hasattr(v, '__iter__'):
    # Typically a list or tuple
    return sum(get_size_of_deep(el) for el in iter(v))
  elif hasattr(v, '__dict__'):
    return sum(
      get_size_of_deep(dk) + get_size_of_deep(dv)
      for dk, dv in v.__dict__.items())
  elif hasattr(v, '__slots__'):
    return sum(get_size_of_deep(getattr(v, k)) for k in v.__slots__)
  else:
    return sys.getsizeof(v)


def stable_hash(x):
  """A hash of `x` that is stable across program runs.
  
  Background:
  As of Python 3, `hash()` is given a fresh seed every time the interpret
  starts; hash codes are not stable without setting the env var
  `PYTHONHASHSEED`.

  Can we just simply adjust for the seed programmatically?
  Note that while it is possible to get the hash seed at runtime:
   * https://stackoverflow.com/questions/41088635/extract-hash-seed-in-unit-testing
  Python doesn't use the seed in an easily-inverted way:
   * https://github.com/python/cpython/blob/630c8df5cf126594f8c1c4579c1888ca80a29d59/Python/pyhash.c#L237

  Thus for stability (and even light portability), we leverage Python
  serialization to provide a key for `x`.
  """

  try:
    import cloudpickle as pkl
  except ImportError:
    import pickle as pkl
  
  key = pkl.dumps(x)
  
  import hashlib
  return int(hashlib.md5(key).hexdigest(), 16)


def ichunked(seq, n):
  """Generate chunks of `seq` of size (at most) `n`.  More efficient
  and less junk than itertools recipes version using izip_longest...
  """
  n = max(1, n)
  seq = iter(seq)
  while True:
    chunk = tuple(itertools.islice(seq, n))
    if chunk:
      yield chunk
    else:
      break


def roundrobin(*seqs):
  """Generate a sequence pulling round-robin from each of `seqs`; similar to
  `itertools.roundrobin()` recipe but 
    (1) won't hide nested `StopIteration`s
    (2) uses a queue to reduce dynamic allocations
  """
  import six
  from collections import deque
  its = deque((iter(s) for s in seqs), maxlen=len(seqs))
  while its:
    it = its.popleft()
    try:
      v = six.next(it)
    except StopIteration:
      continue
    yield v
    its.append(it)


def as_row_of_constants(inst):
  """Row-ify an object instance `inst` keeping only the "class-constant"
  attributes of `inst`, i.e. the members with UPPERCASE names.

  >>> class Foo(object):
  ...   CONST = 5
  ...   def __init(self, x):
  ...     self.x = x

  >>> as_row_of_constants(Foo())
  OrderedDict([('CONST', 5)])

  """
  from collections import OrderedDict
  row = OrderedDict()
  
  def is_constant_field(name):
    return not name.startswith('_') and name.isupper()

  for attr in sorted(dir(inst)):
    if is_constant_field(attr):
      v = getattr(inst, attr)
      if isinstance(v, (str, float, int, list, dict)):
        row[attr] = v
      else:
        subrow = as_row_of_constants(v)
        if subrow:
          if hasattr(v, '__name__'):
            row[attr] = v.__name__
          else:
            row[attr] = v.__class__.__name__
        for col, colval in subrow.items():
          row[attr + '_' + col] = colval
  return row


def fname_timestamp(random_suffix=True):
  timestr = time.strftime("%Y-%m-%d-%H_%M_%S")
  if random_suffix:
    # Ideally we use a UUID but idk
    # https://stackoverflow.com/a/2257449
    import random
    import string
    NUM_CHARS = 5
    chars = (
      random.choice(string.ascii_uppercase + string.digits)
      for _ in range(NUM_CHARS)
    )
    timestr = timestr + "." + ''.join(chars)
  return timestr


class Proxy(object):
  """A thin wrapper around an `instance` that supports custom semantics."""
  
  __slots__ = ('instance',)
  
  def __init__(self, instance):
    self.instance = instance
  
  def __getattr__(self, name):
    return getattr(self.instance, name)
  
  def _on_delete(self):
    pass

  def __del__(self):
    self._on_delete()
    del self.instance


@contextmanager
def quiet():
  """Silence stdout and stderr for any commands in this context"""
  old_stdout = sys.stdout
  old_stderr = sys.stderr
  f = open(os.devnull, 'w')
  
  # python3, pytest, and docker can combine to create an
  # odd 'ValueError: underlying buffer has been detached'
  # error unless we detach() now.
  import codecs
  f = codecs.getwriter("utf-8")(f.detach())
  
  new_stdout = sys.stdout = f
  new_stderr = sys.stderr = f
  try:
    yield new_stdout, new_stderr
  finally:
    try:
      new_stdout.seek(0)
      new_stderr.seek(0)
    except Exception:
      pass
    sys.stdout = old_stdout
    sys.stderr = old_stderr


@contextmanager
def with_cwd(path):
  """Use a current working directory of `path` for this context"""
  old_cwd = os.getcwd()
  os.chdir(path)
  try:
    yield
  finally:
    os.chdir(old_cwd)


@contextmanager
def imageio_ignore_warnings():
  # Imageio needs some fix: https://github.com/imageio/imageio/issues/376
  import imageio.core.util
  def silence_imageio_warning(*args, **kwargs):
    pass
  old = imageio.core.util._precision_warn
  imageio.core.util._precision_warn = silence_imageio_warning
  try:
    yield
  finally:
    imageio.core.util._precision_warn = old


def to_png_bytes(arr):
  """Typically used for testing; when comparing images, we need to compare
  actual and expected via image bytes b/c imageio does some sort of subtle
  color normalization and we want our fixtures to simply be user-readable
  PNGs."""

  import io
  import imageio
  buf = io.BytesIO()
  imageio.imwrite(buf, arr, 'png')
  return buf.getvalue()


def to_jpeg_bytes(arr, quality=100):
  """Given a numpy array image `arr`, return the image encoded as a
  jpeg buffer."""

  import io
  import imageio
  buf = io.BytesIO()
  imageio.imwrite(buf, arr, 'jpg', quality=quality)
  return buf.getvalue()


def get_jpeg_size(jpeg_bytes):
  """Get the size of a JPEG image without reading and decompressing the entire
  file.  Based upon:  
   * https://github.com/shibukawa/imagesize_py/blob/master/imagesize.py#L87
  """
  import struct
  from io import BytesIO
  buf = BytesIO(jpeg_bytes)
  head = buf.read(24)
  if not head.startswith(b'\377\330'):
    raise ValueError("Invalid JPEG header")
  buf.seek(0)
  size = 2
  ftype = 0
  while not 0xc0 <= ftype <= 0xcf or ftype in [0xc4, 0xc8, 0xcc]:
    buf.seek(size, 1)
    byte = buf.read(1)
    while ord(byte) == 0xff:
      byte = buf.read(1)
    ftype = ord(byte)
    size = struct.unpack('>H', buf.read(2))[0] - 2
  # Now we're at a SOFn block
  buf.seek(1, 1)  # Skip `precision' byte.
  height, width = struct.unpack('>HH', buf.read(4))
  return width, height


def run_cmd(cmd, collect=False, nolog=False):
  dolog = not nolog
  cmd = cmd.replace('\n', '').strip()
  
  if dolog:
    from oarphpy.util.log import create_log
    log = create_log()
    log.info("Running %s ..." % cmd)
  
  if collect:
    out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
  else:
    subprocess.check_call(cmd, shell=True)
    out = None
  
  if dolog:
    log.info("... done with %s " % cmd)
  
  return out


def get_non_loopback_iface():
  # Based upon https://stackoverflow.com/a/1267524
  import socket
  non_loopbacks = [
    ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")
  ]
  if non_loopbacks:
    return non_loopbacks[0]

  # Get an iface that can connect to Google DNS ...
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  s.connect(("8.8.8.8", 80))
  iface = s.getsockname()[0]
  s.close()
  return iface


_SYS_INFO_LOCK = threading.Lock()
def get_sys_info():
  global _SYS_INFO_LOCK
  from oarphpy.util.log import create_log
  log = create_log()

  log.info("Listing system info ...")

  info = {}
  info['filepath'] = os.path.abspath(__file__)
  info['PYTHONPATH'] = ':'.join(sys.path)
  
  @contextmanager
  def atomic_ignore_exceptions():
    with _SYS_INFO_LOCK:
      try:
        yield
      except Exception:
        pass

  def safe_cmd(cmd):
    with atomic_ignore_exceptions():
      return run_cmd(cmd, collect=True) or ''
  
  # NB: some commands, especially nvidia-smi, crash under concurrent access
  info['nvidia_smi'] = safe_cmd('nvidia-smi')
  info['cpuinfo'] = safe_cmd('cat /proc/cpuinfo')
  info['disk_free'] = safe_cmd('df -h')
  info['ifconfig'] = safe_cmd('ifconfig')
  info['memory'] = safe_cmd('free -h')
  
  TEST_URI = 'https://raw.githubusercontent.com/pwais/au2018/master/README.md'
  info['have_internet'] = bool(safe_cmd('curl ' + TEST_URI))

  import socket
  info['hostname'] = socket.gethostname()
  info['host'] = get_non_loopback_iface()

  import multiprocessing
  info['n_cpus'] = multiprocessing.cpu_count()
  
  log.info("... got all system info.")

  return info


def copy_n_from_zip(src, dest, n):
  log.info("Copying %s of %s -> %s ..." % (n, src, dest))

  mkdir(os.path.split(dest)[0])

  import zipfile
  with zipfile.ZipFile(src) as zin:
    with zipfile.ZipFile(dest, mode='w') as zout:
      for name in itertools.islice(sorted(zin.namelist()), n):
        zout.writestr(name, zin.read(name))
  
  log.info("... done")


def mkdir(path):
  import errno
  try:
    os.makedirs(path)
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise


def rm_rf(path):
  shutil.rmtree(path)


def all_files_recursive(root_dir, pattern='*'):
  import fnmatch
  paths = []
  for root, dirs, files in os.walk(root_dir):
    for basename in files:
      if fnmatch.fnmatch(basename, pattern):
        paths.append(os.path.join(root, basename))
  return paths


def cleandir(path):
  mkdir(path)
  rm_rf(path)
  mkdir(path)


def missing_or_empty(path):
  if not os.path.exists(path):
    return True
  else:
    for p in all_files_recursive(path):
      return False
    return True


def is_stupid_mac_file(path):
  fname = os.path.basename(path)
  return fname.startswith('._') or fname in ('.DS_Store',)


def download(uri, dest, try_expand=True):
  """Fetch `uri`, which is a file or archive, and put in `dest`, which
  is either a destination file path or destination directory."""
  
  import math
  from oarphpy.util import log
  from oarphpy.util.thruput_observer import ThruputObserver

  # Import urllib
  try:
    import urllib.error as urlliberror
    import urllib.request
    HTTPError = urlliberror.HTTPError
    URLError = urlliberror.URLError
  except ImportError:
    import urllib2 as urllib
    HTTPError = urllib.HTTPError
    URLError = urllib.URLError
    import urllib.request
    
  import patoolib
 
  if os.path.exists(dest):
    return
  
  fname = os.path.split(uri)[-1]
  tempdest = tempfile.NamedTemporaryFile(suffix='_' + fname)
  try:
    log.info("Fetching %s ..." % uri)
    response = urllib.request.urlopen(uri)
    size = int(response.info().get('Content-Length').strip())
    log.info("... downloading %s MB ..." % (float(size) * 1e-6))
    chunk_size = min(size, 8192)
    t = ThruputObserver(
          name=uri,
          log_freq=10000,
          n_total=math.ceil(size / chunk_size))
    while True:
      with t.observe(n=1, num_bytes=chunk_size):
        data = response.read(chunk_size)
        if not data:
          break
        tempdest.write(data)
      t.maybe_log_progress()
    sys.stdout.write("")
    sys.stdout.flush()
    log.info("... fetched!")
  except HTTPError as e:
    raise Exception("[HTTP Error] {code}: {reason}."
                        .format(code=e.code, reason=e.reason))
  except URLError as e:
    raise Exception("[URL Error] {reason}.".format(reason=e.reason))
  
  tempdest.flush()
  
  if try_expand:
    try:
      # Is it an archive? expand!
      mkdir(dest)
      patoolib.extract_archive(tempdest.name, outdir=dest)
      log.info("Extracted archive.")
    except Exception:
      # Just move the file
      shutil.move(tempdest.name, dest)
      tempdest.delete = False
  else:
    shutil.move(tempdest.name, dest)
    tempdest.delete = False
  log.info("Downloaded to %s" % dest)


### GPU Utils

GPUS_UNRESTRICTED = None

class GPUInfo(object):
  __slots__ = (
    'index',
    'name',
    'mem_util_frac',
    'mem_free',
    'mem_used',
    'mem_total'
  )

  def __str__(self):
    data = ', '.join((k + '=' + str(getattr(self, k))) for k in self.__slots__)
    return 'GPUInfo(' + data + ')'

  def __eq__(self, other):
    return all(getattr(self, k) == getattr(other, k) for k in self.__slots__)

  @staticmethod
  def from_nvidia_smi(row):
    info = GPUInfo()
    info.index = int(row['index'])
    info.name = row['name']
    
    info.mem_util_frac = float(row['utilization.memory [%]']) / 100.
    def to_num_bytes(s):
      return int(s) * int(1e6)
    info.mem_free = to_num_bytes(row['memory.free [MiB]'])
    info.mem_used = to_num_bytes(row['memory.used [MiB]'])
    info.mem_total = to_num_bytes(row['memory.total [MiB]'])

    return info

  @staticmethod
  def get_infos(only_visible=True):
    from oarphpy.util import log

    # Much safer than pycuda and Tensorflow, which can both segfault if the
    # nvidia driver is absent :P
    try:
      cmd = "nvidia-smi --query-gpu=index,name,utilization.memory,name,memory.total,memory.free,memory.used --format=csv,nounits"
      out = run_cmd(cmd, collect=True)
    except Exception as e:
      log.info("No GPUs found")
      return []

    # NB: nvidia doesn't actually return *valid* csv.
    # Why would they? They make hardware, not software!
    out = out.decode('utf-8')
    out = out.replace(', ', ',')

    import csv
    rows = list(csv.DictReader(out.split('\n')))
    infos = [GPUInfo.from_nvidia_smi(row) for row in rows]
    
    log.info("Found GPUs: %s" % ([str(info) for info in infos],))

    if only_visible:
      if 'CUDA_VISIBLE_DEVICES' in os.environ:
        allowed_gpus = set(
          int(g) for g in
          os.environ['CUDA_VISIBLE_DEVICES'].split(',')
          if g)
        log.info("... restricting to GPUs %s ..." % (allowed_gpus,))
        infos = [
          info for info in infos
          if info.index in allowed_gpus
        ]
    return infos
  
  @staticmethod
  def num_total_gpus():
    return len(GPUInfo.get_infos())

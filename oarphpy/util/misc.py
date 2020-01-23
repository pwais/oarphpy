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


################################################################################
### Logging

try:
  # FIXME For Tensorflow double-logging bug:
  # FIXME(https://github.com/abseil/abseil-py/issues/99)
  # FIXME(https://github.com/abseil/abseil-py/issues/102)
  # Unfortunately, many libraries that include absl (including Tensorflow)
  # will get bitten by double-logging due to absl's incorrect use of
  # the python logging library:
  #   2019-07-19 23:47:38,829 my_logger   779 : test
  #   I0719 23:47:38.829330 139904865122112 foo.py:63] test
  #   2019-07-19 23:47:38,829 my_logger   779 : test
  #   I0719 23:47:38.829469 139904865122112 foo.py:63] test
  # The code below fixes this double-logging.  FMI see:
  #   https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
  
  import logging
  
  import absl.logging
  logging.root.removeHandler(absl.logging._absl_handler)
  absl.logging._warn_preinit_stderr = False
except ImportError as e:
  pass
except Exception as e:
  print("Failed to fix absl logging bug", e)
  pass

_LOGS = {}
def create_log(name='oarph'):
  global _LOGS
  if name not in _LOGS:
    import logging
    LOG_FORMAT = "%(asctime)s\t%(name)-4s %(process)d : %(message)s"
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    log.addHandler(console_handler)
    _LOGS[name] = log
  return _LOGS[name]

# NB: Spark workers will lazy-construct and cache logger instances
log = create_log()


################################################################################
### Pythonisms

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
  # https://stackoverflow.com/a/1267524
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



################################################################################
### ArchiveFileFlyweight

class _IArchive(object):
  
  def __init__(self, path):
    self.archive_path = path

  @classmethod
  def list_names(cls, archive_path):
    return []

  def get(self, name):
    """Get the entry for `name` as an in-memory array"""
    raise KeyError("Interface stores no data")

  def get_reader(self, name):
    """Get the entry for `name` as a file-like (buffered) object"""
    raise KeyError("Interface stores no data")

  # Only pickle the archive path; subclasses should set up every other member
  # lazily.  This approach allows for easy interop with Spark.

  def __getstate__(self):
     return (self.archive_path,)

  def __setstate__(self, d):
     self.archive_path = d[0]

class _ZipArchive(_IArchive):

  @property
  def _zipfile(self):
    if not hasattr(self, '__zipfile'):
      import zipfile
      self.__zipfile = zipfile.ZipFile(self.archive_path)
    return self.__zipfile

  def get(self, name):
    return self._zipfile.read(name)

  def get_reader(self, name):
    return self._zipfile.open(name)

  @classmethod
  def list_names(cls, archive_path):
    import zipfile
    return zipfile.ZipFile(archive_path).namelist()

class _TarArchive(_IArchive):

  @property
  def _tarfile(self):
    if not hasattr(self, '__tarfile'):
      import tarfile
      self.__tarfile = tarfile.open(self.archive_path)
    return self.__tarfile

  def get(self, name):
    return self.get_reader(name).read()

  def get_reader(self, name):
    return self._tarfile.extractfile(name)

  @classmethod
  def list_names(cls, archive_path):
    import tarfile
    return tarfile.open(archive_path).getnames()

class ArchiveFileFlyweight(object):

  __slots__ = ('name', 'archive')

  def __init__(self, name='', archive=None):
    self.name = name
    self.archive = archive

  def __getstate__(self):
     return (self.name, self.archive)

  def __setstate__(self, d):
     self.name, self.archive = d

  @staticmethod
  def fws_from(archive_path):
    archive_cls = None
    TAR_SUFFIXES = ('tar', 'tar.gz', 'tgz')
    if archive_path.endswith('zip'):
      archive_cls = _ZipArchive
    elif any(archive_path.endswith(suffix) for suffix in TAR_SUFFIXES):
      archive_cls = _TarArchive
    else:
      raise ValueError("Don't know how to read %s" % archive_path)

    archive = archive_cls(archive_path)
    names = archive_cls.list_names(archive_path)
    return [
      ArchiveFileFlyweight(name=name, archive=archive)
      for name in names
    ]

  @property
  def data(self):
    return self.archive.get(self.name)
  
  @property
  def data_reader(self):
    return self.archive.get_reader(self.name)

def copy_n_from_zip(src, dest, n):
  log.info("Copying %s of %s -> %s ..." % (n, src, dest))

  mkdir(os.path.split(dest)[0])

  import zipfile
  with zipfile.ZipFile(src) as zin:
    with zipfile.ZipFile(dest, mode='w') as zout:
      for name in itertools.islice(sorted(zin.namelist()), n):
        zout.writestr(name, zin.read(name))
  
  log.info("... done")


################################################################################
### I/O

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


################################################################################
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


# TODO: clean up and move to own module
# class SystemLock(object):
#   """Uses fasteners / flock to provide a inter-process *and*
#   inter-thread lock using a file.  Yes, in addition to the file lock, 
#   we do need our *own* thread lock, since fasteners isnt thread-safe :(
#     https://github.com/harlowja/fasteners/blob/master/fasteners/process_lock.py#L62
#   """

#   def __init__(self, name_prefix='au.SystemLock', name='', abspath=''):
#     if not name:
#       import uuid
#       name = name_prefix + '.' + str(uuid.uuid4())
#     if not abspath:
#       abspath = os.path.join(tempfile.gettempdir(), name)
#     self._flock = fasteners.InterProcessLock(abspath)
#     self._tlock = threading.Lock()

#   # Make pickle-able for interop with Spark
#   def __getstate__(self):
#     return {'_flock_path': self._flock.path}
  
#   def __setstate__(self, d):
#     self._flock = fasteners.InterProcessLock(d['_flock_path'])
#     self._tlock = threading.Lock()
  
#   @property
#   def path(self):
#     return self._flock.path
  
#   def __enter__(self):
#     self._tlock.acquire()
#     self._flock.acquire()
#     return self
  
#   def __exit__(self, *args):
#     self._flock.release()
#     self._tlock.release()


# class GPUPool(object):
#   """
#   An arbiter providing system-wide mutually exclusive handles to GPUs.  Mutual
#   exclusion is via file locks and cooperative use; handles emitted from this
#   utility have no impact on the underlying GPU devices.  Useful for restricting
#   individual pyspark worker processes to distinct GPUs.  (Otherwise, a Spark
#   executor can easily cause GPU OOMs when launching multiple worker processes
#   or threads).
  
#   Other context:
#   Tensorflow nortoriously enjoys comandeering all available GPU memory,
#   which can result in OOMs when Sessions try to run concurrently.  Morevoer,
#   nvidia-smi offers a feature to support "exclusive use" mode, but we don't
#   necessarily want to lock out other processes (e.g. the OS) and nvidia
#   software (especially drivers) typically have bugs or fragmentation issues.
#   This utility provides mutual exclusion that is concise and independent of
#   nvidia software as well as any cuda-wrapping framework (e.g. pycuda or 
#   Tensorflow) which can even segfault when devices / drivers are missing.
#   """
  
#   # Users need only keep a GPUInfo (proxy) in scope to maintain ownership
#   class _InfoProxy(Proxy):
#     __slots__ = ('instance', '_parent')
#     def _on_delete(self):
#       self._parent._release(self.instance)
#     def __str__(self):
#       return 'Proxy:' + str(self.instance)

#   ALL_GPUS = -1
#   def get_free_gpus(self, n=1):
#     """Return up to `n` handles to free GPU(s) or [] if none available.
#     Use `n` = -1 to retain *all* GPUs."""
#     with self._slock:
#       if n == self.ALL_GPUS:
#         n = GPUInfo.num_total_gpus()
#       handles = []
#       gpus = self._get_gpus()
#       n = min(n, len(gpus))
#       while gpus and len(handles) != n:
#         gpu = gpus.pop(0)
#         h = GPUPool._InfoProxy(gpu)
#         h._parent = self
#         handles.append(h)
#       self._set_gpus(gpus)
#       return handles

#   def __str__(self):
#     return "GPUPool(path='%s')" % self._slock.path

#   def __init__(self, path='', name=''):
#     self._slock = SystemLock(name_prefix='au.GPUPool', name=name, abspath=path)

#   def _set_gpus(self, lst):
#     with open(self._slock.path, 'wb') as f:
#       pickle.dump(lst, f, protocol=pickle.HIGHEST_PROTOCOL)

#   def _get_gpus(self):
#     with open(self._slock.path, 'rb') as f:
#       try:
#         return pickle.load(f)
#       except EOFError:
#         # No process has yet initialized state in self._slock.path
#         return GPUInfo.get_infos()

#   def _release(self, gpu):
#     with self._slock:
#       gpus = self._get_gpus()
#       gpus.append(gpu)
#       log.info("Re-claimed GPU %s, free GPUs: %s" % (
#         gpu, [str(g) for g in gpus]))
#       self._set_gpus(gpus)



# def _Worker_run(inst_datum_bytes):
#   import sys
#   import traceback

#   # Multiprocesing workers ignore System exceptions :(
#   # https://stackoverflow.com/a/23682499
#   try:
#     log.info("Running in subprocess %s ..." % os.getpid())
#     import cloudpickle
#     inst_datum = cloudpickle.loads(inst_datum_bytes)
#     inst, datum = inst_datum
#     return inst.run(*datum['args'], **datum['kwargs'])
  
#   except:
#     cls, exc, tb = sys.exc_info()
#     msg = "Unhandled exception in worker %s (%s):\n%s" % (
#                 cls.__name__, exc, traceback.format_exc())
#     raise Exception(msg)

# class Worker(object):
#   # Default worker requires no GPUs and runs in parent thread
#   N_GPUS = 0
#   CPU_ONLY_OK = False # Only if N_GPUS > 0, can we degrade to CPU-only mode?
#   GPU_POOL = None # E.g. use a pool associated with a specific job

#   SYSTEM_EXCLUSIVE = False
  
#   PROCESS_ISOLATED = False
#   PROCESS_TIMEOUT_SEC = 1e9 # NB: Pi Billion is approx 1 century
  
#   _SYSTEM_LOCK_PATH = os.path.join(tempfile.gettempdir(), 'au.Worker')
#   _SYSTEM_LOCK = SystemLock(abspath=_SYSTEM_LOCK_PATH)

#   # For non-exclusive workers
#   # https://stackoverflow.com/a/45187287
#   class _NullContextManager(object):
#     def __init__(self, x=None):
#         self.x = x
#     def __enter__(self):
#         return self.x
#     def __exit__(self, *args):
#         pass

#   @contextmanager
#   def __maybe_lock_gpus(self):
#     self._gpu_ids = []

#     if self.N_GPUS == 0:
    
#       yield  # Don't need to block
    
#     else:
#       gpu_pool = self.GPU_POOL or GPUPool(name_prefix='au.Worker.GPUPool')

#       if self.N_GPUS == GPUPool.ALL_GPUS:
#         self.N_GPUS = GPUInfo.num_total_gpus()
      
#       handles = []
#       start = time.time()
#       while True:
#         handles.extend(gpu_pool.get_free_gpus(n=self.N_GPUS))
#         if (len(handles) == self.N_GPUS) or self.CPU_ONLY_OK:
#           log.info("Got GPUs %s from %s, waited %s sec" % (
#                       [str(h) for h in handles],
#                       gpu_pool,
#                       time.time() - start))
#           break
#         else:
#           log.info("Waiting for %s GPUs in pool %s, waited for %s sec ..." % (
#             (self.N_GPUS, gpu_pool, time.time() - start)))
#           import random
#           time.sleep(5 + random.random())
        
#       # Expose to subclass if subclass needs them
#       self._gpu_ids = [h.index for h in handles]
#       print('self.__gpu_ids', self._gpu_ids, os.getpid())
#       yield

#       # `handles` will expire and release GPUs

#   def __call__(self, *args, **kwargs):
#     ctx = Worker._NullContextManager()
#     if self.SYSTEM_EXCLUSIVE:
#       ctx = self._SYSTEM_LOCK
    
#     ## BEGIN from contextlib import nested
#     # Gee thanks Guido! :P https://stackoverflow.com/a/39158985
#     try:
#       from contextlib import nested  # Python 2
#     except ImportError:
#       from contextlib import ExitStack, contextmanager

#       @contextmanager
#       def nested(*contexts):
#         """Reimplementation of nested in python 3."""
#         with ExitStack() as stack:
#           for ctx in contexts:
#             stack.enter_context(ctx)
#           yield contexts
#     ## END from contextlib import nested
    
#     ctx = nested(ctx, self.__maybe_lock_gpus())
#     with ctx:
#       log.info("Starting worker %s ..." % self._name)
#       if self.PROCESS_ISOLATED:
#         import cloudpickle
#         import multiprocessing
#         pool = multiprocessing.Pool(
#                     processes=1,
#                     maxtasksperchild=1)
#                       # Prevent process re-use; e.g. we need a Tensorflow
#                       # process to exist for it to ever release GPU memory :(
#         inst_datum_bytes = cloudpickle.dumps(
#           (self, {'args': args, 'kwargs': kwargs}))
#         # We must use async so that parent processs signals get handled
#         # https://stackoverflow.com/a/23682499
#         proxy = pool.map_async(_Worker_run, [inst_datum_bytes])
#         results = proxy.get(timeout=self.PROCESS_TIMEOUT_SEC)
#         result = results[0]

#         # Force process to release resources
#         pool.close()
#         pool.terminate()
#       else:
#         result = self.run(*args, **kwargs)
#       log.info("... done with worker %s" % self._name)
#       return result


#   ## Subclass API

#   @property
#   def _name(self):
#     return self.__class__.__name__
  
#   _gpu_ids = GPUS_UNRESTRICTED

#   # @property
#   # def _gpu_ids(self):
#   #   if not hasattr(self, '__gpu_ids'):
#   #     self.__gpu_ids = GPUS_UNRESTRICTED
#   #   return self.__gpu_ids

#   def run(self, *args, **kwargs):
#     # Base class worker does nothing
#     return None

# class WholeMachineWorker(Worker):
#   N_GPUS = GPUPool.ALL_GPUS
#   GPU_POOL = GPUPool() # Use a distinct pool for this program run
#   SYSTEM_EXCLUSIVE = True
#   PROCESS_ISOLATED = True

# class SingleGPUWorker(Worker):
#   N_GPUS = 1
#   GPU_POOL = GPUPool() # Use a distinct pool for this program run
#   SYSTEM_EXCLUSIVE = False
#   PROCESS_ISOLATED = True

# class AtMostOneGPUWorker(SingleGPUWorker):
#   CPU_ONLY_OK = True


################################################################################
### Tensorflow

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
    # print(prefix, var.name) # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
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

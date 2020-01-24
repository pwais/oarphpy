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


class _IArchive(object):
  """Simple shim to interop with tarfile, zipfile, etc."""
  
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

  ### Only pickle the archive path; subclasses should set up every other member
  ### lazily.  This approach allows for easy interop with Spark.

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
  """A flyweight[1] for a single file in an archive.

  [1] https://en.wikipedia.org/wiki/Flyweight_pattern
  """

  __slots__ = ('name', 'archive')

  def __init__(self, name='', archive=None):
    self.name = name
    self.archive = archive

  ### Access the entry's binary data

  @property
  def data(self):
    return self.archive.get(self.name)
  
  @property
  def data_reader(self):
    return self.archive.get_reader(self.name)

  ### Only pickle the metadata. This approach allows for easy interop
  ### with Spark.

  def __getstate__(self):
     return (self.name, self.archive)

  def __setstate__(self, d):
     self.name, self.archive = d

  @staticmethod
  def fws_from(archive_path):
    """Create and return a list of `ArchiveFileFlyweight`s from the given 
    `archive_path`.
    """
    
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

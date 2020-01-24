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

from oarphpy import util
from oarphpy_test import testutil


def _check_fws(fws, expected):
  # First, check the Flyweights `fws`
  assert len(fws) == len(expected)
  datas = [fw.data for fw in fws]
  assert sorted(datas) == sorted(expected)

  # Now check pickle support
  import pickle
  fws_str = pickle.dumps(fws)
  fws_decoded = pickle.loads(fws_str)
  assert len(fws_decoded) == len(expected)
  datas_decoded = [fw.data for fw in fws_decoded]
  assert sorted(datas_decoded) == sorted(expected)


def test_archive_flyweight_zip():
  TEST_TEMPDIR = testutil.test_tempdir('test_archive_flyweight_zip')
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
  fws = util.ArchiveFileFlyweight.fws_from(fixture_path)
  _check_fws(fws, ss)


def test_archive_flyweight_tar():
  TEST_TEMPDIR = testutil.test_tempdir('test_archive_flyweight_tar')
  fixture_path = os.path.join(TEST_TEMPDIR, 'test.tar')

  # Create the fixture:
  # test.tar
  #  |- foo: "foo"
  #  |- bar: "bar"
  # ... an archive with a few files, where each file contains just a string
  #   that matches the name of the file in the archive
  ss = [b'foo', b'bar', b'bazzzz']
    
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
  
  # Test reading!
  fws = util.ArchiveFileFlyweight.fws_from(fixture_path)
  _check_fws(fws, ss)

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for firestore.py."""

import os
import unittest
from unittest import mock

from absl.testing import parameterized
import constants
import firestore
import numpy as np


_FAKE_NP_LOADTXT_INPUT_2_LINES = np.array(
    [('key_item_1', 'rcm_item_1', 1, 0.9),
     ('key_item_1', 'rcm_item_2', 2, 0.8)],
    dtype=firestore._CSV_INPUT_DTYPES,
)
_FAKE_NP_LOADTXT_INPUT_1_LINE = np.array(
    [('key_item_1', 'rcm_item_1', 1, 0.9)],
    dtype=firestore._CSV_INPUT_DTYPES,
)
_FAKE_COMMON_PATH = '/path/to'
_FAKE_INPUT_FILEPATH = os.path.join(_FAKE_COMMON_PATH, 'input.csv')
_FAKE_WRONG_FILEPATH = '/faile_file_path'


class FirestoreTest(parameterized.TestCase):
  @parameterized.named_parameters([
      {
          'testcase_name': '2_rows_file_success',
          'loadtxt_input_data': _FAKE_NP_LOADTXT_INPUT_2_LINES,
      },
      {
          'testcase_name': '1_row_file_success',
          'loadtxt_input_data': _FAKE_NP_LOADTXT_INPUT_1_LINE,
      }
  ])
  @mock.patch.object(firestore.os.path, 'exists', return_value=True)
  def test_read_csv_to_np_array_with_success(
      self,
      _,
      loadtxt_input_data,
  ):
    mock_np_loadtxt = mock.patch.object(
        firestore.np,
        'loadtxt',
        return_value=loadtxt_input_data,
        autospec=True
    ).start()
    csv_path = _FAKE_INPUT_FILEPATH

    actual = firestore.read_csv_to_np_array(csv_path)

    mock_np_loadtxt.assert_called_with(
        _FAKE_INPUT_FILEPATH,
        delimiter=constants.DELIMITER,
        skiprows=constants.SKIPROWS,
        dtype=np.dtype(
            [(constants.KEYWORD, 'O'),
             (constants.RCM_RESULT, 'O'),
             (constants.RANK, 'i2'),
             (constants.SCORE, 'f4')]
        )
    )
    np.testing.assert_array_equal(
        loadtxt_input_data,
        actual,
    )

  @mock.patch.object(firestore.os.path, 'exists', return_value=False)
  def test_read_csv_to_np_array_with_failure_wrong_file_path(
      self,
      _
  ):
    with self.assertRaisesRegex(
        IOError,
        'The input file dose not exist.'
    ):
      firestore.read_csv_to_np_array(_FAKE_WRONG_FILEPATH)


if __name__ == '__main__':
  unittest.main()

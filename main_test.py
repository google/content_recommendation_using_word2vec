# Copyright 2023 Google LLC
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

"""Tests for content_recommendation_w2v.py."""
import os
import unittest
import main
import pandas as pd

_USER_ID = 'user_id'
_ITEM_LIST = 'item_list'
_CNT = 'cnt'

_DUMMY_DF_TRAINING_FILEPATH = os.path.join('./', 'input.csv')

COL_NAMES_INPUT = (
    _USER_ID,
    _ITEM_LIST,
    _CNT
)

_DUMMY_DF_TRAINNG = pd.DataFrame({
    _USER_ID: ['user_a', 'user_b', 'user_c'],
    _ITEM_LIST: ['ITEM_A,ITEM_B,ITEM_C,ITEM_B,ITEM_A',
                 'ITEM_B,ITEM_A,ITEM_B,ITEM_A,ITEM_B',
                 'ITEM_C,ITEM_D,ITEM_E,ITEM_D,ITEM_C'],
    _CNT: [5, 5, 5]
    })


class MainTest(unittest.TestCase):
  def test_success_read_csv(self):
    """Ensure success with correct csv."""
    expected_df = _DUMMY_DF_TRAINNG

    _DUMMY_DF_TRAINNG.to_csv(_DUMMY_DF_TRAINING_FILEPATH, index=False)
    actual_df = main.read_csv(_DUMMY_DF_TRAINING_FILEPATH)

    pd.testing.assert_frame_equal(actual_df, expected_df)

  def test_raise_error_read_csv_with_empty_path(self):
    """Ensure failed with empty path."""
    with self.assertRaises(IOError):
      _ = main.read_csv('')


if __name__ == '__main__':
  unittest.main()

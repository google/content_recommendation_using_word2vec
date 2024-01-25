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

"""Functions for loading & storing data into Firestore.

Run Firestore functions from project's root directory.

Example:
  `python firestore.py -i [Input data path]`
"""

import os

import constants
import error_messages
import numpy as np


_CSV_INPUT_DTYPES = np.dtype([
    (constants.KEYWORD, 'O'),
    (constants.RCM_RESULT, 'O'),
    (constants.RANK, 'i2'),
    (constants.SCORE, 'f4'),
])


def read_csv_to_np_array(path: str) -> np.ndarray:
  """Reads csv data and returns numpy array.

  Args:
    path: A path to read csv data that you store data into Firestore.

  Returns:
    A numpy array loded from csv path.

  Raises:
    IOError: if the path is not found, or the resource cannot be opened.
  """
  if os.path.exists(path):
    rcm_output_data = np.loadtxt(
        path,
        delimiter=constants.DELIMITER,
        skiprows=constants.SKIPROWS,
        dtype=_CSV_INPUT_DTYPES,
    )
  else:
    raise IOError(error_messages.NOT_EXISTS_INPUT_FILE)

  return rcm_output_data

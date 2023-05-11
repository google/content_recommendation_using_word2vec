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

"""Functions for training & predicting content by content recommendation using word2vec.
"""

import logging
import pandas as pd


logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
)


def read_csv(path: str) -> pd.DataFrame:
  """Read csv data and return dataframe.

  Args:
    path: path to read csv data

  Returns:
    A dataframe loded from path.

  Raises:
    IOError: if the path is not found, or the resource cannot be opened.
  """
  try:
    df = pd.read_csv(path)
  except IOError as e:
    logging.exception('Can not load csv data with %r.', path)
    raise e

  return df

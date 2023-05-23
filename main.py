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
import gensim
import pandas as pd

_SG = 1
_WINDOWS = 5
_MIN_COUNT = 5
_VECTOR_SIZE = 100
_HS = 0
_NEGATIVE = 5
_SEED = 1

logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
)


def _read_csv(path: str) -> pd.DataFrame:
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
    logging.exception('Can not load csv data with %s.', path)
    raise e

  return df


def execute_embedding_w2v(
    training_data: pd.DataFrame,
    ) -> gensim.models.word2vec.Word2Vec:
  """Executes embedding content data by word2vec.

  Args:
    training_data: A DataFrame of training data that the content ID
      is for each line each user in the order that a certain user saw the
      content as the list type. Example is [['ITEM_A', 'ITEM_B', 'ITEM_C'],
      ['ITEM_B', 'ITEM_A', 'ITEM_B'], ['ITEM_C', 'ITEM_D', 'ITEM_E']].
  Returns:
    A model of embedding resul by word2vec.
  """
  model = gensim.models.word2vec.Word2Vec(
      sentences=training_data,
      sg=_SG,
      window=_WINDOWS,
      min_count=_MIN_COUNT,
      vector_size=_VECTOR_SIZE,
      hs=_HS,
      negative=_NEGATIVE,
      seed=_SEED,
  )
  logging.info('Finished training of gensim word2vec.')

  return model


def execute_content_recommendation_w2v_from_csv(
    input_file_path: str,
    content_file_path: str,
    output_file_path: str,
    ) -> None:
  """Trains and predicts contensts recommendation with word2vec.

  Args:
    input_file_path: A CSV format file path of training data that the content ID
      is for each line each user in the order that a certain user saw the
      content.
    content_file_path: A CSV format file path of content data with content id,
      content title and content URL.
    output_file_path: A CSV format file path of output.
  """
  df_training = _read_csv(input_file_path)
  logging.info('Loaded training data with %s.', input_file_path)
  training_data = df_training['item_list'].apply(lambda x: x.split(',')
                                                 ).tolist()
  logging.info(
      'Completed a process of loaded data into training data for'
      'gensim word2vec.'
  )

  # TODO(): Replace '_' with 'df_content' in next CL.
  _ = _read_csv(content_file_path)
  logging.info('Loaded content data.')

  # TODO(): Replace '_' with 'model' in next CL.
  _ = execute_embedding_w2v(training_data)

  # TODO(): Add feature sort recommend results for outputs in next CL.
  # df_result = sort_recommendation_result(model, df_content)

  # TODO(): Replace df_training with df_result in next CL.
  df_training.to_csv(output_file_path, index=False)
  logging.info('Completed exportion of predicted data.')

  logging.info('Completed process.')


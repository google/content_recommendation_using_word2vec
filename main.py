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

Run from project's root directory.

Example: `python -i [Input data path] -c [Content data path] -o [Output path]`

Please check README.md and sample data in the root project for the format of
input data and content data.
"""
import argparse
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

_KEYWORD = 'keyword'
_RCM_RESULT = 'rcm_result'
_RANK = 'rank'
_SCORE = 'score'
_TOP_N = 7

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


def sort_recommendation_results(model: gensim.models.word2vec.Word2Vec,
                                df_content: pd.DataFrame,
                                ) -> pd.DataFrame:
  """Sorts recommendation results for easy use as output data.

  Args:
    model: A model that was trained by gensin word2vec.
    df_content: A DataFrame of content data with content id,
      content title and content URL.

  Returns:
    A dataframe sorted recommendation data with key content id, recommend
    content id, rank, score.
  """
  df_result = pd.DataFrame({_KEYWORD: pd.Series(dtype='object'),
                            _RCM_RESULT: pd.Series(dtype='object'),
                            _RANK: pd.Series(dtype='int64'),
                            _SCORE: pd.Series(dtype='float64'),
                            })

  for _, content in df_content.iterrows():
    try:
      ret = model.wv.most_similar(positive=content[0], topn=_TOP_N)
    except KeyError as e:
      logging.debug(
          'Error happend during loading content item id: %s, %s', content[0], e
          )
      continue
    for i, (rcm_result, score) in enumerate(ret):
      record = pd.DataFrame([[content[0],
                              rcm_result,
                              int(i + 1),
                              score]
                             ], columns=df_result.columns
                            )
      df_result = pd.concat([df_result, record])

  logging.info('Completed process to sort embedding data.')
  return df_result


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

  df_content = _read_csv(content_file_path)
  logging.info('Loaded content data.')

  model = execute_embedding_w2v(training_data)

  df_result = sort_recommendation_results(model, df_content)

  df_result.to_csv(output_file_path, index=False)
  logging.info('Completed exportion of predicted data.')

  logging.info('Completed process.')


def parse_cli_args() -> argparse.Namespace:
  """Parses command line arguments.

  Returns:
    An instance of argparse.Namespace with arg values.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input', '-i',
      help='Input data file path to train models.',
      default=None,
      required=True,
      type=str,
      )
  parser.add_argument(
      '--content', '-c',
      help='Content file path to macth content id with URL in the outputs.',
      default=None,
      required=True,
      type=str,
      )
  parser.add_argument(
      '--output', '-o',
      help='Output file path for prediction results.',
      default=None,
      required=True,
      type=str,
      )

  return parser.parse_args()


def main() -> None:
  """Executes contenst recommendation using word2vec for file type."""
  args = parse_cli_args()
  execute_content_recommendation_w2v_from_csv(args.input,
                                              args.content,
                                              args.output,
                                              )


if __name__ == '__main__':
  main()

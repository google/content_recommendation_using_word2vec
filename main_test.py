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

"""Tests for main.py."""
import argparse
import os
import sys
import unittest
from unittest import mock
import main
import pandas as pd

_USER_ID = 'user_id'
_ITEM_LIST = 'item_list'
_CNT = 'cnt'
_ITEM = 'item'
_TITLE = 'title'
_URL = 'url'
_KEYWORD = 'keyword'
_RCM_RESULTS = 'rcm_result'
_RANK = 'rank'
_SCORE = 'score'
_DUMMY_RANKING_PROCESS_TRUE = True
_DUMMY_RANKING_PROCESS_FALSE = False
_DUMMY_RANKING_ITEN_NAME = 'Dummy_Ranking'
_DEFAULT_RANKING_ITEM_NAME = 'undefined'

_DUMMY_COMMON_PATH = '/path/to'
_DUMMY_INPUT_FILEPATH = os.path.join(_DUMMY_COMMON_PATH, 'input.csv')
_DUMMY_CONTENT_FILEPATH = os.path.join(_DUMMY_COMMON_PATH, 'content.csv')
_DUMMY_OUTPUT_FILEPATH = os.path.join(_DUMMY_COMMON_PATH, 'ouput.csv')
_DUMMY_DF_TRAINING_FILEPATH = os.path.join('./', 'input.csv')

COL_NAMES_INPUT = (
    _USER_ID,
    _ITEM_LIST,
    _CNT
)

_DUMMY_DF_TRAINNG = pd.DataFrame({
    _USER_ID: ['user_a', 'user_b', 'user_c', 'user_d'],
    _ITEM_LIST: ['ITEM_A,ITEM_B,ITEM_C,ITEM_B,ITEM_A,ITEM_B',
                 'ITEM_B,ITEM_A,ITEM_B,ITEM_A,ITEM_B,ITEM_C',
                 'ITEM_B,ITEM_A,ITEM_B,ITEM_A,ITEM_B,ITEM_C',
                 'ITEM_C,ITEM_D,ITEM_E,ITEM_D,ITEM_C,ITEM_A'],
    _CNT: [6, 6, 6, 6]
    })

_DUMMY_DF_CONTENTS = pd.DataFrame({
    _ITEM: ['ITEM_A', 'ITEM_B', 'ITEM_C', 'ITEM_D', 'ITEM_E'],
    _TITLE: ['ITEM A', 'ITEM B', 'ITEM C', 'ITEM D', 'ITEM E'],
    _URL: ['https://example.com/item_a',
           'https://example.com/item_b',
           'https://example.com/item_c',
           'https://example.com/item_d',
           'https://example.com/item_e']
    })
_DUMMY_TRAINING_DATA = [['ITEM_A', 'ITEM_B', 'ITEM_C', 'ITEM_B', 'ITEM_A',
                         'ITEM_B'],
                        ['ITEM_B', 'ITEM_A', 'ITEM_B', 'ITEM_A', 'ITEM_B',
                         'ITEM_C'],
                        ['ITEM_B', 'ITEM_A', 'ITEM_B', 'ITEM_A', 'ITEM_B',
                         'ITEM_C'],
                        ['ITEM_C', 'ITEM_D', 'ITEM_E', 'ITEM_D', 'ITEM_C',
                         'ITEM_A']]

_DUMMY_DF_CONTENTS = pd.DataFrame({
    _ITEM: ['ITEM_A', 'ITEM_B', 'ITEM_C', 'ITEM_D', 'ITEM_E'],
    _TITLE: ['ITEM A', 'ITEM B', 'ITEM C', 'ITEM D', 'ITEM E'],
    _URL: ['https://example.com/item_a',
           'https://example.com/item_b',
           'https://example.com/item_c',
           'https://example.com/item_d',
           'https://example.com/item_e']
    })
_DUMMY_DF_ERROR_CONTENTS = pd.DataFrame({
    _ITEM: ['ITEM_F'],
    _TITLE: ['ITEM F'],
    _URL: ['https://example.com/item_f'],
    })
_DUMMY_DF_RESULTS = pd.DataFrame({
    _KEYWORD: ['ITEM_A', 'ITEM_A', 'ITEM_B', 'ITEM_B', 'ITEM_C', 'ITEM_C'],
    _RCM_RESULTS: ['ITEM_B', 'ITEM_C', 'ITEM_A', 'ITEM_C', 'ITEM_A', 'ITEM_B'],
    _RANK: [1, 2, 1, 2, 1, 2],
    _SCORE: [mock.ANY, mock.ANY, mock.ANY, mock.ANY, mock.ANY, mock.ANY]
    })
_DUMMY_DF_RANKING = pd.DataFrame({
    _KEYWORD: [_DUMMY_RANKING_ITEN_NAME,
               _DUMMY_RANKING_ITEN_NAME,
               _DUMMY_RANKING_ITEN_NAME,
               _DUMMY_RANKING_ITEN_NAME,
               _DUMMY_RANKING_ITEN_NAME
               ],
    _RCM_RESULTS: ['ITEM_B', 'ITEM_A', 'ITEM_C', 'ITEM_D', 'ITEM_E'],
    _RANK: [1, 2, 3, 4, 5],
    _SCORE: [0.0, 0.0, 0.0, 0.0, 0.0]
    })

_DUMMY_MODEL = main.gensim.models.word2vec.Word2Vec(
    sentences=_DUMMY_TRAINING_DATA,
    sg=main._SG,
    window=main._WINDOWS,
    min_count=main._MIN_COUNT,
    vector_size=main._VECTOR_SIZE,
    hs=main._HS,
    negative=main._NEGATIVE,
    seed=main._SEED,
    )


class MainTest(unittest.TestCase):
  def test_success_read_csv(self):
    """Ensures success with correct csv."""
    expected_df = _DUMMY_DF_TRAINNG

    _DUMMY_DF_TRAINNG.to_csv(_DUMMY_DF_TRAINING_FILEPATH, index=False)
    actual_df = main._read_csv(_DUMMY_DF_TRAINING_FILEPATH)

    pd.testing.assert_frame_equal(actual_df, expected_df)

  def test_raise_error_read_csv_with_empty_path(self):
    """Ensures failed with empty path."""
    with self.assertRaises(IOError):
      _ = main._read_csv('')

  @mock.patch('main.pd.read_csv', side_effect=[_DUMMY_DF_TRAINNG,
                                               _DUMMY_DF_CONTENTS,
                                               ])
  def test_execute_content_recommendation_w2v_from_csv_with_required_params(
      self,
      _,
      ):
    """Ensures success with correct csv."""
    with mock.patch('main.pd.DataFrame.to_csv') as mock_to_csv:
      main.execute_content_recommendation_w2v_from_csv(_DUMMY_INPUT_FILEPATH,
                                                       _DUMMY_CONTENT_FILEPATH,
                                                       _DUMMY_OUTPUT_FILEPATH,
                                                      )

      mock_to_csv.assert_called_once_with(_DUMMY_OUTPUT_FILEPATH, index=False)


  @mock.patch('main.pd.read_csv', side_effect=[_DUMMY_DF_TRAINNG,
                                               _DUMMY_DF_CONTENTS,
                                               ])
  def test_execute_content_recommendation_w2v_from_csv_with_all_params(self, _):
    """Ensures success with correct csv."""
    with mock.patch('main.pd.DataFrame.to_csv') as mock_to_csv:
      main.execute_content_recommendation_w2v_from_csv(
          _DUMMY_INPUT_FILEPATH,
          _DUMMY_CONTENT_FILEPATH,
          _DUMMY_OUTPUT_FILEPATH,
          _DUMMY_RANKING_PROCESS_TRUE,
          _DUMMY_RANKING_ITEN_NAME,
          )

      mock_to_csv.assert_called_once_with(_DUMMY_OUTPUT_FILEPATH, index=False)

  def test_execute_embedding_w2v(self):
    """Ensures success with correct trainin_data."""
    with mock.patch('main.gensim.models.word2vec.Word2Vec') as mock_gensim:
      _ = main.execute_embedding_w2v(_DUMMY_TRAINING_DATA)

      mock_gensim.assert_called_once_with(sentences=mock.ANY,
                                          sg=main._SG,
                                          window=main._WINDOWS,
                                          min_count=main._MIN_COUNT,
                                          vector_size=main._VECTOR_SIZE,
                                          hs=main._HS,
                                          negative=main._NEGATIVE,
                                          seed=main._SEED,
                                          )

  def test_sort_recommendation_result(self):
    """Ensures success with sort_recommendation_result function."""
    actual_df_result = main.sort_recommendation_results(_DUMMY_MODEL,
                                                        _DUMMY_DF_CONTENTS,
                                                        )
    pd.testing.assert_frame_equal(
        actual_df_result.reset_index(drop=True).drop(columns=[_SCORE]),
        _DUMMY_DF_RESULTS.reset_index(drop=True).drop(columns=[_SCORE]),
        )

  def test_sort_recommendation_result_with_keyerror(self):
    """Ensures keyerror with sort_recommendation_result function."""
    with self.assertLogs(level='DEBUG') as log_output:
      _ = main.sort_recommendation_results(_DUMMY_MODEL,
                                           _DUMMY_DF_ERROR_CONTENTS,
                                          )
    self.assertIn('Error happend during loading content item id',
                  log_output.output[0])

  def test_parse_cli_args_returns_namespace_with_required_args(self):
    """Ensures parse_cli_args with success case."""
    test_args = [
        'main.py',
        '-i',
        _DUMMY_INPUT_FILEPATH,
        '-c',
        _DUMMY_CONTENT_FILEPATH,
        '-o',
        _DUMMY_OUTPUT_FILEPATH,
        ]
    expected = argparse.Namespace(
        input=_DUMMY_INPUT_FILEPATH,
        content=_DUMMY_CONTENT_FILEPATH,
        output=_DUMMY_OUTPUT_FILEPATH,
        is_ranking=_DUMMY_RANKING_PROCESS_FALSE,
        ranking_item_name=_DEFAULT_RANKING_ITEM_NAME,
        )

    with mock.patch.object(sys, 'argv', test_args):
      actual = main.parse_cli_args()

    self.assertIsInstance(actual, argparse.Namespace)
    self.assertEqual(actual, expected)

  def test_parse_cli_args_returns_namespace_with_args(self):
    """Ensures parse_cli_args with success case."""
    test_args = [
        'main.py',
        '-i',
        _DUMMY_INPUT_FILEPATH,
        '-c',
        _DUMMY_CONTENT_FILEPATH,
        '-o',
        _DUMMY_OUTPUT_FILEPATH,
        '-r',
        '-ri',
        _DUMMY_RANKING_ITEN_NAME,
        ]
    expected = argparse.Namespace(
        input=_DUMMY_INPUT_FILEPATH,
        content=_DUMMY_CONTENT_FILEPATH,
        output=_DUMMY_OUTPUT_FILEPATH,
        is_ranking=_DUMMY_RANKING_PROCESS_TRUE,
        ranking_item_name=_DUMMY_RANKING_ITEN_NAME
        )

    with mock.patch.object(sys, 'argv', test_args):
      actual = main.parse_cli_args()

    self.assertIsInstance(actual, argparse.Namespace)
    self.assertEqual(actual, expected)

  def test_execute_ranking_process(self):
    actual_df = main.execute_ranking_process(
        _DUMMY_TRAINING_DATA,
        _DUMMY_RANKING_ITEN_NAME,
    )

    pd.testing.assert_frame_equal(actual_df.reset_index(drop=True),
                                  _DUMMY_DF_RANKING.reset_index(drop=True),
                                  )


if __name__ == '__main__':
  unittest.main()

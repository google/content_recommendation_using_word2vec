# Content Recommendation using word2vec

## Problem
Content recommendation engines are one of key parts to increase page views in
publisher's website or apps, but many publishers do not manage own content
recommendation because creating a content recommendation engine yourself is
hard work.

## Solution
Content Recommendation using word2vec provides sample contents recommendation
engine code using Google Analytics 4 data from BigQuery based on word2vec
embedding.

## Deploy
### Requirements
- Google Analytics 4 data in BigQuery
- Python 3.10 ~

### Initial Setup
1. Setup Python environment (e.g. pyenv etc.) with libraries based on requirements.txt.

Example
```
pip install requirements.txt
```

2. Prepare training data to use SQL for BigQuery based on sample sql
 (sample_extract_input_data_from_GA4.sql). In training data, each row has
user_id, item_list ordered by time.

3. Prepare content data to use SQL for BigQuery based on sample sql
 (sample_extract_content_data_from_GA4.sql). Contents data need include
 contents id, contents title and contents url etc in each row.

4. Optional: adjust hyper parameter in word2vec or term of input data if you
want.

5. Optional: Directory extract input and contents data from BigQuery.

### Scheduled execution
1. Run main.py in root directory.
```
python main.py -i [Input data path] -c [Content data path] -o [Output path]

Example with sample data:
python main.py -i sample_input_data.csv -c sample_content_data.csv -o output.csv
```


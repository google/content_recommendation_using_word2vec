/*Copyright 2023 Google LLC.
This solution, including any related sample code or data, is made available on an “as is,” “as available,” and “with all faults” basis, solely for illustrative purposes, and without warranty or representation of any kind. This solution is experimental, unsupported and provided solely for your convenience. Your use of it is subject to your agreements with Google, as applicable, and may constitute a beta feature as defined under those agreements. To the extent that you make any data available to Google in connection with your use of the solution, you represent and warrant that you have all necessary and appropriate rights, consents and permissions to permit Google to use and process that data. By using any portion of this solution, you acknowledge, assume and accept all risks, known and unknown, associated with its usage, including with respect to your deployment of any portion of this solution in your systems, or usage in connection with your business, if at all.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License. */

WITH COTENT_URL_TITLE_ID AS (
  SELECT
    (SELECT value.string_value FROM UNNEST(event_params) AS params WHERE params.key = 'page_location') AS url
    , (SELECT value.string_value FROM UNNEST(event_params) AS params WHERE params.key = 'page_title') AS title
    , (SELECT UPPER(REGEXP_EXTRACT(value.string_value, r'/([^/]+)/?$'))  FROM UNNEST(event_params) AS params WHERE params.key = 'page_location') AS item
  FROM
    `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` as T
  WHERE
    event_name ='add_to_cart'
    AND _TABLE_SUFFIX BETWEEN '20201201' AND '20201231'
    AND geo.country = 'United States'
)
SELECT DISTINCT
  item
  , title
  , url
FROM
  COTENT_URL_TITLE_ID
WHERE
  item not like '%.HTML'
;
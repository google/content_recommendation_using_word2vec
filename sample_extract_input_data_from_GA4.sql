/*Copyright 2023 Google LLC.
This solution, including any related sample code or data, is made available on an “as is,” “as available,” and “with all faults” basis, solely for illustrative purposes, and without warranty or representation of any kind. This solution is experimental, unsupported and provided solely for your convenience. Your use of it is subject to your agreements with Google, as applicable, and may constitute a beta feature as defined under those agreements. To the extent that you make any data available to Google in connection with your use of the solution, you represent and warrant that you have all necessary and appropriate rights, consents and permissions to permit Google to use and process that data. By using any portion of this solution, you acknowledge, assume and accept all risks, known and unknown, associated with its usage, including with respect to your deployment of any portion of this solution in your systems, or usage in connection with your business, if at all.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License. */

WITH USER_ID_AND_CONTENT_DATA AS (
  SELECT DISTINCT
    user_pseudo_id AS user_id,
    event_timestamp as t,
    UPPER(REGEXP_EXTRACT(prm.value.string_value, r'/([^/]+)/?$')) AS item,
  FROM
    `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` as T
    CROSS JOIN
      T.event_params AS prm
  WHERE
    event_name = 'add_to_cart'
    AND prm.key = 'page_location'
    AND _TABLE_SUFFIX BETWEEN '20201201' AND '20201231'
    AND geo.country = 'United States'
)
SELECT
  user_id,
  STRING_AGG(item, ',' ORDER BY t asc) AS item_list,
  COUNT(item) AS cnt
FROM
  USER_ID_AND_CONTENT_DATA 
WHERE
  item not like '%.HTML'
GROUP BY
  user_id
;
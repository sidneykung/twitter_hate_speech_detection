# Twitter Hate Speech Detection

## Overview


## Business Understanding


## Data & Methods
(explain where data came from)

Temporary column names:
(Replace with a table with initial engineered columns too- label, username and tweet)

- `count` : number of CrowdFlower users who coded each tweet (min is 3, sometimes more users coded a tweet when judgments were determined to be unreliable by CF).
- `hate_speech` : number of CF users who judged the tweet to be hate speech.
offensive_language = number of CF users who judged the tweet to be offensive.
- `neither` : number of CF users who judged the tweet to be neither offensive nor non-offensive.
- `class` : class label for majority of CF users. 0 - hate speech 1 - offensive language 2 - neither

Reference:
Davidson, T., Warmsley, D., Macy, M. and Weber, I., 2017. Automated Hate Speech Detection and the Problem of Offensive Language. ArXiv,.

## Results


## Conclusion

## Repository Contents

Notebooks Order
1. data_cleaning.ipynb
2. nlp_preprocessing.ipynb
3. eda_notebook.ipynb

## For More Information

See the [full model process](link) in the`final_notebook_name` Jupyter Notebook.
For additional info, contact Sidney Kung at sidneyjkung@gmail.com

# Twitter Hate Speech Detection

## Overview


## Business Understanding

Human content moderation exploits people by consistently traumatizing and underpaying them, both domestically and overseas. In 2019, an article on The Verge exposed the extensive list of horrific working conditions that employees faced at Cognizant, which was Facebook’s primary moderation contractor. Unfortunately, every major tech company, including Twitter, uses human moderators to some extent. 

This project aims to automate content moderation to identify hate speech using machine learning binary classification algorithms. Hate speech is defined abusive or threatening speech that expresses prejudice against a particular group, especially on the basis of race, religion or sexual orientation. Usually, the difference between hate speech and offensive language comes down to subtle context or diction. Any company with a website where users post content will benefit from automating as much as the moderation process as possible.

**Please note:** Because of the subject matter of this project, all notebooks contain offensive language from the dataset.

## Data & Methods

The dataset for this capstone project comes from a study called Automated Hate Speech Detection and the Problem of Offensive Language  conducted by Thomas Davidson and a team at Cornell University in 2017. The GitHub repository can be found [here](). The dataset is provided as a `.csv` file with 24,802 text posts from Twitter where 6% of the tweets were labeled as hate speech. 

Since content moderation is so subjective, the labels on this dataset were crowdsourced and determined by majority-rules. The “class” column labels each tweet as 0 for hate speech, 1 for offensive language or 2 for neither. In order to create a different project and adapt the data to my specific business context, I will be treating the data as a binary classification problem. 

Therefore, the final model will be predicting whether a tweet is hate speech or not. To prepare the data for this, I will be manually replacing existing 1 and 2 values as 0, and replacing 0 as 1 to indicate hate speech.

(convert to table)
- `total_votes`: number of CrowdFlower users who coded each tweet (minimum is 3, sometimes more users coded a tweet when judgments were determined to be unreliable by CF).
- `hate_speech_votes`: number of CF users who judged the tweet to be hate speech.
- `other_votes`: number of CF users who judged the tweet to be offensive language or neither.
- `label`: class label for majority of CF user votes. 1 - hate speech 0 - not hate speech
- `tweet`: raw tweets

## Results


## Conclusion


## Repository Contents

- `models`: folder with model iterations
- `pickle`: folder with pickled dataframes and models
- `preprocessing`: folder with original data, data cleaning notebook and EDA notebook
- `visuazations` : folder with graphs & images

## For More Information

See the [full model process](link) in the `final_notebook_name.ipynb` Jupyter Notebook.

For additional info, contact Sidney Kung at sidneyjkung@gmail.com

## References:

Davidson, T., Warmsley, D., Macy, M. and Weber, I., 2017. Automated Hate Speech Detection and the Problem of Offensive Language. ArXiv,.

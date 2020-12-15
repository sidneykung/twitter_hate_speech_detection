# Twitter Hate Speech Detection
## *Can content moderation be automated?*
#### Project ongoing until January 6

## Overview
***One-paragraph description of the project, including the business problem, data, methods, results, and reccomendations.***


## Business Problem
Human content moderation exploits people by consistently traumatizing and underpaying them, both domestically and overseas. In 2019, an article on The Verge exposed the extensive list of horrific working conditions that employees faced at Cognizant, which was Facebook’s primary moderation contractor. Unfortunately, every major tech company, including Twitter, uses human moderators to some extent. 

This project aims to automate content moderation to identify hate speech using machine learning binary classification algorithms. Hate speech is defined as abusive or threatening speech that expresses prejudice against a particular group, especially on the basis of race, religion or sexual orientation. Usually, the difference between hate speech and offensive language comes down to subtle context or diction. Any company with a website where users post content will benefit from automating as much as the moderation process as possible.

**Please note:** Because of the subject matter of this project, all notebooks contain offensive language from the dataset.

***Summary of the business problem you are trying to solve, and the data questions that you plan to answer in order to solve them.***

Questions to consider:
- What are the business's pain points related to this project?
- How did you pick the data analysis question(s) that you did?
- Why are these questions important from a business perspective?

## Data & Methods
The dataset for this capstone project comes from a study called Automated Hate Speech Detection and the Problem of Offensive Language  conducted by Thomas Davidson and a team at Cornell University in 2017. The GitHub repository can be found [here](https://github.com/t-davidson/hate-speech-and-offensive-language). The dataset is provided as a `.csv` file with 24,802 text posts from Twitter where 6% of the tweets were labeled as hate speech. 

Since content moderation is so subjective, the labels on this dataset were crowdsourced and determined by majority-rules. The “class” column labels each tweet as 0 for hate speech, 1 for offensive language or 2 for neither. In order to create a different project and adapt the data to my specific business context, I will be treating the data as a binary classification problem. 

Therefore, the final model will be predicting whether a tweet is hate speech or not. To prepare the data for this, I will be manually replacing existing 1 and 2 values as 0, and replacing 0 as 1 to indicate hate speech.

(convert to table)
- `total_votes`: number of CrowdFlower users who coded each tweet (minimum is 3, sometimes more users coded a tweet when judgments were determined to be unreliable by CF).
- `hate_speech_votes`: number of CF users who judged the tweet to be hate speech.
- `other_votes`: number of CF users who judged the tweet to be offensive language or neither.
- `label`: class label for majority of CF user votes. 1 - hate speech 0 - not hate speech
- `tweet`: raw tweets

***Methods: Describe the process for analyzing or modeling the data.***

Questions to consider:
- How did you prepare, analyze or model the data?
- Why is this approach appropriate given the data and the business problem?

## Data Understanding

1. What is the overall polarity of the tweets?
2. What are the most popular hashtags of each tweet type?
3. Which phrases have the most importance in modeling?

## Results
***Present key results***

Questions to consider:
- How do you interpret the results?
- How confident are you that your results would generalize beyond the data you have?

## Recommendations

## Final Modeel Performance

## Conclusion
***Provide conclusions about the work done, including any limitations or next steps.***

Questions to consider:
- What would you recommend the business do as a result of this work?
- What are some reasons why your analysis might not fully solve the business problem?
- What else could you do in the future to improve this project?

## Next Steps
- Evaluate model with new tweet or other online forum data
- LDA Topic Modeling with Gensim
- Deploy MVP on Webapp via StreamLit

## Repository Contents
```bash
.
├── models                             # contains model iterations
├── pickle                             # contains cleaned data
├── preprocessing                      # contains all data preperation iterations and EDA notebooks
│   ├── twitter_data.csv               # raw dataset
│   └── data_cleaning.ipynb            # cleaning raw data to project format
├── visualizations                     # contains visualizations and local images
├── presentation.pdf                   # slide deck
├── README.md                          # public-facing preview
└── final_notebook.ipynb               # final version of EDA, feature engineering and modeing process
```

## For More Information

See the [full project overview](link) in the `final_notebook.ipynb` Jupyter Notebook.

For additional info, contact Sidney Kung at sidneyjkung@gmail.com

## References

Davidson, T., Warmsley, D., Macy, M. and Weber, I., 2017. Automated Hate Speech Detection and the Problem of Offensive Language. ArXiv,.

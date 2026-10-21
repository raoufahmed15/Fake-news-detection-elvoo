# Fake News Detection

A machine learning project for detecting whether a news article is
**Fake** or **Real** using natural language processing (NLP) and
supervised learning.

## Project Overview

This project uses the Fake and Real News Dataset to train
text-classification models. It combines each article's title and body,
preprocesses the text, converts it into numerical features using TF-IDF,
and compares machine-learning models using standard evaluation metrics.

## Objectives

-   Load and explore the Fake and Real News Dataset.
-   Clean article titles and text, remove English stopwords, and apply
    lemmatization when the required NLTK data is available.
-   Convert text into numerical features using TF-IDF.
-   Train and compare Logistic Regression and Linear Support Vector
    Machine (SVM) models.
-   Evaluate performance using accuracy and F1-scores.
-   Inspect confusion matrices and influential words.
-   Visualize frequent words and compare Fake and Real news using word
    clouds.

## Dataset

The project uses the **Fake and Real News Dataset**, commonly
distributed with two CSV files:

-   `Fake.csv` --- articles labeled as Fake.
-   `True.csv` --- articles labeled as Real.

The notebook combines the files and assigns labels:

-   `Fake` → `0`
-   `Real` → `1`

Place both CSV files in the same folder, in a `data` folder, or use the
supported Kaggle input path. The notebook checks common locations
automatically.

## Workflow

1.  **Data loading** --- locate and load `Fake.csv` and `True.csv`.
2.  **Exploratory data analysis** --- inspect dataset information,
    missing values, duplicate rows, and class distribution.
3.  **Text preprocessing** --- combine title and article text, normalize
    the text, remove stopwords, and apply lemmatization when available.
4.  **Train/test split** --- create a stratified split so the class
    proportions are maintained.
5.  **Feature extraction** --- use TF-IDF with unigrams and bigrams,
    fitting the vectorizer on the training set only.
6.  **Model training** --- train and compare:
    -   Logistic Regression
    -   Linear Support Vector Machine (Linear SVM)
7.  **Evaluation** --- compare accuracy and F1-scores and inspect
    confusion matrices.
8.  **Interpretation** --- review terms most strongly associated with
    Fake and Real predictions.
9.  **Prediction example** --- use the trained models to classify new
    article text.
10. **Visualization** --- generate word clouds and frequent-word plots
    for Fake and Real articles.

## Technologies

-   Python
-   Jupyter Notebook
-   Pandas and NumPy
-   NLTK (stopwords and WordNet lemmatization)
-   Scikit-learn
-   TF-IDF vectorization
-   Matplotlib and Seaborn
-   WordCloud

## Run the Project

1.  Download or clone the project repository.

2.  Download the Fake and Real News Dataset and ensure `Fake.csv` and
    `True.csv` are available in a supported location.

3.  Install the required libraries if they are not already installed:

    ``` bash
    pip install pandas numpy matplotlib seaborn nltk scikit-learn wordcloud jupyter
    ```

4.  Open the notebook:

    ``` bash
    jupyter notebook "fake-news-detection.ipynb"
    ```

5.  Run the cells from top to bottom.

> **Note:** NLTK may need to download English stopwords and WordNet
> data. The notebook includes fallback behavior for stopwords and
> optional lemmatization if those resources are unavailable. The
> word-cloud section requires the `wordcloud` package.

## Project Structure

A typical project layout may look like this:

``` text
Fake-News-Detection/
├── fake-news-detection.ipynb
├── data/
│   ├── Fake.csv
│   └── True.csv
└── README.md
```

The notebook also checks common Kaggle dataset locations and the current
directory.

## Evaluation Notes and Limitations

-   The notebook reports measured results after it is run; no accuracy
    or F1-score is assumed in advance.
-   Duplicate and empty articles are removed, and the `subject` and
    `date` columns are not used as model features.
-   Dataset-specific wording or duplicate-related patterns can still
    make the task easier than detecting misinformation from entirely new
    sources. Results should therefore be interpreted with this
    limitation in mind.
-   This is a learning project and should not be treated as a definitive
    fact-checking system.

## Future Improvements

-   Evaluate on articles from independent sources or a time-based
    holdout set.
-   Tune model hyperparameters using cross-validation.
-   Add a reusable prediction interface or web application.
-   Save and load the trained model and TF-IDF vectorizer.
-   Add more robust testing for unseen topics and sources.

## Author

**Raouf Ahmed**

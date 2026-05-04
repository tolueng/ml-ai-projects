# Email Spam Classifier using TensorFlow

## Overview

This project is a Machine Learning / Natural Language Processing project that classifies emails as either:

- **Ham**: normal email / not spam
- **Spam**: unwanted or suspicious email

The project uses Python, text preprocessing, tokenization, padding, and an LSTM neural network built with TensorFlow.

The final model can also test a custom email written by the user and predict whether it is spam or ham.

---

## Project Structure

```text
01-email-spam-classifier/
│
├── README.md
├── requirements.txt
├── spam_email_classifier.ipynb
│
├── data/
│   └── spam_ham_dataset.csv
│
└── results/
    └── optional saved charts and outputs
```

### What each file/folder is for

| File / Folder | Purpose |
|---|---|
| `README.md` | Explains the project, how it works, and how to run it |
| `requirements.txt` | Lists the Python packages needed for this project |
| `spam_email_classifier.ipynb` | Main Jupyter Notebook containing the full project code |
| `data/` | Stores the dataset used by the project |
| `data/spam_ham_dataset.csv` | The email dataset used for training and testing |

---

## How to Install the Required Packages

Before running the notebook, install the required Python packages.

If you are using Jupyter Notebook or VS Code Notebook, run this in a notebook cell:

```python
%pip install -r requirements.txt
```

If you are using the terminal, run:

```bash
pip install -r requirements.txt
```

---

## Requirements

The `requirements.txt` file should contain:

```txt
numpy
pandas
matplotlib
seaborn
nltk
wordcloud
tensorflow
scikit-learn
```

---

## Why These Libraries Are Used

| Library | Why it is used |
|---|---|
| `numpy` | Used for numerical operations and array handling |
| `pandas` | Used to load, view, clean, and manage the dataset |
| `matplotlib` | Used to create charts and visualizations |
| `seaborn` | Used to create cleaner statistical plots, such as label count plots |
| `string` | Used to access punctuation characters for text cleaning |
| `nltk` | Used for Natural Language Processing tasks |
| `stopwords` from `nltk` | Used to remove common words like “the”, “is”, “and”, etc. |
| `wordcloud` | Used to visualize common words in spam and ham emails |
| `tensorflow` | Used to build and train the deep learning model |
| `Tokenizer` | Converts text into numerical sequences that the model can understand |
| `pad_sequences` | Makes all email sequences the same length |
| `train_test_split` | Splits the dataset into training and testing data |
| `EarlyStopping` | Stops training early if the model stops improving |
| `ReduceLROnPlateau` | Reduces the learning rate when validation loss stops improving |
| `warnings` | Used to hide unnecessary warning messages |

Some libraries, such as `string` and `warnings`, do not appear in `requirements.txt` because they are already built into Python.

---

## Dataset

The dataset is loaded using:

```python
data = pd.read_csv("data/spam_ham_dataset.csv")
```

The dataset contains email messages and their labels.

The important columns are:

| Column | Meaning |
|---|---|
| `text` | The email content |
| `label` | Whether the email is `ham` or `spam` |

---

## Step-by-Step Explanation of the Code

## Step 1: Import Required Libraries

The project starts by importing all required libraries.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

These libraries are used for working with data and creating visualizations.

```python
import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
nltk.download("stopwords")
```

These are used for text cleaning and Natural Language Processing.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
```

These are used to build, train, and improve the deep learning model.

---

## Step 2: Load the Dataset

```python
data = pd.read_csv("data/spam_ham_dataset.csv")
data.head()
```

This reads the dataset and shows the first few rows.

```python
data.shape
```

This checks the size of the dataset.

For example, if the output is:

```text
(5171, 4)
```

that means the dataset has:

- 5,171 rows
- 4 columns

In this project, each row represents one email.

---

## Step 3: Visualize the Original Label Distribution

```python
sns.countplot(x="label", data=data)
plt.show()
```

This shows how many emails are spam and how many are ham.

This step is important because many spam datasets are unbalanced. Usually, there are more ham emails than spam emails.

---

## Step 4: Balance the Dataset

```python
ham_msg = data[data["label"] == "ham"]
spam_msg = data[data["label"] == "spam"]
```

This separates the dataset into two groups:

- ham emails
- spam emails

```python
ham_msg_balanced = ham_msg.sample(spam_msg.shape[0], random_state=42)
```

This randomly selects the same number of ham emails as spam emails.

This is done because if the dataset has too many ham emails, the model may become biased and predict most emails as ham.

The `random_state=42` makes the random selection reproducible. That means the same rows will be selected every time the notebook runs.

```python
balanced_data = pd.concat([ham_msg_balanced, spam_msg], axis=0).reset_index(drop=True)
```

This combines the balanced ham emails and spam emails into one new dataset.

```python
balanced_data.shape
```

This checks the size of the new balanced dataset.

```python
sns.countplot(x="label", data=balanced_data)
plt.title("Balanced Dataset Distribution of Spam and Ham Messages")
plt.xticks(ticks=[0, 1], labels=["Ham", "Spam"])
plt.show()
```

This shows the new balanced distribution.

---

## Step 5: Clean the Email Text

Before training the model, the email text must be cleaned.

### Remove the word `Subject`

```python
balanced_data["text"] = balanced_data["text"].str.replace("Subject", "")
```

Many emails contain the word `Subject`. This word appears often and may not help the model understand whether an email is spam or ham.

---

### Remove Punctuation

```python
punctuations_list = string.punctuation

def remove_punctuations(text):
    temp = str.maketrans("", "", punctuations_list)
    return text.translate(temp)
```

This function removes punctuation such as:

```text
! ? . , : ; " ' ( )
```

Then it is applied to the email text:

```python
balanced_data["text"] = balanced_data["text"].apply(lambda x: remove_punctuations(x))
```

This helps simplify the text before training.

---

### Remove Stopwords

```python
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    imp_words = []

    for word in str(text).split():
        word = word.lower()

        if word not in stop_words:
            imp_words.append(word)

    output = " ".join(imp_words)
    return output
```

Stopwords are common words such as:

```text
the
is
and
to
for
in
```

These words are often removed because they appear frequently and may not provide strong meaning for spam detection.

Then the function is applied:

```python
balanced_data["text"] = balanced_data["text"].apply(lambda x: remove_stopwords(x))
```

---

## Step 6: Create Word Clouds

```python
def plot_wordcloud(data, typ):
    email_corpus = " ".join(data["text"])
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=100
    ).generate(email_corpus)

    plt.figure(figsize=(7, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud for {typ} Emails", fontsize=15)
    plt.show()
```

A word cloud shows the most common words in a group of emails.

The project creates two word clouds:

```python
plot_wordcloud(balanced_data[balanced_data["label"] == "ham"], typ="Non-spam")
plot_wordcloud(balanced_data[balanced_data["label"] == "spam"], typ="Spam")
```

This helps compare common words in ham emails and spam emails.

---

## Step 7: Tokenization and Padding

Machine learning models cannot understand raw text directly. Text must be converted into numbers.

```python
max_words = 10000
max_len = 100
```

### `max_words = 10000`

This means the tokenizer will keep up to the 10,000 most common words.

### `max_len = 100`

This means every email will be represented using 100 word positions.

If an email is shorter than 100 words, padding is added.

If an email is longer than 100 words, it is shortened.

---

## Step 8: Split the Data

```python
train_X, test_X, train_Y, test_Y = train_test_split(
    balanced_data["text"],
    balanced_data["label"],
    test_size=0.2,
    random_state=42,
    stratify=balanced_data["label"]
)
```

This splits the dataset into:

- training data
- testing data

`test_size=0.2` means 20% of the data is used for testing and 80% is used for training.

`stratify=balanced_data["label"]` helps keep the spam and ham ratio balanced in both the training and testing sets.

---

## Step 9: Convert Text to Numbers

```python
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_X)
```

The tokenizer learns the vocabulary from the training emails.

`oov_token="<OOV>"` is used for words that were not seen during training.

OOV means:

```text
Out Of Vocabulary
```

Then the text is converted into number sequences:

```python
train_sequences = tokenizer.texts_to_sequences(train_X)
test_sequences = tokenizer.texts_to_sequences(test_X)
```

---

## Step 10: Pad the Sequences

```python
train_sequences = pad_sequences(
    train_sequences,
    maxlen=max_len,
    padding="post",
    truncating="post"
)

test_sequences = pad_sequences(
    test_sequences,
    maxlen=max_len,
    padding="post",
    truncating="post"
)
```

This makes all email sequences the same length.

The model expects every input to have the same size, so padding is necessary.

---

## Step 11: Convert Labels to Numbers

```python
train_Y = (train_Y == "spam").astype(int)
test_Y = (test_Y == "spam").astype(int)
```

The labels are converted like this:

| Original Label | Numeric Label |
|---|---:|
| `ham` | 0 |
| `spam` | 1 |

The model needs numeric labels because it cannot train directly on text labels.

---

## Step 12: Build the Model

```python
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(max_len,)),

    tf.keras.layers.Embedding(
        input_dim=max_words,
        output_dim=32
    ),

    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
```

### Model Architecture

| Layer | Purpose |
|---|---|
| `Input` | Defines the input shape |
| `Embedding` | Converts word numbers into dense vector representations |
| `LSTM` | Learns patterns from word sequences |
| `Dense` | Learns higher-level patterns |
| `Sigmoid Output` | Outputs a probability between 0 and 1 |

The final layer uses:

```python
activation="sigmoid"
```

This is used because this is a binary classification problem:

- ham
- spam

The output is a number between 0 and 1.

A value closer to 1 means spam.

A value closer to 0 means ham.

---

## Step 13: Compile the Model

```python
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
```

### Why `binary_crossentropy`?

This project has two possible labels: ham or spam. So binary crossentropy is the correct loss function.

### Why `adam`?

Adam is an optimizer that helps update the model weights during training.

### Why `accuracy`?

Accuracy shows how often the model predicts the correct label.

---

## Step 14: Train the Model

```python
es = EarlyStopping(
    patience=3,
    monitor="val_accuracy",
    restore_best_weights=True
)

lr = ReduceLROnPlateau(
    patience=2,
    monitor="val_loss",
    factor=0.5,
    verbose=0
)
```

### `EarlyStopping`

Stops training if the validation accuracy does not improve after a few epochs.

This helps prevent overfitting.

### `ReduceLROnPlateau`

Reduces the learning rate when validation loss stops improving.

This can help the model continue learning more carefully.

The model is trained using:

```python
history = model.fit(
    train_sequences,
    train_Y,
    validation_data=(test_sequences, test_Y),
    epochs=20,
    batch_size=32,
    callbacks=[lr, es]
)
```

---

## Step 15: Evaluate the Model

```python
test_loss, test_accuracy = model.evaluate(test_sequences, test_Y)
print("Test Loss :", test_loss)
print("Test Accuracy :", test_accuracy)
```

This checks how well the model performs on the test data.

The test data is data that the model did not train on.

---

## Step 16: Plot Model Accuracy

```python
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()
```

This chart compares:

- training accuracy
- validation accuracy

This helps show whether the model is learning well.

---

## Step 17: Test Your Own Email

The notebook includes a function to test a new email.

```python
def clean_new_email(email_text):
    email_text = str(email_text)
    email_text = email_text.replace("Subject", "")
    email_text = email_text.replace("subject", "")
    email_text = remove_punctuations(email_text)
    email_text = remove_stopwords(email_text)

    return email_text
```

This cleans the new email using the same cleaning steps used on the training data.

This is very important. A new email must be processed the same way as the training emails.

---

## Prediction Function

```python
def predict_email(email_text):
    cleaned_email = clean_new_email(email_text)

    email_sequence = tokenizer.texts_to_sequences([cleaned_email])

    email_padded = pad_sequences(
        email_sequence,
        maxlen=max_len,
        padding="post",
        truncating="post"
    )

    spam_probability = model.predict(email_padded, verbose=0)[0][0]

    if spam_probability >= 0.5:
        prediction = "Spam"
    else:
        prediction = "Ham"

    return {
        "Original Email": email_text,
        "Cleaned Email": cleaned_email,
        "Spam Probability": float(spam_probability),
        "Prediction": prediction
    }
```

The model returns a spam probability.

If the probability is greater than or equal to `0.5`, the email is classified as spam.

If the probability is less than `0.5`, the email is classified as ham.

---

## Example: Test a Spam Email

```python
my_email = """
Subject: Congratulations! You have won a free iPhone.
Click this link now to claim your prize.
"""

result = predict_email(my_email)
result
```

Expected type of output:

```python
{
    "Original Email": "...",
    "Cleaned Email": "...",
    "Spam Probability": 0.91,
    "Prediction": "Spam"
}
```

The exact probability may be different depending on training results.

---

## Example: Test a Ham Email

```python
my_email = """
Subject: Team Meeting Tomorrow

Hi everyone,

Just a reminder that we have our team meeting tomorrow at 10 AM.
Please review the project notes before the meeting.

Best,
Sarah
"""

result = predict_email(my_email)
result
```

Expected prediction:

```text
Ham
```

---

## Important Notes

## Do not create a new tokenizer for testing

When testing your own email, the notebook uses the same tokenizer that was fitted on the training data.

This is important because the model learned using that tokenizer's word numbers.

Do not do this:

```python
new_tokenizer = Tokenizer()
```

Instead, use the existing:

```python
tokenizer
```

---

## Do not upload private emails

If this project is uploaded to GitHub, do not include real private emails.

Use fake examples for testing, such as:

```text
Subject: Meeting tomorrow
Please review the notes before class.
```

or:

```text
Subject: You won a free prize
Click now to claim your reward.
```

---

## How to Run the Project

1. Clone or download the repository.

2. Open the project folder:

```bash
cd 01-email-spam-classifier
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

Or inside a notebook:

```python
%pip install -r requirements.txt
```

4. Make sure the dataset is located here:

```text
data/spam_ham_dataset.csv
```

5. Open the notebook:

```text
spam_email_classifier.ipynb
```

6. Run the notebook cells from top to bottom.

7. At the end, test your own email using:

```python
result = predict_email(my_email)
result
```

---

## Possible Improvements

This project can be improved in the future by:

- Saving the trained model
- Creating a Streamlit web app
- Adding a confusion matrix
- Adding precision, recall, and F1-score
- Testing different model architectures
- Using more advanced NLP models
- Improving preprocessing

---

## Project Summary

This project shows a complete beginner-friendly NLP workflow:

1. Load email data
2. Explore label distribution
3. Balance the dataset
4. Clean the text
5. Visualize common words
6. Convert text into numbers
7. Train an LSTM model
8. Evaluate model accuracy
9. Test custom emails

The goal is to understand how machine learning can be used to detect spam emails using Python and TensorFlow.
# SMS Spam Classifier (TF-IDF + Multinomial Naive Bayes)

Simple SMS spam detection model using **TF-IDF vectorization** and **Multinomial Naive Bayes**, trained on the classic SMS Spam Collection dataset.

## What it does

- Loads and cleans the SMS dataset (`ham` / `spam` labels)
- Converts text to numeric features using `TfidfVectorizer`
- Trains a `MultinomialNB` classifier
- Evaluates with accuracy, precision, recall, F1-score, confusion matrix, and cross-validation
- Tests the model on new/unseen messages, including a separate 1000-message test file

## Requirements

```bash
pip install pandas scikit-learn
```

## Files needed

Place these two files in the same folder as the notebook (or update the paths inside):

- `SMSSpamCollection.txt` — training dataset (tab-separated: `label<TAB>message`)
- `1000_unseen_texts.csv` — extra test set with columns `text` and `actual_label`

## How to run

1. **Clone / download** this repo and open `countVector.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. **Update file paths** in the notebook — it's currently set up for Google Colab (`/content/...`). If running locally, change:
   ```python
   "/content/SMSSpamCollection.txt"  →  "SMSSpamCollection.txt"
   "/content/1000_unseen_texts.csv"  →  "1000_unseen_texts.csv"
   ```
3. **Run the cells in order**, top to bottom:
   - Cell 1: loads data into a DataFrame
   - Cell 2: splits data, applies TF-IDF
   - Cell 3: trains the model, prints metrics
   - Cell 4: 6-fold cross-validation
   - Cell 5: predicts on custom sample messages
   - Cell 6: evaluates on the 1000 unseen-text file
   - Last cell: standalone demo of `CountVectorizer` on a toy example

That's it — final accuracy/precision/recall will print in the output cells.

## Notes

- `alpha=0.01` in `MultinomialNB` controls smoothing — lower values fit the training data more tightly.
- Swap `TfidfVectorizer` for `CountVectorizer` if you want to compare Bag-of-Words vs TF-IDF performance.

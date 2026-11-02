# 🧠 Next Word Prediction with LSTM

An interactive Streamlit web app that predicts the next word in a sentence using a deep learning LSTM model built with TensorFlow and Keras.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)
![License](https://img.shields.io/badge/License-MIT-green)

<!-- Add a screenshot: save it as screenshot.png in the repo and uncomment the line below -->
<!-- ![App screenshot](screenshot.png) -->

## Overview

Type the start of a sentence and the model suggests the most likely next words, along with how confident it is in each one. You can pick a suggestion to build your sentence word by word, or let the model generate several words automatically.

## Features

- **Live preview:** your sentence with the predicted word highlighted
- **Top suggestions:** the most likely next words with probability bars
- **Confidence gauge:** shows how sure the model is about its top prediction
- **One-click building:** select a suggestion to add it to your sentence
- **Multi-word generation:** continue a sentence by 1 to 10 words
- **Extras:** example prompts, adjustable settings and a history of generations
- **Modern dark UI:** glass-style cards with gradient accents

## How it works

1. The input text is converted to integers with a fitted Keras `Tokenizer`.
2. The sequence is padded on the left to length `max_len - 1`.
3. The LSTM model outputs a probability for every word in the vocabulary.
4. The words with the highest probabilities are shown as predictions.

For multi-word generation, the top prediction is appended to the text and the process repeats.

## Tech stack

- Python
- TensorFlow / Keras (LSTM)
- Streamlit
- NumPy

## Project structure

```
next-word-prediction/
├── app.py                  # Streamlit app
├── lstm_model.h5           # Trained Keras model
├── tokenizer.pkl           # Fitted Keras Tokenizer
├── max_len.pkl             # Maximum sequence length
├── requirements.txt        # Dependencies
├── .streamlit/
│   └── config.toml         # Dark theme settings
├── LICENSE
└── README.md
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/abhiii04-25/next-word-prediction.git
cd next-word-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Model files

The app needs these three files in the project root:

| File | Purpose |
|------|---------|
| `lstm_model.h5` | Trained LSTM model |
| `tokenizer.pkl` | Tokenizer fitted on the training text |
| `max_len.pkl` | Length of the longest training sequence |

The model should take an input of `max_len - 1` tokens and output a probability for each word in the tokenizer's vocabulary. If a file is over 100 MB, use [Git LFS](https://git-lfs.com) to store it on GitHub.

## Example

| Input | Predicted next word |
|-------|---------------------|
| `to be or not to` | `be` |
| `once upon a` | `time` |

Results depend on the text your model was trained on.

## Deployment

You can deploy this app free on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push the repository to GitHub.
2. Sign in to Streamlit Community Cloud and choose **New app**.
3. Select your repository, branch `main` and the main file `app.py`.

If the model fails to load after deployment, pin the TensorFlow version you trained with in `requirements.txt`.

## Future improvements

- Temperature sampling for more varied predictions
- Support for a larger or custom training dataset
- Light and dark theme toggle
- Model comparison (LSTM vs GRU vs Transformer)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

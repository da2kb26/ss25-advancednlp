
# Deep Averaging Network (DAN) for Sentiment Analysis

This project implements a **Deep Averaging Network (DAN)** using **PyTorch** for binary sentiment classification. It classifies input sentences as **Positive** or **Negative**, based on real-world product and movie reviews.

---

## 📌 Overview

- 🔤 Tokenization: Uses **Stanza** to split sentences into words.
- 🧠 Model: DAN (Deep Averaging Network) — a simple but effective neural network.
- 🗃 Dataset: Real reviews from Amazon, IMDB, and Yelp.
- 🔍 Task: Binary classification — `Positive (1)` or `Negative (0)`.
- 👨‍💻 Live prediction: You can enter your own sentence to see the sentiment result.

---

## 📁 Dataset

The dataset comes from the **UCI Sentiment Labelled Sentences Data Set** and is located in the `data/` folder. It includes:

- `amazon_cells_labelled.txt`
- `imdb_labelled.txt`
- `yelp_labelled.txt`

**Format:**
```
<sentence> \t <label>
```
- `label` is either `0` (Negative) or `1` (Positive)

---

## 🏗️ Model Architecture (DAN)

1. **Embedding Layer**: Converts word IDs into vector representations.
2. **Average Layer**: Averages the embeddings of all words in a sentence.
3. **Fully Connected Layers**: Two linear layers with ReLU and Dropout.
4. **Output Layer**: Predicts class (positive/negative).

---

## 🧪 Training Details

- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Epochs: 10
- Batch Size: 64
- Accuracy and classification report generated after training.

---

## ▶️ How to Run

### ✅ Requirements

- Python 3.8+
- PyTorch
- Stanza
- scikit-learn

### 📦 Install Dependencies

```bash
pip install torch stanza scikit-learn
```

### 🧠 Download Stanza English Model

```bash
python -c "import stanza; stanza.download('en')"
```

### 🚀 Run the Script

```bash
python dan_model.py
```

After training, you'll be able to enter your own sentence like:

```bash
Enter a sentence (or type 'exit'): I really enjoyed the movie!
Prediction: Positive
```

---

## 🧠 Note

This implementation does **not use the pretrained model** from the original paper (Cer et al., 2018).  
Instead, it **reimplements the DAN model from scratch** using PyTorch and trains it on a small review dataset for educational purposes.


---

## 📚 Reference

- Original DAN paper: [Cer et al., 2018 - Universal Sentence Encoder](https://arxiv.org/abs/1803.11175)
- Dataset: UCI Sentiment Labelled Sentences

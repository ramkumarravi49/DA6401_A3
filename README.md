# DA6401 Assignment 3: Transliteration with Sequence-to-Sequence Models

This repository contains code and artifacts for the DA6401 Assignment 3 on transliteration using RNN-based sequence-to-sequence models with and without attention. The models take a Latin-script word as input and output the corresponding Devanagari-script word.

---

## 📌 Table of Contents

* [Overview](#overview)
* [Directory Structure](#directory-structure)
* [Setup & Installation](#setup--installation)
* [Training the Model](#training-the-model)
* [Testing the Model](#testing-the-model)
* [Inference & Evaluation](#inference--evaluation)
* [External Resources](#external-resources)
* [Wandb Link](#wandb-report-link)
* [Acknowledgements](#acknowledgements)

---

## 📖 Overview

This project uses RNN, GRU, and LSTM cells in encoder-decoder architecture, both with and without Bahdanau attention. The models are trained on the [Dakshina dataset](https://github.com/google-research-datasets/dakshina), Hindi subset.

* Best model: LSTM with attention
* Dataset and checkpoints are hosted on Google Drive
* Exploratory notebooks and Weights & Biases sweeps were carried out in Kaggle (see below)

---

## 🗂️ Directory Structure

### GitHub Repo Structure

```
DA6401_A3-main/
├── code/                     # All train/test/model code and required data
│   ├── helper.py
│   ├── model.py
│   ├── test.py
│   ├── train.py
│   ├── attn_best_model.pth
│   └── dakshina_dataset_v1.0/
│       └── hi/
│           └── lexicons/
│               ├── hi.translit.sampled.train.tsv
│               ├── hi.translit.sampled.dev.tsv
│               └── hi.translit.sampled.test.tsv
│
├── attention_inferences/    # Additional outputs (connectivity, predictions)
│   ├── corrected_cases.tsv
│   ├── connectivity_*.png
│   ├── connectivity_gif/
│   └── predictions_attention/
│
├── vanilla_inferences/
│   ├── no-attn-full-code.ipynb
│   └── predictions_vanilla/
│
├── LICENSE
└── README.md
```

### Local Expected Directory Structure

After downloading from Google Drive, place the following **inside the `code/` directory**:

```
<project_root>/code/
├── attn_best_model.pth
└── dakshina_dataset_v1.0/
    └── hi/
        └── lexicons/
            ├── hi.translit.sampled.train.tsv
            ├── hi.translit.sampled.dev.tsv
            └── hi.translit.sampled.test.tsv
```

---

## ⚙️ Setup & Installation

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### `requirements.txt`

```txt
torch
numpy
tqdm
```

---

## 🚀 Training the Model

Train your model using `train.py`. The script supports command-line configuration via `argparse`.

### Training With Attention (Best Defaults Preloaded)

```bash
python code/train.py --attention
```

This will train the model using the default best hyperparameters:

* emb\_size: 256
* hidden\_size: 512
* enc\_layers: 2
* cell: LSTM
* dropout: 0.5
* lr: 1e-4
* batch\_size: 64
* epochs: 20

The best model will be saved as `attn_best_model.pth`.

### Training Without Attention

```bash
python code/train.py
```

This will run the default configuration **without** attention.
To explicitly change hyperparameters:

```bash
python code/train.py --cell GRU --epochs 10 --hidden_size 256
```

---

## Testing the Model

To test the best model mentioned in report stored at `attn_best_model.pth`, run:

```bash
python code/test.py
```

This uses batch size = 1 for accurate exact-match evaluation.

---

## 📊 Inference & Evaluation (Kaggle Notebooks)

The full exploratory and post-analysis work (Q4c, Q5c, Q6) has been done in Kaggle notebooks. These include:

* Error analytics
* Generating `corrected_cases.tsv`
* Attention heatmaps
* Connectivity visualizations

### 🔗 Kaggle Notebooks

* [No Attention - Full Code](https://www.kaggle.com/code/ramkumarrcs24m037/no-attn-full-code)
* [With Attention - Full Code](https://www.kaggle.com/code/ramkumarrcs24m037/attn-final-full-code)

> **Note:** All Weights & Biases sweeps were executed within these Kaggle notebooks (1B sweeps).

---

## 🔗 External Resources

* 📁 **Google Drive (Dataset + Checkpoint)**:
  [Download here](https://drive.google.com/drive/folders/1fyO9_v5fhBJBK-j5GgSF2JAhjZEoMmPl?usp=sharing)

  Contents:

  * `dakshina_dataset_v1.0/`
  * `attn_best_model.pth`

After download, place both items inside the `code/` directory as shown above.

---
## 🔗 WandB Report Link
* [Wandb A3 Rport Link](https://api.wandb.ai/links/cs24m037-iit-madras/yhbccmti)
* [Wandb A3 Project Link](https://wandb.ai/cs24m037-iit-madras/DA6401_A3/sweeps)


## 👨‍💻 Author

**Ramkumar R**
M.Tech CSE, IIT Madras
Roll No: CS24M037



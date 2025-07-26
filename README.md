# Fine-Tuned-DistilBERT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pathum987/Fine-Tuned-DistilBERT/blob/main/Fine_Tuned_DistilBERT.ipynb)

This project fine-tunes the DistilBERT transformer model on the IMDb movie reviews dataset to classify reviews as **positive** or **negative**.

---

## 📌 Project Overview

This project demonstrates a complete workflow for a text classification task using modern NLP tools.

- **Dataset**: Uses the Keras IMDb dataset with balanced sampling.  
- **Preprocessing**: Decodes integer-encoded reviews back to raw text for processing.  
- **Tokenization**: Tokenizes input using the Hugging Face DistilBERT tokenizer.  
- **Fine-Tuning**: Fine-tunes the `distilbert-base-uncased` model using the Hugging Face Trainer API.  
- **Evaluation**: Evaluates the model's performance on a held-out test set.  
- **Inference**: Provides a ready-to-use pipeline for predicting sentiment on new, unseen reviews.  

---

## ✅ Results

The fine-tuned model achieves a final **accuracy of 90.70%** on the test set, demonstrating its effectiveness in understanding and classifying sentiment in text.

---

## ⚙️ Installation

To run this project locally, clone the repository and ensure you have the required libraries installed:

```bash
# Clone the repository
git clone https://github.com/Pathum987/Fine-Tuned-DistilBERT.git

# Navigate to the project directory
cd Fine-Tuned-DistilBERT

# Install the required libraries manually
pip install transformers torch tensorflow numpy
```

> 📌 Note: There is no `requirements.txt` file in this repository. You can install dependencies using the above command or generate your own by running `pip freeze > requirements.txt` in your environment.

---

## 🚀 Usage

The entire process, from data loading to model evaluation and inference, is contained within the Jupyter Notebook.

- **Run Locally**: Open the `Fine_Tuned_DistilBERT.ipynb` notebook in a Jupyter environment.
- **Run in Colab**: Click the "Open in Colab" badge at the top of this README to launch the notebook directly in Google Colab.

---

## 📂 File Descriptions

- `Fine_Tuned_DistilBERT.ipynb`: The main notebook containing all the code for data processing, model training, and evaluation.  
- `README.md`: This file, providing an overview and guide to the project.  

---

## 📄 License

This project is open-source and available under the MIT License.

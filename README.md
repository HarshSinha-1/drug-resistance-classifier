🧬 DeepResist: Genomic Drug Resistance Classifier

📌 Project Overview

DeepResist is a Deep Learning model designed to predict whether a specific DNA sequence belongs to a Drug-Resistant gene (superbug) or is a Susceptible (harmless/normal) sequence.

Antimicrobial Resistance (AMR) is a global health threat. This project treats DNA sequences as a "language" and uses Natural Language Processing (NLP) techniques combined with a 1D Convolutional Neural Network (CNN) to detect patterns of resistance.

🚀 Key Results

The model was trained on the MEGARes v3.00 database. After balancing the dataset (handling the class imbalance problem), the model achieved state-of-the-art performance.

Metric

Value

Note

Training Accuracy

99.72%

Model learned the patterns perfectly

Validation Accuracy

99.08%

Model generalizes well to new data

Dataset Size

10,174 Sequences

5087 Resistant / 5087 Susceptible

📈 Performance Leap: The model started at 51% accuracy (random guessing) in Epoch 1 and evolved to 99% accuracy by Epoch 20, demonstrating rapid and effective learning.

🧠 How It Works (The Concept)

Imagine DNA as a long book written with only 4 letters: A, T, G, C.

Resistant genes have specific "spelling mistakes" (mutations) or specific "paragraphs" that allow them to survive antibiotics.

Susceptible genes do not have these specific patterns.

The Pipeline

Input: Raw DNA Sequence (ATGCGT...)

K-mer Tokenization: Breaking the DNA into overlapping "words" (K=6).

Example: ATGCG → ATGC, TGCG

Encoding: Converting these words into numbers (Vectors).

1D CNN (The Brain): A Neural Network scans these numbers to find "motifs" (patterns) associated with resistance.

Output: Classification (Resistant or Susceptible).

graph LR
A[Raw DNA Sequence] --> B[K-mer Tokenization]
B --> C[Integer Encoding]
C --> D[Embedding Layer]
D --> E[1D CNN Feature Extractor]
E --> F[Probability Score]
F --> G((Prediction: Resistant/Susceptible))


📂 Dataset & Preprocessing

1. The Source

Positive Class (Resistant): Sourced from the MEGARes v3.00 Database.

Negative Class (Susceptible): Since public databases mostly contain resistant genes, I generated a Synthetic Background Class.

2. Solving the "Class Imbalance" Trap

Initially, the dataset had many resistant genes and very few susceptible ones, causing the model to get stuck at 67% accuracy (guessing "Resistant" every time).

Solution: I generated synthetic DNA sequences based on natural nucleotide distribution to match the count of real resistant genes exactly.

Result: A perfectly balanced dataset (1:1 Ratio).

Resistant Samples: 5087

Susceptible Samples: 5087

3. K-mer Tokenization Strategy

Instead of reading one letter at a time, we used 6-mers (groups of 6 nucleotides). This captures the local context of the DNA, similar to how we read words rather than individual letters to understand a sentence.

🛠️ Model Architecture

The model is built using PyTorch with the following layers:

Embedding Layer: Converts integer-encoded DNA words into dense vectors.

Conv1d Layers (x3): Three layers of 1D convolutions with increasing filter sizes (64 -> 128 -> 256) to capture simple-to-complex genetic patterns.

Adaptive Avg Pooling: To handle variable-length DNA sequences.

Dropout: To prevent overfitting.

Fully Connected Layers: To make the final decision.

💻 Installation & Usage

1. Prerequisites

pip install biopython scikit-learn torch pandas numpy matplotlib seaborn


2. File Structure

megares_drugs_database_v3.00.fasta: Input FASTA file.

megares_drugs_annotations_v3.00.csv: Annotations file.

train.py: Main script for training the model.

3. Running the Training

# The script automatically detects GPU (CUDA) if available
python train.py


📊 Visuals

(You can add the screenshots of the ROC Curve and Confusion Matrix generated in the notebook here)

Confusion Matrix Explained

True Positives: Correctly identified Resistant genes.

True Negatives: Correctly identified Susceptible sequences.

False Positives: Normal DNA wrongly flagged as dangerous (Low is better).

False Negatives: Dangerous DNA missed by the model (Critical to keep this low).

📜 License

This project is open-source. Feel free to use it for educational purposes.

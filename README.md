
# ML Implementation from Scratch

In this project, I have implemented 2 models from scratch, without using Sklearn, Tensorflow, Pytorch or anyother prebuilt ML models. The two models are:
  1) K-Means Clustering
  2) Decoder Only Transformer


## Project Structure
Along with the codes of the models, I have also provided the IRIS dataset on which I trained the K-Means Clustering model and the text file on which I trained my Transformer model.
```
├── K_Means_Clustering.ipynb
├── IRIS.xls
├── Decoder1.ipynb
├── Decoder2.ipynb
├── Story.txt 
└── README.md
```
## Approach of the 2 Models

### K-Means Clustering
- Implemented **K-Means Clustering** from scratch (without using scikit-learn).
- Applied the algorithm on the **Iris dataset** for clustering.
- Developed **accuracy score** and **confusion matrix** functions from scratch.
- Avoided use of `sklearn.metrics` for evaluation — all metrics implemented manually.

### Decoder-Only Transformer
- Uploaded **two notebooks** implementing Decoder-Only Transformer:
  1. **Lightweight Model**
     - Does not use **AdamW optimizer**.
     - Could be trained for **500 epochs**.
     - Produced **better quality output** due to longer training.
  2. **Heavier Model**
     - Includes **AdamW optimizer**.
     - Could only be trained for **100 epochs** due to computational limits.
     - Output quality lower compared to lightweight version.
- Defined multiple **custom classes** from scratch for:
  - Tokenization
  - Embedding
  - Positional Encoding
  - Multi-Head Attention
  - Feed-Forward Network
  - Training loop
## Challenges Faced

- Could not use PyTorch/TensorFlow — all components implemented from scratch in NumPy.  
- NumPy was slow for large matrix operations and optimizer steps.  
- Limited computational resources (T4 GPU in Colab & laptop GPU) allowed only a few hundred iterations instead of ~500 needed for better learning.  
- As a result, the Decoder-Only Transformer is slightly undertrained.

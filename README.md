# MNIST
# 🔍 FFN_GeGLU vs FFN_ReLU on MNIST

## 📌 Description

This experiment evaluates the performance of two Feed-Forward Network (FFN) variants on the MNIST digit classification task:

- **FFN_ReLU**: A standard FFN using ReLU activation.
- **FFN_GeGLU**: A gated linear unit FFN using the GELU activation (inspired by [GLU paper](https://arxiv.org/pdf/2002.05202)).

The objective is to test the claim that **FFN_GeGLU outperforms FFN_ReLU** in terms of test accuracy under the same training conditions.

The models are implemented in **PyTorch** using `einsum` notation for matrix operations and trained using **PyTorch Lightning**. The experiment also performs:
- **Random hyperparameter search** across learning rate and batch size.
- Evaluation at hidden dimensions: `2, 4, 8, 16`.
- Trials with `k = 2, 4, 8` random hyperparameter combinations per hidden size.
- **Bootstrap-based error bars** on test accuracy.
- Visualization of accuracy vs hidden dimension.

---

## ⚙️ Setup

### Requirements

Make sure you have Python 3.8+ and install dependencies via:

```bash
pip install torch torchvision pytorch-lightning matplotlib numpy
###Running the Experiment
To run the experiment:

Clone the repository:

git clone https://github.com/your-username/ffn_relu_vs_geglu_mnist.git
cd ffn_relu_vs_geglu_mnist
Run the notebook or script to start training:

# If using Jupyter Notebook:
jupyter notebook MNIST_FFN_Comparison.ipynb

# Or if using a script:
python run_experiment.py
The code will:

Train both FFN_ReLU and FFN_GeGLU models.

Randomly search batch sizes [8, 64] and learning rates [1e-1, 1e-2, 1e-3, 1e-4].

Log validation and test accuracy.

Select the best trial per hidden dimension using validation accuracy.

Plot test accuracy (with error bars) vs hidden dim.

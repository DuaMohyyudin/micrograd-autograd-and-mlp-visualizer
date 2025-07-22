# micrograd.py – Enhanced Autograd Engine and MLP From Scratch in Python

> A clean, modular reimplementation of [Karpathy’s micrograd](https://github.com/karpathy/micrograd), with extended features including multiple activation functions, optimizer classes, mini-batch training, and a matplotlib-based computation graph visualizer.

---

## ✨ What's New in This Version

This enhanced version of `micrograd.py` includes:

- ✅ **Activation Functions**: `tanh`, `ReLU`, `sigmoid`, `exp`
- ✅ **Optimizers**: Class-based `SGD` and `Adam`
- ✅ **Loss Functions**: MSE and Binary Cross Entropy
- ✅ **Mini-Batch Support**: `DataLoader` class with shuffle
- ✅ **Visualization**: Pure Python computation graph using `matplotlib`
- ✅ **Extensible**: Clean OOP-based design for easy experimentation

---

## 📌 What’s New Compared to micrograd?

| Feature                 | Karpathy’s micrograd | This Project (Micrograd++)       |
|------------------------|----------------------|----------------------------------|
| Activation functions    | tanh only            | tanh, ReLU, sigmoid, exp         |
| Visual graph rendering  | via Graphviz         | via matplotlib (pure Python)     |
| Optimizers              | Manual SGD           | SGD + Adam class-based           |
| Loss functions          | Manual MSE           | MSE + CrossEntropy ready         |
| Batching                | None                 | Mini-batch with shuffle support  |
| Dataset support         | Toy manually entered | Ready for batching abstraction   |


🧠 Learning Focus
This project is ideal for learners and researchers who want to:

Understand autograd systems and backpropagation

Visualize computation graphs in forward/backward mode

Build and train neural nets without any external ML libraries

Compare optimizers and activation functions in isolation



# micrograd.py – Enhanced Autograd Engine and MLP From Scratch in Python

> A clean, modular reimplementation of [Karpathy’s micrograd](https://github.com/karpathy/micrograd), with extended features including multiple activation functions, optimizer classes, mini-batch training, and a matplotlib-based computation graph visualizer.

---

This code implements a minimal deep learning framework with:

A custom automatic differentiation engine (like a tiny version of PyTorch or TensorFlow).

A simple neural network architecture (MLP: Multi-Layer Perceptron).

A training loop to teach it how to fit small input-output patterns.

A custom visualizer using matplotlib to show how computations flow backward through the graph.

1. Core Concept: The Value class
This is the heart of the code. Each number used in calculations is wrapped in a Value object. This object:

Stores the actual numeric value.

Remembers how it was created (e.g., via addition, multiplication, etc.).

Knows its parents—the inputs that were used to calculate it.

Tracks its gradient (i.e., how much a small change in this value affects the final output).

Stores a tiny function (_backward) that can compute how its gradient should be passed backward to its inputs.

This is how automatic differentiation is implemented.

2. Math Support: Arithmetic and Functions
The Value class is smart. It knows how to behave like a number in:

Addition and subtraction

Multiplication and division

Power operations

Unary negation (-x)

Common functions like tanh and exp

Every time you perform an operation, a new Value object is created, and a link is stored back to the inputs. This builds a computation graph.

3. Backpropagation
Once you compute some final output (like a loss), you can call .backward() on that result. This:

Triggers a reverse pass through the graph.

Each node runs its _backward function.

This flows the gradient values backward from the output to the inputs using the chain rule from calculus.

This allows the model to know how to update its weights to reduce error.

4. Graph Drawing
There's a function that takes the final Value result and:

Walks through the computation graph recursively.

Assigns x-y positions to each node based on depth and order.

Draws boxes with values and gradients.

Draws arrows for dependencies.

Shows operations in red (+, *, etc.)

This helps visualize how data and gradients flow during training.

5. Neural Network Building Blocks
The neural network is built using three classes:

🧠 Neuron:
Each neuron has a set of weights and a bias.

It takes a list of input values, multiplies each by its weight, adds them all up, adds the bias, and then applies the tanh activation function.

Outputs a single Value.

🧱 Layer:
A layer is just a list of neurons.

It takes an input and passes it to every neuron, collecting the outputs.

🔗 MLP (Multi-Layer Perceptron):
This is a list of layers chained together.

Each layer feeds its output to the next.

This is your complete model.

6. Test Case 1: Basic Expression Test
This test creates a small chain of operations (like a * b + c) and computes a final loss. It:

Labels all values (a, b, c, etc.) for easier visualization.

Calls .backward() to get gradients.

Draws the graph to show what the computation looks like.

This is a minimal example of how the engine works.

7. Test Case 2: Neural Network Training
This test builds a 3-layer neural network to classify a few simple examples. Here's what it does:

Defines 4 small input vectors and their target outputs (some are +1, others -1).

Initializes an MLP with architecture: 3 input → 4 neurons → 4 neurons → 1 output.

Makes predictions and prints them before training.

Runs 20 training steps:

Predicts outputs

Calculates the squared error between predictions and actual labels

Clears gradients

Runs backpropagation

Updates parameters using simple gradient descent (p.data -= learning_rate * p.grad)

Prints loss after each step and final predictions.

This shows how even a basic custom-made neural network can learn from data if gradients and updates are correctly handled.

8. Unnecessary Warning
The script also checks if graphviz is installed and warns you it's not needed (since this version uses only matplotlib for drawing).

✅ Summary
You're building:

A miniature deep learning engine

A clean visualizer for computation graphs

A simple but working neural network from scratch

A training pipeline that uses backpropagation

And you’re doing it all without using any machine learning library — which is amazing for learning and understanding how deep learning actually works under the hood.

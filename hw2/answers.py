r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**1 A**

The input X tesor is (64,1024). The output tensor Y is (64,512).
Therefore, the jacobian of Y with respect to X will be (64, 512, 64, 1024).

**1 B**

The Jacobian is sparse. Each output batch only depends on the inputs of that batch. So for every index (b1, i, b2, j) such that 0 $\leq$ i $\leq$ 512 and 0 $\leq$ j $\leq$ 1024 and such that b1 $\neq$ b2, the jacobian is zero.


**1 C**

No need to explicitly compute it. In a linear layer, the Jacobian is simple:  
$$
\frac{\partial \mathbf{Y}}{\partial \mathbf{X}} = \mathbf{W}.
$$  
Using the chain rule, we have:  
$$
\frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{Y}} \times \mathbf{W}^\top.
$$  
This is computed as a matrix multiplication. There's no need to construct the full 4D Jacobian, as it would be computationally expensive and unnecessary.

---

**2 A:**  
Including the batch size and output dimensions, the Jacobian is a 4D tensor with the following shape:  
$$
(out\_features,\; in\_features,\; N,\; out\_features) = (512,\; 1024,\; 64,\; 512).
$$  
The exact arrangement of the axes may vary depending on the context, but fundamentally, each element of $\mathbf{Y}$ (for every batch) depends on every element of $\mathbf{W}$.

**2 B:**  
This Jacobian is not sparse. Every output neuron can potentially depend on all input features, resulting in a dense structure.

**2 C:**  
Again, no need to explicitly materialize it. Using the standard gradient formula for a linear layer (in batch form):  
$$
\frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^\top \times \frac{\partial L}{\partial \mathbf{Y}}.
$$  
Modern frameworks leverage efficient matrix operations to compute this without ever forming the complete 4D Jacobian.

"""

part1_q2 = r"""

No, backpropagation is not strictly required to train neural networks with gradient-based optimization. Finite differences, Direct gradient computation, and forward-mode differentiation are examples of alternative ways to optimize which are generally less effective.

Backpropogation is generally much more efficient than other methods due to its ease of calculation and so expecially when dealing with larger models, backpropogation is the prefered method.
"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.05
    reg = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.04
    lr_vanilla = 0.03
    reg = 0.001
    lr_momentum = 0.005
    lr_rmsprop = 0.00035
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = torch.nn.CrossEntropyLoss()
    lr, weight_decay, momentum = 0.01, 0.0001, 0.85  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.

    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**
**1. Number of Parameters**
**Regular Block:**

First 3×3 conv: $3 \times 3 \times 256 \times 256 + 256 \text{ (bias)} = 590,080$

Second 3×3 conv: $3 \times 3 \times 256 \times 256 + 256 \text{ (bias)} = 590,080$

Total: $1,180,160$ parameters

**Bottleneck Block:**

1×1 conv (256→64): $1 \times 1 \times 256 \times 64 + 64 \text{ (bias)} = 16,448$

3×3 conv (64→64): $3 \times 3 \times 64 \times 64 + 64 \text{ (bias)} = 36,928$

1×1 conv (64→256): $1 \times 1 \times 64 \times 256 + 256 \text{ (bias)} = 16,640$

Total: $70,016$ parameters

The bottleneck block uses dramatically fewer parameters (about 6% of the regular block).

**2. Floating Point Operations (FLOPs)**

**Regular Block:**

First 3×3 conv: $3 \times 3 \times 256 \times 256 \times H \times W = 589,824 \times H \times W$ operations

Second 3×3 conv: $3 \times 3 \times 256 \times 256 \times H \times W = 589,824 \times H \times W$ operations

Total: $1,179,648 \times H \times W$ operations

**Bottleneck Block:**

First 1×1 conv: $1 \times 1 \times 256 \times 64 \times H \times W = 16,384 \times H \times W$ operations

Middle 3×3 conv: $3 \times 3 \times 64 \times 64 \times H \times W = 36,864 \times H \times W$ operations

Final 1×1 conv: $1 \times 1 \times 64 \times 256 \times H \times W = 16,384 \times H \times W$ operations

Total: $69,632 \times H \times W$ operations

The bottleneck block requires significantly fewer computations (about 6% of the regular block).

**3. Ability to Combine Input**

**Spatial Combination (Within Feature Maps):**

Regular Block: Two 3×3 convolutions create a 5×5 receptive field, allowing for larger spatial context within each feature map

Bottleneck Block: Only one 3×3 convolution, resulting in a 3×3 receptive field

The regular block has stronger spatial combination capability

**Feature Map Combination (Across Feature Maps):**

**Regular Block:** Each layer operates directly on 256 channels, providing strong feature map mixing

**Bottleneck Block:**

First 1×1 reduces to 64 channels, combining information across all 256 input channels

Middle 3×3 processes the compact representation

Final 1×1 expands back to 256 channels, mixing information from the reduced feature maps

Both architectures have similar capacity for cross-channel feature combination



The bottleneck design trades some spatial combination capacity for computational efficiency while maintaining feature combination ability. The dramatic reduction in parameters and computations (about 94% fewer) makes it possible to build much deeper networks without excessive computational cost.
"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Part 1.** 

From the results, the depth of the model, controlled by the number of layers (L), has a significant effect on accuracy:

Shallow Models (L=2): Perform reasonably well initially but tend to underfit the data compared to deeper models, especially as the complexity of the task increases. This is evident in the lower test accuracy over epochs.
Moderate Depth (L=4): Strikes a balance and often achieves the best accuracy across training and testing. This depth seems to provide sufficient capacity to learn complex features without overfitting.
Deeper Models (L=8): While they can achieve high training accuracy, their test accuracy may degrade, suggesting overfitting. Additionally, the loss values exhibit high variance and instability, especially during training.
Best Depth: Based on the graphs, L=4 provides the best trade-off between training and test accuracy. This depth likely balances the ability to learn intricate patterns while avoiding overfitting or becoming too complex for effective training.

---

**Part 2:**

Untrainable Networks (L=16): The results show that for very deep networks (L=16), the model struggles to train, as indicated by stagnant accuracy and loss values.
Causes:

Vanishing/Exploding Gradients: Deeper networks are prone to gradient issues during backpropagation, making it hard for the model to update weights effectively.
Overparameterization: Excessive parameters can lead to inefficient training and difficulty in optimizing the loss function.
Small Feature Maps: If pool_every is not set correctly, deeper layers may encounter zero-width or near-zero-width feature maps, leading to issues in the forward pass.

Recommendations to Resolve Trainability Issues:

1: Gradient Stabilization:

Use batch normalization or layer normalization to normalize inputs to each layer, mitigating the vanishing/exploding gradient problem.
Adopt residual connections (ResNet architecture) to enable better gradient flow in deep networks.

2: Adjust Hyperparameters:

Use smaller learning rates with adaptive optimizers (e.g., Adam or RMSprop) to stabilize training for deep networks.
Reevaluate pool_every to ensure feature maps do not shrink too early or become too small.

"""

part5_q2 = r"""
**Observations**

**Effect of K on L=2 Configurations**
1. **Higher Filters Improve Accuracy**:  
   As `K` increases, there is a noticeable improvement in test accuracy.  
   This suggests that adding more filters enables the shallow network to capture richer features.

2. **Overfitting in High K (L2_K128)**:  
   The configuration with `K=128` overfits quickly.  
   Evidence includes rapid convergence and limited improvement on the test set.

3. **Difficulty Controlling Training Dynamics**:  
   Early stopping was necessary to prevent overfitting.  
   Attempts to use adaptive learning rates were unsuccessful in stabilizing training.

**Effect of K on L=4 Configurations**
1. **Balanced Depth and Filters Work Best**:  
   The `L4_K64` configuration provided the best performance.  
   Achieved a good balance between capacity and generalization.

2. **Overfitting in High K (L4_K128)**:  
   Similar to `L=2`, the higher filter configuration showed signs of overfitting.  
   However, its overall performance was more stable compared to `L2_K128`.

**Effect of K on L=8 Configurations**
1. **Reduced Performance Compared to Shallower Networks**:  
   The deeper `L=8` configurations consistently underperformed compared to `L=4` and `L=2`.  
   Likely due to the vanishing gradient problem and difficulty in optimizing deep networks without additional techniques (e.g., residual connections).

2. **High Variance in Training Loss**:  
   The loss curves for `L8` configurations, especially `L8_K128`, exhibited significant variance.  
   Indicates instability in training.

---

**Comparison to Experiment 1.1**

1. **Consistency in Optimal Depth**:  
   Similar to Experiment 1.1, the `L=4` configuration outperformed both shallower (`L=2`) and deeper (`L=8`) networks.

2. **Performance of L=8 Remains Poor**:  
   Issues with trainability and performance of the `L=8` configurations were evident in both experiments.  
   Reinforces the need for architectural adjustments (e.g., residual connections) to make deeper networks more effective.

3. **Filters (K) Complement Depth**:  
   Experiment 1.2 highlights that increasing `K` can improve performance in shallow and moderately deep networks.  
   However, the benefits diminish in deeper networks due to optimization challenges.






"""

part5_q3 = r"""

**Analysis of part 3:**
    **Best Configuration:** The `L=2` configuration with `K=[64, 128]` achieved the highest test accuracy (~65.54%), outperforming deeper networks.

    **Moderate Depths:** The `L=3` configuration had slightly lower accuracy (~64.01%), showing diminishing returns from additional depth.

    **Overfitting in Deep Networks:** The `L=4` configuration exhibited overfitting, with accuracy peaking (~64.25%) before declining.

    **Effect of High Filters:** Higher filters (`K=128`) accelerated overfitting, especially in deeper networks.

**Insights**
    **Generalization:** Shallower networks (`L=2`) generalize better and are more stable during training.

    **Diminishing Returns:** Deeper networks face diminishing returns due to overfitting and gradient issues.
    
    **Filter Counts:** While higher filters improve capacity, they also increase overfitting risks.

"""

part5_q4 = r"""
**Analysis of part 4:**



**Impact of Residual Connections:**
    Residual connections improved training stability in deeper networks (`L=16`, `L=32`) for `K=[32]`, resolving vanishing gradients observed in Experiment 1.1.
    Test accuracy improved for moderate depths (`L=8`) but plateaued or slightly decreased for very deep networks (`L=16`, `L=32`), indicating diminishing returns.

**Performance with High Filters (`K=[64, 128, 256]`):**
    Shallow networks (`L=2`) achieved the highest test accuracy, consistent with Experiment 1.3.
    Moderate depths (`L=4`) performed well but showed overfitting as depth increased.
    Deeper configurations (`L=8`) were more stable with residuals but did not outperform shallower networks in test accuracy, likely due to overfitting.

**More Comparisons to Experiments 1.1 and 1.3:**
    Residual connections resolved instability in deeper networks that we had in 1.3, enabling effective training up to `L=32`.
    Shallower networks (`L=2`, `L=4`) still generalized better, with limited benefits from additional depth even with residuals.
    High filter counts (`K`) accelerated overfitting, though residual connections mitigated this somewhat.

**Insights**
    Effectiveness of Residual Connections: Essential for training deep networks, preventing vanishing gradients and improving stability.
    Optimal Depth: Moderate depths (`L=4`, `L=8`) balance capacity and generalization, while very deep networks (`L=16`, `L=32`) face diminishing returns.
    Filter Counts and Overfitting: Higher filters enhance representation capacity but increase overfitting risks, which residual connections help to mitigate but not eliminate.

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
 Weekly Summary — Deep Learning 

Tasks completed this week:

1) Tensors & NCHW basics
- Verified image/batch tensor shapes and visualized a random 28×28 grayscale tensor.
- Seed used: 1337.
- Figure: `tensors_example.png`

2) Feedforward Neural Network (FFNN) on MNIST (≥ 3 epochs)
- Trained for 3 epochs with SGD (lr=0.05).
- Results:
  - Epoch 1: loss 0.7789, test acc 0.8947
  - Epoch 2: loss 0.3532, test acc 0.9126
  - Epoch 3: loss 0.3045, test acc 0.9215
- Figures: `ffnn_loss.png`, `ffnn_acc.png`

3) Backprop Check (Finite-Difference Gradient Check)
- Implemented numerical gradient approximation and compared to autograd.
- Relative error :1.150553e-03 (≤ 1e−2 is good).
- Figure:`gradcheck_bar.png` (first 20 params: analytical vs numerical)

4) CNN Baseline + First-Layer Filter Visualization
- CNN: Conv(1→8, k=5, p=2) → MaxPool(2) → Conv(8→16, k=3, p=1) → FC(64) → FC(10).
- Trained for 3 epochs with SGD (lr=0.01).
- Results :
  - Epoch 1: loss 1.5134, test acc 0.8448
  - Epoch 2: loss 0.3986, test acc 0.9067
  - Epoch 3: loss 0.3105, test acc 0.9192
- Figures: `cnn_loss.png`, `cnn_acc.png`, `cnn_filters_conv1.png`

5) Learning-Rate Sweep (Convergence & Stability)
- Trained FFNN (3 epochs) with learning rates: 0.005, 0.05, 0.5.
- Results:
  - lr=0.005 → final loss 0.8747, acc 0.8501 (slow but stable)
  - lr=0.05 → final loss 0.3029, acc 0.9232 (good balance)
  - lr=0.5 → final loss 0.0940, acc 0.9636 (fast convergence; may risk overshoot in longer runs)
- Figures: `sweep_loss_compare.png`, plus per-lr loss/acc plots  
  (`sweep_lr_0_005_loss.png`, `sweep_lr_0_005_acc.png`, `sweep_lr_0_05_loss.png`, `sweep_lr_0_05_acc.png`, `sweep_lr_0_5_loss.png`, `sweep_lr_0_5_acc.png`).

6) Convolution Arithmetic (Formula vs PyTorch)
- Verified output sizes for several (kernel, stride, padding, dilation) configs.
- All cases matched PyTorch, e.g. (28×28, k=5, s=1, p=2, d=1) → (28×28).
- Figure: `conv_arith_checks.png`


Interpretation & notes
- FFNN vs CNN: After just 3 epochs, both reached ~0.92 test accuracy; with more epochs and/or data augmentation the CNN typically surpasses FFNN.
- Gradient check: Relative error ~1e−3 confirms the backprop implementation is consistent with finite differences.
- Learning rate: In short runs, higher lr (0.5) converged faster. For longer runs you may prefer 0.05 to avoid potential instability.
- Conv arithmetic: The closed-form output-size equation matched library behavior across tested cases.



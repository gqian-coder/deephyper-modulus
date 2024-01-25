# Fourier Neural Operater for Darcy Flow

This example demonstrates how to set up a data-driven model for a 2D Darcy flow using
the Fourier Neural Operator (FNO) architecture inside of Modulus.
Training progress can be tracked through [MLFlow](https://mlflow.org/docs/latest/index.html).
This example runs on a single GPU, go to the
`darcy_nested_fno` example for exploring a multi-GPU training.

## Getting Started

To train the model, run

```bash
python train_fno_darcy.py
```

training data will be generated on the fly.

Progress can be monitored using MLFlow. Open a new terminal and navigate to the training
directory, then run:

To train the baseline model, run
```bash
python Trainer.py 
```

## Additional Information

## References

- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)

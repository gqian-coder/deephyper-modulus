# Fourier Neural Operater for Darcy Flow

This example demonstrates how to set up a data-driven model for a 2D Darcy flow using
the Fourier Neural Operator (FNO) architecture inside of Modulus.
Training progress can be tracked through [MLFlow](https://mlflow.org/docs/latest/index.html).
This example runs on a single GPU, go to the
`darcy_nested_fno` example for exploring a multi-GPU training.

## Getting Started

To train the baseline model, run
```bash
python Trainer.py 
```

To run deephyper with fno , run
```bash
python train_fno_darcy_dhp.py 
```

## Additional Information

## References

- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)

# esim\_torch

This package exposes python bindings for ESIM with GPU support. 
Test your installation with 

```bash
cd esim_torch/
python test.py
```

which should create a plot. 

The currently supported functions are listed in the example below:
```python
import esim_torch

# constructor
esim = esim_torch.EventSimulator_torch(
    contrast_threshold_pos,  # contrast threshold, currently only constant thresholds are supported 
)

events = esim.forward(
    images,        # torch tensor with type float32, shape T x H x W
    timestamps_ns  # torch tensor with type int64,   shape T 
)

```

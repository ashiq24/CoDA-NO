# CoDA-NO

## How to Run CoDA-NO

Download our repo, install our `requirements.txt`, and run the following script:

```
python main.py --config <CONFIG_NAME> --ntrain N
```

where `N` is the number of training examples to use (all provided datasets
contain 1000 data points) and `<CONFIG_NAME>` is a valid config name as
described below.

### Configurations

There are several configs described in `config/ssl_ns_elastic.yaml`. They are
semantically named like `MODEL-NAME_DATASET-NAME_VISCOSITY` where:
* `VISCOCITY` may be `10`, `5`, `1`, or `.5` (i.e. 0.5).
* `DATASET-NAME` may be:
  * `NS`, where the governing equations are just
    Navier-Stokes, or
  * `NSES`, where the governing equations are both
    Navier-Stokes and the Elastic Wave equation.
  * In the case of fine-tuning configs (prefixed with `ft` as their model name)
    there will be 2 datasets in the middle part (ex. `NS_NSES`). The latter name
    is the dataset that will actually be fine-tuned on, while the former name is
    the dataset that the model was pretrained on. The former tells the trainer
    which Variable Specific Positional Encoding to load for this model.
* `MODEL-NAME` may be one of the following:
  * `"codano"`: Our novel Co-Domain Attention Neural Operator architecture 
    (sandwiched between GINO blocks); this executes the pretraining phase (where
    the operator learns an encoding by reconstructing masked inputs).
  * `"ft"`: Our novel CoDA-NO architecture; this executes the **fine-tuning**
    phase where the operator learns to predict the next time step.  
  * `"fno"`: Fourier Neural Operator (FNO)
  * `"gnn"`
  * `"deeponet"`: DeepONet
  * `"vit"`: Vision Transformer
  * `"unet"`: U-shaped Network

Before running `main.py`, you will need to:
* update the WandB credentials if you intend to log to WandB
  (this is turned off by default).
* update the "input_mesh_location" and "data_location" in the config file.

### NS+Elastic Dataset
**Dataset**: https://zenodo.org/records/10603460
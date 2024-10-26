# Pretraining  Codomain Attention Neural Operators for Solving Multiphysics PDEs
<p align="center">
    <img src="https://github.com/ashiq24/CoDA-NO/blob/web_resources/images/banner.png" alt="">
    <br>
</p>
**Coda-NO** is designed to adapt seamlessly to new multi-physics systems. Pre-trained on fluid dynamics data from the Navier-Stokes equations, which include variables $$u_x$$, $u_y$, and $p$, CoDA-NO can easily transition to multi-physics fluid-solid interaction systems that incorporate new variables \(d_x\) and \(d_y\), all without requiring any architectural changes.</em>
**Abstract**: Existing neural operator architectures face
challenges when solving multiphysics problems with coupled partial differential equations (PDEs), due to complex geometries, interactions between physical variables, and the lack of large amounts of high-resolution training data. To address these issues, we propose Codomain Attention Neural Operator (CoDA-NO), which tokenizes functions along the codomain or channel space, enabling self-supervised learning or pretraining of multiple PDE systems. Specifically, we extend positional encoding, self-attention, and normalization layers to the function space. CoDA-NO can learn representations of different PDE systems with a single model. We evaluate CoDA-NO's potential as a backbone for learning multiphysics PDEs over multiple systems by considering few-shot learning settings. On complex downstream tasks with limited data, such as fluid flow simulations and fluid-structure interactions, we found CoDA-NO to outperform existing methods on the few-shot learning task by over $36$%. [Paper Link](https://arxiv.org/pdf/2403.12553.pdf)

## Model Architecture
<p align="center">
    <img src="https://github.com/ashiq24/CoDA-NO/blob/web_resources/images/pipeline.png" alt="">
    <br>
    <em> <strong>Left:</strong> Architecture of the Codomain Attention Neural Operator. <strong>Right:</strong> Mechanism of codomain attention.</em>
</p>
Each physical variable (or co-domain) of the input function is concatenated with variable-specific positional encoding (VSPE). Each variable, along with the VSPE, is passed
through a GNO layer, which maps from the given non-uniform geometry to a latent regular grid. Then, the output on a uniform grid
is passed through a series of CoDA-NO layers. Lastly, the output of the stacked CoDA-NO layers is mapped onto the domain of the
output geometry for each query point using another GNO layer.

At each CoDA-NO layer, the input function is tokenized codomain-wise to generate token functions. Each token function is passed through the K, Q, and V operators to
get key, query, and value functions. The output function is calculated via an extension of the self-attention mechanism to the function space.

## Navier Stokes+Elastic Wave and Navier Stokes Dataset
The fluid-solid interaction dataset is available at [**Dataset link**](https://drive.google.com/drive/u/0/folders/1dN5de1n0qVYLEWf6JwXjqbCNUXl4Z8Tj).

## Experiments

The configurations for all the experiments are at `config/ssl_ns_elastic.yaml`.

To set up the environments and install the dependencies, please run the following command:
```
bash installation.sh
```
To run the experiments, download the datasets, update the "input_mesh_location" and "data_location" in the config file,  update the wandb cradentials and execute the following command

```
python main.py --config "config name" --ntrain N
```

`--config`: which configuration to use from the config file 'config/ssl_ns_elastic.yaml`.

`--ntrain`: Number of training data points.

## Scripts
For training CoDA-NO architecture on NS and NS+EW datasets (both pre-training and fine-tuning) please execute the following scrips:
```
codano_ns.sh
codano_nses.sh
```
For training the baseline, execute the following scripts
```
fno_baseline.sh 

gnn_baseline.sh

deeponet_baseline.sh

unet_baseline.sh

vit_baseline.sh
```

## Reference
If you find this paper and code useful in your research, please consider citing:
```bibtex
@article{rahman2024pretraining,
  title={Pretraining Codomain Attention Neural Operators for Solving Multiphysics PDEs},
  author={Rahman, Md Ashiqur and George, Robert Joseph and Elleithy, Mogab and Leibovici, Daniel and Li, Zongyi and Bonev, Boris and White, Colin and Berner, Julius and Yeh, Raymond A and Kossaifi, Jean and Azizzadenesheli, Kamyar and Anandkumar, Anima},
  journal={arXiv preprint arXiv:2403.12553},
  year={2024}
}
```

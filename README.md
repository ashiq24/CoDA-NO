# CoDA-NO

## NS+Elastic Dataset
**dataset link** :https://drive.google.com/drive/u/0/folders/1dN5de1n0qVYLEWf6JwXjqbCNUXl4Z8Tj

config file: config/ssl_ns_elastic.yaml

Update the wandb cradentials.

Update the "input_mesh_location" and "data_location" in the config file and run the following command.


```
python main.py --config "config name" --ntrain N
```

"--config" -> whihc configuration to use from the config file "ssl_ns_elastic.yaml"

"--ntrain" -> Number of training data points.

## Scripts
codano.sh -> For training CoDA-NO architecture.

**fno_baseline.sh and gnn_baseline.sh** are for baselines. Baselines will be trained on different number of training examples.

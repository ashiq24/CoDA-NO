## General Response
We thank the reviewers for reviewing and appreciating our work. We're going to address the common concern here, and in each reviewer's thread, we answer individual questions.

We agree with the reviewer that fluid dynamics is easier to model when the viscosity gets higher. However, the more relevant measure of the complexity of the fluid flow is the Reynolds number, which depends on the viscosity, velocity, and density of the fluid. For our setup, the fluid considered is water, with a density of 1000 kg.m-3 and a maximum inlet velocity of approximately $4 m.s^{-1}$, leading to Reynolds numbers in the range $200-2000$ for our experiments. Only when the flow becomes turbulent ample movements of the elastic strap (Figure 4) can be observed in the fluid-structure interaction case. Modeling fluid-solid interaction or only fluid motion with such a Reynolds number is quite challenging. Fluid-structure interactions at even higher Reynolds numbers could be performed with the TurtleFSI solver used in this paper; however, they would require a very high computational cost. It's worth noting that TurtleFSI's fluid solver, including its fluid-structure interaction solver, uses a direct numerical simulation (DNS) of fluid dynamics and does not take into account any turbulence models. This means that in order to accurately capture the small-scale energy-dissipating vortices that form when the flow interacts with the cylinder and strap at high Reynolds numbers, a very fine spatial domain discretization is required. Furthermore, an extremely small time step is necessary to ensure numerical stability. For these reasons, the contribution [1], which introduced the benchmark fluid-structure interaction problem studied here, only deals with flows that have Reynolds numbers less than or equal to 200. Previous studies have also focused on fluid flow with a similar or lower Reynolds number when modeling fluid dynamics [2-3].

Following the suggestion of the reviewers, we present the result of the fluid-solid interactions PDE at viscosity $\mu = 0.5$. We can observe that our CoDA-No model can adapt to even lower viscosities even when it is pre-trained on higher viscosities ($\mu \in \{1, 10\}$).

### Table 1

| models | Pre-training Dataset | ntrain = 5             | ntrain=25             | ntrain=100             |
|--------|----------------------|------------------------|-----------------------|------------------------|
| GINO   |                      |                        |                       |                        |
| DeepO  |                      |                        |                       |                        |
| GNN    |                      |                        |                       |                        |
| ViT    |                      |                        |                       |                        |
| U-net  |                      |                        |                       |                        |
| Ours   |                      |                        |                       |                        |
| Ours   | NS                   |                        |                       |                        |
| Ours   | NS-EW                |                        |                       |                        |

[1]  Turek, Stefan, and Jaroslav Hron. "Proposal for numerical benchmarking of fluid-structure interaction between an elastic object and laminar incompressible flow."

[2]​​Tian, Yifeng, Michael Woodward, Mikhail Stepanov, Chris Fryer, Criston Hyett, Daniel Livescu, and Michael Chertkov. "Lagrangian large eddy simulations via physics-informed machine learning."

[3]Lucas, Dan, and Rich Kerswell. "Spatiotemporal dynamics in two-dimensional Kolmogorov flow over large domains."


## Reviewer 1yj2
We thank reviewer 1yj2 for reviewing and appreciating our work as interesting and well-organized. Now we address the concerns raised by reviewer 1yj2.

### Table 2 New metric - L1 and relative L2

Results on NS-EW dataset
| models | Pre-training Dataset | ntrain = 5 (L1/Rel-L2) | ntrain=25 (L1/Rel-L2) | ntrain=100 (L1/Rel-L2) |
|--------|----------------------|------------------------|-----------------------|------------------------|
| GINO   |                      | 0.1852/0.2967          | 0.151/0.2216          | 0.1608/0.219           |
| DeepO  |                      | 0.4534/0.687           | 0.2666/0.4312         | 0.1846/0.3254          |
| GNN    |                      | 0.08388/0.1309         | 0.05631/0.08211       | 0.05927/0.08265        |
| ViT    |                      | 0.2026/0.3663          | 0.1565/0.2762         | 0.07654/0.1242         |
| U-net  |                      | 0.7937/1.186           | 0.2845/0.4635         | 0.1746/0.291           |
| Ours   |                      | 0.09283/0.1649         | 0.0463/0.09232        | 0.0321/0.0583          |
| Ours   | NS                   | 0.07482/0.1412         | 0.03243/0.07264       | 0.03093/0.05936        |
| Ours   | NS-EW                | 0.06686/0.1281         | 0.04087/0.07727       | 0.03335/0.0578         |


Results on NS Dataset

| models | Pre-training Dataset | ntrain = 5 (L1/Rel-L2) | ntrain=25 (L1/Rel-L2) | ntrain=100 (L1/Rel-L2) |
|--------|----------------------|------------------------|-----------------------|------------------------|
| GINO   |                      | 0.2366/0.3652          | 0.1335/0.1991         | 0.1065/0.1555          |
| DeepO  |                      | 0.441/0.695            | 0.3958/0.5615         | 0.2353/0.3375          |
| GNN    |                      | 0.1415/0.1876          | 0.07486/0.09653       | 0.04947/0.07095        |
| ViT    |                      | 0.2791/0.4316          | 0.1582/0.2388         | 0.1377/0.188           |
| U-net  |                      | 2.001/ 3.508           | 0.6839/1.178          | 0.2983/0.4225          |
| Ours   |                      | 0.2464/0.3557          | 0.08393/0.1413        | 0.03373/0.07466        |
| Ours   | NS                   | 0.08053/0.1483         | 0.04025/0.08173       | 0.02447/0.06076        |
| Ours   | NS-EW                | 0.07556/0.1431         | 0.041/0.0692          | 0.02226/0.0579         |


Figure For Energy spectrum
![image](https://anonymous.4open.science/r/annonimous_support-0F53/energy_spectrum_plot.png)
### table 3 Error Bar
 
Results on NS-EW dataset

| models | Pre-training Dataset | ntrain = 5                         | ntrain=25                            | ntrain=100                         |
|--------|----------------------|------------------------------------|--------------------------------------|-------------------------------------|
| GINO   |                      | 0.121 $\pm$ 0.023                      |  0.0530 $\pm$ 0.0053                     | 0.0345$\pm$0.0086                       |
| DeepO  |                      |  0.534 $\pm$ 0.005                      | 0.192 $\pm$ 0.0072                      |  0.1384$\pm$0.0293                      |
| GNN    |                      |  0.121 $\pm$ 0.136                      | 0.0304 $\pm$ 0.02104                    | 0.0200$\pm$0.0120                       |
| ViT    |                      |  0.276 $\pm$ 0.093                      | 0.0837 $\pm$ 0.0284                     | 0.0208$\pm$0.0044                      |
| U-net  |                      |  1.002 $\pm$ 0.197                      |  0.8368 $\pm$ 0.3503                    | 0.5814$\pm$0.5680                      |
| Ours   |                      |  0.059 $\pm$ 0.017                      |  0.0096 $\pm$ 0.0010                    | 0.0038$\pm$0.0003                       |
| Ours   | NS                   |  0.068 $\pm$ 0.055                      | 0.0078 $\pm$ 0.0002                     |  0.0036$\pm$0.0005                        |
| Ours   | NS-EW                | 0.044 $\pm$ 0.041                       | 0.0057 $\pm$ 0.0012                     | 0.0034$\pm$0.0006                       |


Results on NS Dataset

| models | Pre-training Dataset | ntrain = 5             | ntrain=25             | ntrain=100             |
|--------|----------------------|------------------------|-----------------------|------------------------|
| GINO   |                      | 0.1436 $\pm$ 0.02315         | 0.03714 $\pm$ 0.0044        |  0.0330 $\pm$ 0.0105                      |
| DeepO  |                      | 1.621 $\pm$ 0.2417           | 0.3162 $\pm$ 0.1146         | 0.1978 $\pm$ 0.03452                       |
| GNN    |                      | 0.0113 $\pm$ 0.0046         | 0.00461 $\pm$ 0.00129       |  0.0051 $\pm$ 0.0024                      |
| ViT    |                      |  0.1961 $\pm$ 0.0326         | 0.04098 $\pm$ 0.0057        | 0.03028 $\pm$ 0.01606                       |
| U-net  |                      |  2.94 $\pm$ 1.989            |   0.6568 $\pm$ 0.3635       |  0.3955 $\pm$ 0.3987                      |
| Ours   |                      | 0.0612 $\pm$ 0.0364          |  0.00942 $\pm$ 0.0006       |  0.00453 $\pm$ 0.0006                      |
| Ours   | NS                   |  0.0276 $\pm$ 0.00321        | 0.00572 $\pm$ 0.0005        |  0.00390 $\pm$ 0.0001                      |
| Ours   | NS-EW                |  0.0273 $\pm$ 0.0054         | 0.005665 $\pm$ 0.0005       | 0.004023 $\pm$ 0.0001                      |


### Table 4 Ablation Study

| models             | Pre-training Dataset | ntrain = 5 (NS / NS-EW) | ntrain = 25 (NS / NS-EW) | ntrain = 100 (NS / NS-EW) |
|--------------------|----------------------|-------------------------|--------------------------|---------------------------|
| No CoDA-NO layer   | -                    | 0.271 / 0.211           | 0.061 / 0.113            | 0.017 / 0.020             |
| CoDA-NO            | -                    | 0.182 / 0.051           | 0.008 / 0.084            | 0.006 / 0.004             |
| CoDA-No (NO VSPE)  | NS                   | 0.049 / 0.079           | 0.009 / 0.0132           | 0.004 / 0.009             |
| CoDA-No (NO VSPE)  | NS EW                | 0.045 / 0.057           | 0.010 / 0.011            | 0.008 / 0.004             |
| CoDA-No (NO Norm.) | NS                   |  * / *                  | 0.023 / *                | 0.008 / 0.006             |
| CoDA-No (NO Norm.) | NS EW                | 0.057 / 0.232           | 0.012 / 0.052            | 0.006 / 0.006             |
| CoDA-No            | NS                   | 0.025 / 0.071           | 0.007 / 0.008            | 0.004 / 0.005             |
| CoDA-No            | NS EW                | 0.024 / 0.040           | 0.006 / 0.005            | 0.005 / 0.003             |

### Table 6 Inference Time

| models          | GNN   | GINO  | DeepO | ViT   | Unet  | CoDA-NO |
|-----------------|-------|-------|-------|-------|-------|---------|
| Inference Time  | 0.012 | 0.012 | 0.006 | 0.071 | 0.024 | 0.440   |
| Training Time   | 0.136 | 0.136 | 0.131 | 0.273 | 0.268 | 1.25    | 

Table 5 Super Resolution Table

| Model    |Pretraining Dataset   |  mu = 5             |   mu = 1| mu =10  |
|----------|----------------------|---------------------|---------|---------|
| u-net    |                      | 0.1401              | 0.2345  | 0.2252  |
| Vit      |                      | 0.05166             | 0.1844  | 0.04682 |  
| GINO     |                      | 0.102               | 0.1139  | 0.1071  |
| DeepO    |                      |  0.113              | 0.107   | 0.350   |
| GNN      |                      | 0.01381             | 0.1008  | 0.01799 |  
| CoDA-NO  | NS-EW                | 0.03569             | 0.06672 | 0.04389 |  
| CoDA-NO  | NS                   | 0.03409             | 0.05625 | 0.03749 |   



### Table 7 Parameters

Overfitting of basslines with higher parameters on NS-EW dataset.
| models  | # Parameter (Used/Big) x 1e6 | # train = 5  (Used / Big) | # train=25  (Used/ Big) | # train=100  (Used / Big) |
|---------|------------------------------|--------------------------|------------------------|--------------------------|
| GINO    | 60/200                       | 0.122 / 0.3423           | 0.053 / 0.06607        | 0.043 / 0.036            |
| DeepO   | 6 / 25                       | 0.482 /  0.4958          | 0.198 /  0.3039        | 0.107 / 0.083            |
| GNN     | 0.5/ 7                       | 0.045 / 0.2689           | 0.009 / 0.03174        | 0.009 / 0.061            |
| ViT     | 27/100                       | 0.211 / 0.2663           | 0.113 / 0.1255         | 0.020 / 0.022            |
| U-net   | 30/48                        | 3.579 / 9.462            | 0.842 / 3.957          | 0.203 / 0.412            |





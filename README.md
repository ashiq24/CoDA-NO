## General Response
We thank the reviewers for reviewing and appreciating our work. We're going to address the common concern here, and in each reviewer's thread, we answer individual questions.

We agree with the reviewer that fluid dynamics is easier to model when the viscosity increases. However, the more relevant measure of the complexity of the fluid flow is the Reynolds number, which depends on the fluid's viscosity, velocity, and density. 

For our setup, the fluid considered is water, with a density of 1000 kg.m-3 and a maximum inlet velocity of approximately $4 m.s^{-1}$, leading to Reynolds numbers in the range $200-2000$ for our experiments. Only when the flow becomes turbulent can ample movements of the elastic strap (Figure 4) be observed in the fluid-structure interaction case. Modeling fluid-solid interaction or only fluid motion with such a Reynolds number is quite challenging and used as a benchmark problem [2-3]. 

Modeling fluid-solid interaction with an even higher Reynolds number requires a very high computational cost. Because TurtleFSI's (used in this study) fluid solver, including its' fluid-structure interaction solver, uses a direct numerical simulation (DNS) of fluid dynamics and does not employ any turbulence models. This means that in order to accurately capture the small-scale energy-dissipating vortices that form when the flow interacts with the cylinder and strap at high Reynolds numbers, a very fine spatial domain discretization is required. Furthermore, an extremely small time step ($\Delta t$) is necessary to ensure numerical stability. For these reasons, the contribution [1], which introduced the benchmark fluid-structure interaction problem studied here, only deals with flows that have Reynolds numbers less than or equal to 200. 

In order to show the effectiveness of our proposed model, we pre-train the CoDA-NO model on PDEs with viscosity $\mu \in \{1, 10\}$. We finetune the pre-trained model with a different few shot training examples with viscosities $\mu \in \{1, 5, 10\}$ with $\mu = 5$ as unseen viscosity (Table 1-2, Figure 5, Supplementary Sec A4 Table 3-4). 

It's crucial to highlight a significant disparity between the pre-training and fine-tuning stages, particularly concerning examples with viscosities 1 and 10. This disparity arises from the utilization of distinct inlet boundary conditions during pre-training and fine-tuning phases (see Section 5, Experiment Setup and Ablation Studies). Consequently, even though the viscosities align with the pre-training dataset during fine-tuning on PDEs featuring $\mu \in {1, 10}$, the model faces formidable challenges in adapting due to variations in inlet conditions. The finetuning dataset with viscosity=5 has different viscosity as well as intel conditions compared to the pre-training dataset, serving as an out-of-distribution PDE setup.

Thus, our designed experiments serve as rigorous benchmarks, testing the model's adaptability across diverse PDEs.


Following the suggestion of the reviewers, we present the result of the fluid-solid interactions PDE at viscosity $\mu = 0.5$. We can observe that our CoDA-No model can adapt to even lower viscosities even when it is pre-trained on higher viscosities ($\mu \in \{1, 10\}$).

### Table 

| Models | Pre-training Dataset |  # Train = 5           | # Train=25            | # Train=100            |
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

We thank reviewer 1yj2 for reviewing and appreciating our work as interesting and well-organized. Now we will address the concerns raised.

> Concern regarding experiment design

Please see the general response.

> Question regarding different matrices (like L1, relative L2, energy spectra)

L2 and relative L2 are scaled versions of each other. L2 penalizes large errors more than L1  and is a common choice [1] as a metric across different tasks. We report the additional metrics (L1 and relative L2) here. We can see that our model performs better across different metrics.

The mesh used for the simulation is highly nonlinear, i.e., point density at some location (near the sphere) is very high compared to the rest of the domain. Calculating other metrics like divergence on such a nonuniform mesh is highly prone to numerical errors, which may lead to wrong evaluations of the model. For example, calculating divergence from ground-truth fluid flow does give 0.
Here we provide the L1 and relative L2 errors. We also provide energy spectrum for different models.


#### Table: L1 and relative L2 Error

Results on the fluid-solid interaction dataset combining Navier-Stokes and Elastic wave equation **(NS-EW dataset)**.

| Models | Pre-training Dataset | # Train = 5 (L1/Rel-L2)| # Train=25 (L1/Rel-L2)| # Train=100 (L1/Rel-L2)|
|--------|----------------------|------------------------|-----------------------|------------------------|
| GINO   |                      | 0.185/0.296          | 0.151/0.221          | 0.160/0.219           |
| DeepO  |                      | 0.453/0.687           | 0.266/0.431         | 0.184/0.325          |
| GNN    |                      | 0.083/0.130        | 0.056/0.082       | 0.059/0.082        |
| ViT    |                      | 0.202/0.366          | 0.156/0.276         | 0.076/0.124         |
| U-net  |                      | 0.793/1.186           | 0.284/0.463         | 0.174/0.291           |
| Ours   |                      | 0.092/0.164         | 0.046/0.092        | 0.032/0.058          |
| Ours   | NS                   | 0.074/0.141         | 0.032/0.072       | 0.030/0.059        |
| Ours   | NS-EW                | 0.066/0.128         | 0.040/0.077       | 0.033/0.057         |

Results on fluid motion dataset governed by Navier-Stokes equation **(NS Dataset)**.

| Models | Pre-training Dataset | # Train = 5 (L1/Rel-L2)| # Train=25 (L1/Rel-L2)| # Train=100 (L1/Rel-L2)|
|--------|----------------------|------------------------|-----------------------|------------------------|
| GINO   |                      | 0.236/0.365          | 0.133/0.199         | 0.106/0.155          |
| DeepO  |                      | 0.441/0.695            | 0.395/0.561         | 0.235/0.337          |
| GNN    |                      | 0.141/0.187          | 0.074/0.096       | 0.049/0.071        |
| ViT    |                      | 0.279/0.431          | 0.158/0.238         | 0.137/0.188           |
| U-net  |                      | 2.001/ 3.508           | 0.683/1.178          | 0.298/0.422          |
| Ours   |                      | 0.246/0.355          | 0.083/0.141        | 0.033/0.074        |
| Ours   | NS                   | 0.080/0.148         | 0.040/0.081       | 0.024/0.0607        |
| Ours   | NS-EW                | 0.075/0.143         | 0.041/0.069          | 0.022/0.057         |


###  Energy Spectrum
Here, we show the energy spectrum for the NS-EW dataset for $\mu = 5$ calculated from the test set. All models are trained on 100 training examples. Due to numerical error, the measured spectral energy does not decay smoothly in the high-frequency region. However, our models' energy spectrum remains closest to the ground truth. 

![image](https://anonymous.4open.science/r/annonimous_support-0F53/energy_spectrum_plot.png)

We plan to add these additional results to the manuscript. 


> Additional Implementation Details

We thank the reviewer for pointing it out. Following the maskedViT[2], we dropped ~75% of the points.

The model is set in an auto-regressive way, and the input sequence length is 1. We plan to rewrite the relevant section to highlight these details.

[1] Wang, Rui, et al. "Towards physics-informed deep learning for turbulent flow prediction." Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining."

[2] He, Kaiming, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. "Masked autoencoders are scalable vision learners."



## Reviewer R8HL
We appreciate the reviewer's positive feedback on our work. We are pleased that the reviewer finds the problem both interesting and challenging. We will now address the concerns here

> Concern Regarding Limiting experiments and a single viscosity level

We tested our model on three different viscosity levels $\mu \in 1,5,10$ with different inlet conditions from the pre-training dataset, along with varying numbers of interacting variables (single and multiphysics). Please take a look at Table 1-2, Figure 5, Supplementary Sec A4 Table 3-4 for the detailed results.

For an elaborate discussion about the experiment design - please take a look at the general response.

> Transfer to Real World Application

In this work, we propose CoDA-NO, the first neural operator architecture that, to the best of our knowledge, allows the adaptation of a single-physics pre-trained model to a coupled multi-physics pde. To justify the effectiveness of our approach, we provide experiments of adaptation from only fluid dynamics to coupled multi-physics fluid-solid interaction problems.

We agree with the perspective of the reviewer that it is very important to transfer from simulated single physics to real-world multi-physics. However, real-world multi-physics systems (e.g., weather data) are often noisy and incomplete, with incompatibility across different data sources (e.g., weather stations) and large scale. These challenges require a separate effort, and we plan to address these in future work where we extend the proposed CoDA-NO for simulation single physics to real-world multi-physics adaptation.

> Models Performance Compared to Baseline GNN and Justification for Using CoDA-NO

The proposed CoDA-NO uses a Graph neural Operator (GNO) as part of the backbone. Unlike GNNs, graph neural operators are resolution invariant [1,2]. With respect to the GNN baseline, our proposed model achieves a 36% better performance when modeling multi-physics PDE with unseen viscosity, which is a considerable margin. We should also note that the encoder module of CoDA-NO is pretrained once and used for different experiment setups. Only the predictor module is initialized from scratch for every experiment setup. On the other hand, each baseline is trained completely from scratch for every experiment.

We agree with the reviewer that the proposed CoDA-NO has additional complexity. However, to adapt seamlessly to new PDEs using the same encoder module, the model needs to be expressive enough and need to be pre-trained on enough data. However, this pre-training only needs to be done once, after which the model can generalize to a variety of PDEs. This makes such architectures well-suited as foundation models for scientific computing.

Also, to emphasize the resolution-invariance nature of CoDA-No, a neural operator [1,2], and GNN, we present an additional experiment on zero-shot super-resolution. Here, all models are trained on a mesh with 1317 points (low resolution) and a given viscosity. We use 100 few-shot training examples. During inference, the model is queried directly on a denser and non-uniform target mesh consisting of 2193 points (high-resolution mesh). We can observe that CoDA-NO achieves higher performance compared to other baselines.

####  Super Resolution Experiment on Fluid-Solid Interaction Dataset 
NS-EW = Fluid-solid interaction dataset combining Navier-Stokes and Elastic wave equation.
NS  = Fluid motion dataset governed by Navier-Stokes equation.
| Model    |Pretraining Dataset   |  $\mu = 5$         | $\mu = 1$| $\mu =10$|
|----------|----------------------|---------------------|---------|---------|
| u-net    |                      | 0.140              | 0.234  | 0.225  |
| Vit      |                      | 0.051             | 0.184  | 0.046 |  
| GINO     |                      | 0.102               | 0.113  | 0.107  |
| DeepO    |                      |  0.113              | 0.107   | 0.350   |
| GNN      |                      | 0.013             | 0.100  | 0.017 |  
| CoDA-NO  | NS-EW                | 0.035             | 0.066 | 0.043 |  
| CoDA-NO  | NS                   | 0.034             | 0.056 | 0.037 | 

The training and inference time of CoDA-NO and other baselines are listed in the following table. The time required by CoDA-NO is larger than that of the other baselines. 

#### Inference Time in Sec.

| Models          | GNN   | GINO  | DeepO | ViT   | Unet  | CoDA-NO |
|-----------------|-------|-------|-------|-------|-------|---------|
| Inference Time  | 0.012 | 0.012 | 0.006 | 0.071 | 0.024 | 0.440   |
| Training Time   | 0.136 | 0.136 | 0.131 | 0.273 | 0.268 | 1.250   | 

CoDA-NO, despite its additional complexity, presents a compelling case for adoption. Its ability to adapt seamlessly to different PDEs with varying variables and physical coefficients, coupled with its remarkable performance gap of 32% and zero-shot super-resolution capability, marks it as a justifiable choice. Although the implementation of a complex system may seem daunting, the benefits it offers make it a worthwhile investment.

> Choice of Self-Supervised Learning (SSL) Objective
 
SSL via standard autoregressive prediction task requires the pertaining data to be recorded in a fixed time interval. To allow the proposed CoDA-NO model to be pretrained on data with irregular time intervals, we use self-supervised (masked—reconstruction) reconstruction to increase the model's applicability and ability to pre-train only on discrete span shot of the system.

> Discussing on Time Complexity

We will report the inference and training time of the proposed model along with the baselines in the revised manuscript.


[1] Kovachki, Nikola, et al. "Neural operator: Learning maps between function spaces with applications to pdes."

[2] Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020). "Neural operator: Graph kernel network for partial differential equations."


## Reviewer z3xS 
We thank the reviewer z3xS for reviewing and appreciating the work. We will now address the concerns

> Missing Reference on Traditional Solvers being Expensive

We thank the reviewer for pointing this out. The obstacles in generating data using traditional solvers are discussed in [1,2]. We will add these citations in the revised manuscripts.

> On the Motivation of Each of the Model's Components and Difference with Existing Self-Attention

The motivation of the CoDA-NO layer is discussed in the paragraph “Permutation Equivariant Neural Operator” [line 183]. We aim to develop a model that can seamlessly adapt from single-physics to multi-physics. For this, the model should be able to handle an arbitrary number of variables (codomain) and need to be equivariant with respect to the ordering variables (codomains). As the attention mechanism is a set operation, we design a new neural operator layer that employs the attention mechanism among all the co-domains (or variables), treating each PDE as a set of interacting variables.

Variable-specific positional encoding (VSPE) is required to inform the model of each variable's identity. As the model is permutation equivariant, VSPE helps the model capture variable-dependent interactions (e.g., how the velocity variable interacts with displacement).

The normalization layer is a very crucial component of the transformer architecture [3]. However, the designed normalization layer will break the resolution invariance nature of neural operator mapping between function spaces. So, we propose a normalization layer for function spaces, which can be seen as an extension of the instance norm [line 244].

The patching done in the traditional transformer/attention mechanism is not applicable to function space data as they break resolution invariance. In CoDA-NO, we avoid patching and have designed a technique to compute attention over the entire variable. Here, the variables are functions; as an example, for our experiments, the variables are functions on the 2D domain (xy plane). As we are working with functions, we cannot utilize the key, query, and value matrix, as they are not designed to operate over infinite-dimensional (function) spaces. Consequently, we have redesigned the attention mechanism to work with infinite-dimensional vector spaces. Our technique is generalizable and can function in arbitray domains.

Also, compared to the fixed positional encoding in regular transformer - we use learnable variable-specific positional encoding to convey variable-specific information to the model. Our proposed CoDA-NO layer, along with VSPE and normalization, offers, to the best of our knowledge, the first complete transformer architecture for function spaces.

To further clarify these motivations and differences, we will rewrite the revised manuscript highlighting these points.

> Regarding Figure 5

We thank the reviewer for pointing this out. To reduce the clutter, we will divide the curves into baseline and ablation studies and report them separately.

> Error Bar over multiple Runs

Here we present the error bar for Table 1-2 over three runs. To avoid clutter, we report the number separately of the fluid-solid interaction (NS-EW) dataset and fluid flow (NS) dataset.

#### Table: Error Bar
 
Results on the NS-EW dataset

| Models | Pre-training Dataset | # Train = 5                         | # Train=25                            | # Train=100                         |
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

| Models | Pre-training Dataset | # Train = 5             | # Train=25             | # Train=100             |
|--------|----------------------|------------------------|-----------------------|------------------------|
| GINO   |                      | 0.1436 $\pm$ 0.02315         | 0.03714 $\pm$ 0.0044        |  0.0330 $\pm$ 0.0105                      |
| DeepO  |                      | 1.621 $\pm$ 0.2417           | 0.3162 $\pm$ 0.1146         | 0.1978 $\pm$ 0.03452                       |
| GNN    |                      | 0.0113 $\pm$ 0.0046         | 0.00461 $\pm$ 0.00129       |  0.0051 $\pm$ 0.0024                      |
| ViT    |                      |  0.1961 $\pm$ 0.0326         | 0.04098 $\pm$ 0.0057        | 0.03028 $\pm$ 0.01606                       |
| U-net  |                      |  2.94 $\pm$ 1.989            |   0.6568 $\pm$ 0.3635       |  0.3955 $\pm$ 0.3987                      |
| Ours   |                      | 0.0612 $\pm$ 0.0364          |  0.00942 $\pm$ 0.0006       |  0.00453 $\pm$ 0.0006                      |
| Ours   | NS                   |  0.0276 $\pm$ 0.00321        | 0.00572 $\pm$ 0.0005        |  0.00390 $\pm$ 0.0001                      |
| Ours   | NS-EW                |  0.0273 $\pm$ 0.0054         | 0.005665 $\pm$ 0.0005       | 0.004023 $\pm$ 0.0001                      |

> Ablation on Different Model Components

Table 2 presents the ablation of our proposed CoDA-NO layer and pre-training mechanism. In the ViT model, we use the regular attention layer instead of our proposed CoDA-NO layer, which serves as an ablation of the newly proposed CoDA-NO layer. We also present the result where no pre-training is performed. 
We'd like to present an ablation of other proposed components here. We will report their results in the revised manuscript.

#### Table: Ablation Study

"*" Symbol means the model fails to converge and has a very high train error.
VSPE: Variable Specific Positional Encoding.
  
| Models             | Pre-training Dataset | # Train = 5 (NS / NS-EW) | # Train = 25 (NS / NS-EW) | # Train = 100 (NS / NS-EW) |
|--------------------|----------------------|-------------------------|--------------------------|---------------------------|
| No CoDA-NO layer   | -                    | 0.271 / 0.211           | 0.061 / 0.113            | 0.017 / 0.020             |
| CoDA-NO            | -                    | 0.182 / 0.051           | 0.008 / 0.084            | 0.006 / 0.004             |
| CoDA-No (NO VSPE)  | NS                   | 0.049 / 0.079           | 0.009 / 0.0132           | 0.004 / 0.009             |
| CoDA-No (NO VSPE)  | NS EW                | 0.045 / 0.057           | 0.010 / 0.011            | 0.008 / 0.004             |
| CoDA-No (NO Norm.) | NS                   |  * / *                  | 0.023 / *                | 0.008 / 0.006             |
| CoDA-No (NO Norm.) | NS EW                | 0.057 / 0.232           | 0.012 / 0.052            | 0.006 / 0.006             |
| CoDA-No            | NS                   | 0.025 / 0.071           | 0.007 / 0.008            | 0.004 / 0.005             |
| CoDA-No            | NS EW                | 0.024 / 0.040           | 0.006 / 0.005            | 0.005 / 0.003             |

 


> Parameter Comparison among Different Models

We report the parameter count of different baselines in the following table. The architectures of the baselines are adapted from their publicly released implementations. For CoDA-NO, the Encoder contains $34 \times 1e6$ parameter, which is pretrained and reused for every experiment. And the predictor contains $9 \times 1e6$ parameters.
#### Table: Models' Paramter
| Models          | GNN   | GINO  | DeepO | ViT   | Unet  | CoDA-NO |
|-----------------|-------|-------|-------|-------|-------|---------|
| # Parameter x 1e6 | 0.6 | 60    | 6     | 27    | 30    |43       |

It might seem that models are not compared fairly, as the CoDA-NO has a smaller parameter count. However, here, we test the model on a few shot learning problems. Increasing the model's parameter count worsens the overfitting problem.

To prove this fact, we perform experiments on a fluid-solid interaction dataset with increased parameter count. We will observe that increasing the parameter count almost always hurts the performance, especially for very few hot learning scenarios.

#### Table: Overfitting of basslines with higher parameters on NS-EW dataset.
| models  | # Parameter (Used/Big) x 1e6 | # Train = 5  (Used / Big) | # Train=25  (Used/ Big) | # Train=100  (Used / Big) |
|---------|------------------------------|--------------------------|------------------------|--------------------------|
| GINO    | 60/200                       | 0.122 / 0.342           | 0.053 / 0.066        | 0.043 / 0.036            |
| DeepO   | 6 / 25                       | 0.482 /  0.495          | 0.198 /  0.303        | 0.107 / 0.083            |
| GNN     | 0.6/ 7                       | 0.045 / 0.268           | 0.009 / 0.031        | 0.009 / 0.061            |
| ViT     | 27/100                       | 0.211 / 0.266           | 0.113 / 0.125         | 0.020 / 0.022            |
| U-net   | 30/48                        | 3.579 / 9.462            | 0.842 / 3.957          | 0.203 / 0.412            |

> On Limitation

The major limitation of the motivation and novelty of the proposed module is addressed above in the discussion.


[1].Schneider, T., Teixeira, J., Bretherton, C. S., Brient, F., Pressel, K. G., Sch  ̈ar, C., and Siebesma, A. P. Climate goals and computing the future of clouds.

[2]. Keyes, D. E., McInnes, L. C., Woodward, C., Gropp, W., Myra, E., Pernice, M., ... & Wohlmuth, B. (2013). Multiphysics simulations: Challenges and opportunities.

[3] Xiong, Ruibin, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tieyan Liu. "On layer normalization in the transformer architecture."




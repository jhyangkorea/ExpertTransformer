# ExpertTransformer
Expert Transformer (ET) is a fine-grained Mixture-of-Experts (MoE) building block for Transformer architectures that enables specialization at both the sequence and token levels. The module replaces standard self-attention and feed-forward sublayers with routed attention experts and routed FFN experts

# How to Use This Code
1. Install Requirements

Install the required dependencies:

pip install -r requirements.txt

2. Download Datasets

Most datasets can be downloaded from the Autoformer repository (https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy).
The NN5 dataset is available from the Monash Time Series Forecasting Repository(https://forecastingdata.org/)

After downloading, create a folder named ./dataset
Place all dataset .csv files inside this directory

3. Training

All training scripts are located in:

./scripts/PExpertTransfor

For example, to run multivariate forecasting on the ETTh1 dataset without multi-step training

sh ./scripts/PExpertPatchTST/ETTh1.sh

After training completes, results will be saved in:

./result.txt

You can modify the number of attention experts, FFN experts, and active experts directly in the corresponding .sh script file.

4. Multi-Stage Training

The multi-stage training procedure must be executed in the specific order as defined in:

multi_step_etth1.sh

Please follow the sequence outlined in that script to reproduce the staged training results.

**Acknowledgement**

This implementation is built on top of the PatchTST framework:

https://github.com/yuqinie98/PatchTST

We thank the original authors for making their code publicly available.

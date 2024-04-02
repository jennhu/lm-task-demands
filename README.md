# Auxiliary task demands mask the capabilities of smaller language models

This repository contains code and data for the paper "Auxiliary task demands mask the capabilities of smaller language models" by Jennifer Hu & Michael C. Frank.

## 0. Code structure

The repository is structured with the following main folders:

- `configs`: YAML files used to control experiment settings
- `external_stimuli`: CSV files containing stimuli from external datasets
- `figures`: rendered figures used in the paper
- `notebooks`: Jupyter notebooks for all analyses and reproducing figures
- `output`: CSV files with outputs from model runs
- `src`: Python scripts for implementing the experiments

## 1. Reproducing analyses/figures

If you simply want to reproduce the analyses and figures from the paper,
please see the Jupyter notebooks in the `notebooks` folder. 
These notebooks only require basic scientific Python and visualization tools
(e.g., `pandas`, `numpy`, `seaborn`, `matplotlib`).

- `model_size.ipynb`: reads data from `output` and generates the subplots from Figures 2-3 in the paper (model size experiments using 13 language models)
- `training_time.ipynb`: reads data from `output` and generates the subplots from Figure 4 in the paper (training time experiments using checkpoints of OLMo 7B)

By default, figures are saved to the `figures` directory.

## 2. Reproducing results

If you want to re-run our experiments, or make any modifications,
please see the `src` folder, which contains all necessary scripts and code.

### a) Dependencies

The packages used in our experiments can be found in the `requirements.txt` file.
To create a new environment based on these requirements, please run:
```bash
conda create --name taskdemands --file requirements.txt
```
Then activate the environment with:
```bash
conda activate taskdemands
```

### b) Running experiments

The main controller script is `src/run_experiment.py`. It can take either a YAML config file (see the `configs` folder for examples) or standard Python command-line arguments.

#### i. Configuation files

The config files in the `configs` folder were generated by running the script `python configs/make_configs.py`. You can customize it for your own purposes if you want to add models or tasks.

Config files are named with the convention `configs/<TASK>/<MODEL>.yaml`. When there are subconditions (as is the case for the reflective reasoning `crt` task), then the files are named `configs/<TASK>/<CONDITION>/<MODEL>.yaml`.

#### ii. Naming conventions

The code and data use the following naming conventions for tasks/datasets/domains:

| Task ID      | Cognitive domain  | Reference |
| ------------ | --------------------------- | --------- |
| `digit_mat`  | Analogical reasoning        | [Webb et al. (2023)](https://www.nature.com/articles/s41562-023-01659-w)  |
| `crt`        | Reflective reasoning        | [Hagendorff et al. (2023)](https://www.nature.com/articles/s43588-023-00527-x) |
| `lambada`    | Word prediction             | [Paperno et al. (2016)](https://aclanthology.org/P16-1144/) |
| `blimp`      | Grammaticality judgment     | [Warstadt et al. (2020)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00321/96452/BLiMP-The-Benchmark-of-Linguistic-Minimal-Pairs) |
| `dgl`        | Grammaticality judgment     | [Dentella et al. (2023)](https://www.pnas.org/doi/10.1073/pnas.2309583120); [Hu et al. (2024)](https://arxiv.org/abs/2402.01676) |

As mentioned above, the `crt` task additionally has three conditions (`crt1`, `crt2`, and `crt3`),
corresponding to the three types of tests conducted by [Hagendorff et al. (2023)](https://www.nature.com/articles/s43588-023-00527-x).
Please see their paper for more details.

Models are typically named by their Huggingface model identifier within the CSV output files, but the output files themselves are saved without the initial organization name (e.g., "Mistral-7B-v0.1").

#### iii. Example usage

1. Example with config file (analogical reasoning task, Gemma 2B model):
```bash
python src/run_experiment.py -c configs/digit_mat/gemma-2b.yaml
```

2. Example with command-line arguments:
```bash
python src/run_experiment.py \
    --model mistralai/Mistral-7B-v0.1 \
    --task crt \
    --condition crt2
```

3. Example with large model (>7B parameters) requiring `accelerate` with multiple GPUs:
```bash
python src/run_experiment.py \
    --model meta-llama/Llama-2-70b-hf \
    --accelerate \
    --task blimp
```

4. Example with OLMo training steps ("revisions"):
```bash
python src/run_experiment.py \
    --model allenai/OLMo-7B \
    --task lambada \
    --revision step10000-tokens44B
```
In our paper, we used the following revisions (not including the final checkpoint):
```
step10000-tokens44B
step31000-tokens137B
step91000-tokens403B
step151000-tokens668B
step211000-tokens933B
step271000-tokens1199B
step391000-tokens1730B
step481000-tokens2128B
step541000-tokens2393B
```
The full list of OLMo 7B revisions is available at [https://huggingface.co/allenai/OLMo-7B/blob/main/revisions.txt](https://huggingface.co/allenai/OLMo-7B/blob/main/revisions.txt).

#### iv. Huggingface token

In order to access gated models such as Llama-2, you will need to provide a Huggingface token.
By default, the `src/run_experiment.py` script will look for this token at the path `src/hf_token.txt`, which is ignored by git. You can change this path with the argument `--hf_token_path`.

Alternatively, if you'd like to set your token using another method, you can edit the relevant lines in `src/run_experiment.py` (lines 44-45).

#### v. Annotations

Please note that the model outputs for the `crt` task (reflective reasoning)
were manually annotated. The output CSV files have two relevant columns:

- `generated_response_naive_label`: automatically generated label based on simple pattern matching. This is often wrong, but it's useful as a first-pass.
- `generated_response_label`: our manual annotations. We focused on annotating "correct" vs "intuitive" answers; occasionally we simply leave the naive label in place if the answer is neither of the correct or intuitive options.
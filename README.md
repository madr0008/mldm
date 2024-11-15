Repository for the model presented in the soon to be published article “Addressing Multilabel Imbalance with an Efficiency-Focused Approach Using Diffusion Model-Generated Synthetic Samples”.

This is the implementation of a diffusion model for oversampling multi-label data.
This implementation is an adaptation of the [TabDDPM model](https://github.com/rotot0/tab-ddpm).

## Running the model
1. Install [conda](https://docs.conda.io/en/latest/miniconda.html) in order to manage the virtual environment.
2. Execute the following commands to create the environment and install the necessary dependencies:
    ```bash
    export REPO_DIR=/path/to/the/code
    cd $REPO_DIR

    conda create -n mldm python=3.9.7
    conda activate mldm

    pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt

    conda env config vars set PYTHONPATH=${PYTHONPATH}:${REPO_DIR}
    conda env config vars set PROJECT_DIR=${REPO_DIR}

    conda deactivate
    conda activate mldm
    ```

A Docker container can be built from the [Dockerfile](Dockerfile), with the required libraries pre-installed. However, you will still need to create the conda virtual environment by following the commands mentioned above.

### Datasets

The multi-label datasets (MLD) supported by the algorithm are those in ARFF format, accompanied by an XML file specifying the label names. This format is the same used by the [MULAN library](https://mulan.sourceforge.net/).

The [Cometa](https://cometa.ujaen.es/) repository gathers a wide variety of MLDs, either complete or pre-partitioned.

### Running the Algorithm

In order to execute the algorithm on a dataset, simply run the following commands:

``` bash
conda activate mldm
cd $PROJECT_DIR
python scripts/pipeline.py --config_file=config.toml
```

The parameters for running the model are specified in a configuration file in toml format. The structure and parameters included in this file are explained [here](CONFIG_DESCRIPTION.md).

## File structure
`mldm/` -- Directory containing the implementation of the proposed method

- `mldm/gaussian_multinomial_diffusion.py` -- diffusion model
- `mldm/modules.py` -- additional modules forming the main model
- `mldm/utils.py` -- mathematical functions for the model

`scripts/` -- Directory containing project scripts

- `scripts/pipeline.py` -- main script for invoking training and sampling processes
- `scripts/sample.py` -- script for the sampling process
- `scripts/train.py` -- script for the training process
- `scripts/utils_train.py` -- script with auxiliary functions for training

`lib/` -- Directory containing local libraries for the project

- `lib/data.py` -- definition of classes and methods for working with MLDs
- `lib/util.py` -- script with auxiliary functions for training

## References

This project is based on prior work reflected in the following papers:

- Kotelnikov, A., Baranchuk, D., Rubachev, I., & Babenko, A. (2022). TabDDPM: Modelling Tabular Data with Diffusion Models. arXiv preprint arXiv:2209.15421.


- Kim, J., Lee, C., Shin, Y., Park, S., Kim, M., Park, N., & Cho, J. (2022, August). Sos: Score-based oversampling for tabular data. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (pp. 762-772).

#!/bin/bash

# check env.yml
ENV_PREFIX="$WORKSPACE/env"
ENV_YAML="$WORKSPACE/env.yml"
ENV_CACHE_YAML="$WORKSPACE/env/env.yml"
if [ ! -f $ENV_YAML ]; then
    echo "$ENV_YAML is not found."
    exit 1
fi

# update env
if diff --brief <(cat $ENV_CACHE_YAML 2>/dev/null) <(cat $ENV_YAML); then
    echo "Environment is up to date"
else
    if [ -d ./env ]; then
        echo "Environment is out of date, removing old environment"
        rm -rf ./env
    fi
    echo "Creating new environment"

    source $CONDA_BASE_PREFIX/etc/profile.d/conda.sh && \
    conda init bash && \
    conda config --show && 
    conda clean -i -y && \
    conda env create -p $ENV_PREFIX -f $ENV_YAML && \
    conda activate $ENV_PREFIX && \
    cp $ENV_YAML $ENV_CACHE_YAML
fi

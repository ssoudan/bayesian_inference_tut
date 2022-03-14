# Logistic Functions and Probabilistic Programming

Sebastien Soudan

Quick tutorial on Bayesian inference. See the [notebook](bayesian_inference.ipynb) for the actual code.

(If you know a way to have latex equations in GH markdown that can be read both in light and dark themes, let me know.)

The notebook is [here](bayesian_inference.ipynb).

## Running the notebook

This is a Jupyter notebook.  

### Deps

    conda create --name bayesian_inference_tut python=3.10
    conda activate bayesian_inference_tut 
    conda install -y -c conda-forge numpyro matplotlib scipy pandas seaborn plotly arviz 
    pip install jax jaxlib graphviz ipyimpl jupyter

or on a M1 Mac:

    conda create -f environment.yml
    conda activate bayesian_inference_tut
    pip install jupyter

### Running 

    conda activate bayesian_inference_tut
    jupyter notebook bayesian_inference.ipynb

### Animation

To create the animation with ImageMagick, see [img/ANIMATE.sh](img/ANIMATE.sh).

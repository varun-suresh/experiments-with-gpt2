# Experiments with GPT-2
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/language-models-are-unsupervised-multitask/sentiment-analysis-on-imdb)](https://paperswithcode.com/sota/sentiment-analysis-on-imdb?p=language-models-are-unsupervised-multitask)

In this repo, I want to experiment with GPT-2 (124M parameter) model and understand how to train and fine tune it well. Instead of using the [Hugging Face](https://huggingface.co/) implementation, I followed [Andrej Karpathy's nanoGPT implementation](https://github.com/karpathy/nanoGPT/tree/master) and made changes wherever necessary.

As a first experiment, I fine tuned the model for sentiment classification. Results and code are in the sentiment classification folder

## Setup
This repository was setup using [uv](https://github.com/astral-sh/uv). To setup on your computer, install uv using
```
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```
If you are on windows, you can follow the instrunctions on the UV page. 

Once you have UV installed, clone the repository by running
```
git clone https://github.com/varun-suresh/experiments-with-gpt2
```
From the project folder, run 
```
uv sync
```
In the `.env` file, add the root directory of this repository to the PYTHONPATH env variable. Then from the root folder of this repo, run
```
export UV_ENV_FILE=.env
```
You should now have everything setup! 
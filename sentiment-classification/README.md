# Sentiment classification using GPT-2

## Summary
Using pre-trained GPT-2  weights from [Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/gpt2), I finetuned the model on the [large movie review dataset](https://ai.stanford.edu/~amaas/data/sentiment/) to predict whether a review is positive or negative.

I ran the experiment using GPT-2 (124M parameters) in three configurations using :
1. Zero Shot : Using the pre-trained GPT-2 as is, I wrote a prompt to predict whether a review is positive or negative.
2. Fine-Tuned : I fine-tuned the last two transformer blocks in the model. I also replaced the Language Modeling head with a binary classification head.
3. Fine-Tuned (LoRA): I fine-tuned the last two transformer blocks using [LoRA](https://arxiv.org/abs/2106.09685).

| Method            |   accuracy |   precision |   recall |
|:-------------     |-----------:|------------:|---------:|
| Zero Shot         |   0.70784  |    0.83863  | 0.51472  |
| Fine-Tuned        |   0.92360  |    0.92923  | 0.91704  |
| Fine-Tuned(LoRA)  |   0.91068  |    0.89946  | 0.92472  |


## Setup
### Hardware:
I used an Nvidia GEForce GTX 1080 (8GB memory and 10Gbps memory speed) for fine-tuning. 

### Data
I downloaded the movie reviews from the [large movie review dataset](https://ai.stanford.edu/~amaas/data/sentiment/). It contains 25k train samples and 25k test samples. In each set, there are 12.5k positive and negative samples.

### Encoding
I used [tiktoken](https://github.com/openai/tiktoken) for GPT-2. This was the tokenizer used by OpenAI when training the model.

### Data size
I used the last ~250 tokens in the review (256 including the prompt). This was to keep the train and eval times short and to speed up experiments. The prompt I used was "Review: *review* Sentiment:". For example, if the movie review is "The movie was really fun", my prompt would be "Review: The movie was really fun Sentiment:"

### Fine-tuning
The GPT-2 model contains 12 transformer blocks followed by a language modeling head. I froze the first 10 transformer blocks, only fine-tuning on the last two blocks. Additionally, I included a binary classification head instead of the language modeling head.

### LoRA
I used Microsoft's [loralib](https://github.com/microsoft/LoRA) package on the last two transformer blocks on the query and value weights with rank = 8.

## Experiments

### Zero Shot Learning
Given a prompt like "Review: The movie was awesome! Sentiment:", I compare the likelihood of the next token being " Positive" and " Negative" and classify the review.

To try this out, you can use the notebook [sentiment_classification.ipynb](https://github.com/varun-suresh/experiments-with-gpt2/blob/main/sentiment-classification/sentiment_classification.ipynb). You can try out different prompts by modifying [reviewsDataset.py](https://github.com/varun-suresh/experiments-with-gpt2/blob/main/sentiment-classification/reviewsDataset.py#L17)

I tried a few prompts and the results varied quite a lot. An additional space ("Positive" and " Positive") results in different tokens and the model is quite sensitive to these prompts. Among the few prompts I tried, the one I used eventually had the best results. 

### Fine Tuning
In this setting, I froze the Positional Embedding weights, Token Encoding weights and the first 10 transformer blocks. Instead of the language modeling head, I used a binary classification head (a fully-connected layer with just one output followed by a sigmoid to make the output between 0 and 1 where 0 is negative and 1 is positive). I used the binary cross entropy loss function. 

To fine-tune the model, you can use the notebook [sentiment_classification.ipynb](https://github.com/varun-suresh/experiments-with-gpt2/blob/main/sentiment-classification/sentiment_classification.ipynb). The model parameters are in [gpt_config.py](https://github.com/varun-suresh/experiments-with-gpt2/blob/main/gpt_config.py) and the training parameters are in [train_config.py](https://github.com/varun-suresh/experiments-with-gpt2/blob/main/sentiment-classification/train_config.py). The evaluation code is in [eval.py](https://github.com/varun-suresh/experiments-with-gpt2/blob/main/sentiment-classification/eval.py). The training and fine-tuning can all be run from the sentiment classification notebook.

Learnings:
1. To avoid vanishing/exploding gradients, I clipped the gradients at 5.
2. In the first 2k iterations of training, I increased the learning rate from 0 to 1e-4 (warmup). This may not be particularly important as I was starting from a pre-trained model but seems to be really important if training from scratch.
3. Plotted the parameters and the gradients of the parameters in tensorboard. It was really helpful in figuring out the vanishing/exploding gradient problems.
4. Since the dataset I'm finetuning on is small, dropout (regularization) is really important.

Parameter count when fine-tuning:

Transformer Block Layer 11: 
```
Query Weights = 768 * 768 + 768 (Embedding size = 768, Weights + Bias) = 590592
Key Weights = 768 * 768 + 768
Value Weights = 768 * 768 + 768 = 590592
Layer Norm (2) = 768 * 2 (gamma and beta) * 2(2 layer norms in a transformer block) = 3072
Feed Forward Weights 1 = 768 * (4*768) + 4*768 = 2362368
Feed Forward Weights 2 = 4*768 *768 + 768 = 2360064

Total = 7087872
```
Binary Classification Head
```
Weights = 768 * 1
```

When finetuning, about 14M parameters are being modified (14M out of 124M).

### Fine Tuning with LoRA
When using LoRA, the pre-trained weights are unchanged. It introduces two matrices A and B whose product is added to the weight matrix. Let's consider a concrete example:

Let W_k be a weights matrix. In this case, let's consider the keys weight in a transformer block. The weights have the dimension 768 * 768 and the bias is a 768 dimensional tensor.

Instead of modifying this large matrix, we can write 
```
math
$\ W_k = W_k + $\Delta$ W \$
```

$\ Delta W = AB^T \$

where A and B are two low rank matrices of dimension 768 * 8 . AB^T will result in a matrix of dimension 768 * 768, but the number of learned parameters are 768 * 8 * 2 = 12288 parameters for a transformer block, significantly lower than ~7M parameters when fine tuning all the parameters.

## Results

Since I considered the last 256 tokens for sentiment classification, I calculated the precision, recall and accuracy by the review length binned by size. As expected, the accuracy goes down slightly as the length of the review gets larger ( More of the review gets ignored as it gets longer).


**Results for Zero Shot learning**

| bin          |   TP |   FP |   FN |   TN |   accuracy |   precision |   recall |
|:-------------|-----:|-----:|-----:|-----:|-----------:|------------:|---------:|
| (0, 256]     | 4499 |  807 | 3252 | 6836 |   0.736326 |    0.847908 | 0.580441 |
| (256, 512]   | 1360 |  301 | 1865 | 3164 |   0.676233 |    0.818784 | 0.421705 |
| (512, 768]   |  354 |   78 |  611 |  820 |   0.630166 |    0.819444 | 0.366839 |
| (768, 1024]  |  137 |   39 |  215 |  286 |   0.624815 |    0.778409 | 0.389205 |
| (1024, 1280] |   73 |   10 |  100 |  150 |   0.66967  |    0.879518 | 0.421965 |

**Results for Fine Tuned model**
| bin          |   TP |   FP |   FN |   TN |   accuracy |   precision |   recall |
|:-------------|-----:|-----:|-----:|-----:|-----------:|------------:|---------:|
| (0, 256]     | 7162 |  441 |  589 | 7202 |   0.933091 |    0.941997 | 0.92401  |
| (256, 512]   | 2936 |  276 |  289 | 3189 |   0.915546 |    0.914072 | 0.910388 |
| (512, 768]   |  873 |   95 |   92 |  803 |   0.899624 |    0.90186  | 0.904663 |
| (768, 1024]  |  308 |   45 |   44 |  280 |   0.868538 |    0.872521 | 0.875    |
| (1024, 1280] |  154 |   14 |   19 |  146 |   0.900901 |    0.916667 | 0.890173 |

**Results for Fine Tuned model with LoRA**
| bin          |   TP |   FP |   FN |   TN |   accuracy |   precision |   recall |
|:-------------|-----:|-----:|-----:|-----:|-----------:|------------:|---------:|
| (0, 256]     | 7273 |  710 |  478 | 6933 |   0.922827 |    0.911061 | 0.938331 |
| (256, 512]   | 2919 |  376 |  306 | 3089 |   0.898057 |    0.885888 | 0.905116 |
| (512, 768]   |  866 |  119 |   99 |  779 |   0.882984 |    0.879188 | 0.897409 |
| (768, 1024]  |  314 |   63 |   38 |  262 |   0.850812 |    0.832891 | 0.892045 |
| (1024, 1280] |  156 |   20 |   17 |  140 |   0.888889 |    0.886364 | 0.901734 |

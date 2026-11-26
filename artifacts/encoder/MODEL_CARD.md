---
license: apache-2.0
tags:
- Fake News Detection
- Text Classification
#metrics:
#- accuracy
model-index:
- name: distilroberta-base-finetuned-fake-news-detection
  results: []
---

# distilroberta-base-finetuned-fake-news-detection

This model is a fine-tuned version of [distilroberta-base](https://huggingface.co/distilroberta-base) on [this](https://huggingface.co/datasets/GonzaloA/fake_news) Fake News Detection Dataset, which has been constructed by combining multiple Fake News datasets from Kaggle.

This is the classification report after training for 3 full epochs:

|  | Precision | Recall | F-1 Score | Support |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| Not Hate Speech (0)        | 0.99   | 0.99 | 0.99          | 4335   |
| Hate Speech (1)       | 0.99   | 0.99 | 0.99          | 3782   |
| accuracy        |    |  | 0.99          | 8117   |
| macro avg        | 0.99   | 0.99 | 0.99          | 8117   |
| weighted avg        | 0.99   | 0.99 | 0.99          | 8117   |

## Training and evaluation data

All of the process to train this model is available in [this](https://github.com/vikram71198/Transformers/tree/main/Fake%20News%20Detection) repository. The dataset has been split into 24,353 examples for training & 8,117 examples for validation & testing each.

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 32
- eval_batch_size: 32
- optimizer: default AdamW Optimizer
- num_epochs: 3
- warmup_steps: 500
- weight_decay: 0.01
- random seed: 42

I also trained for 3 full epochs on Colab's Tesla P100-PCIE-16GB GPU.

### Training results

| Epoch | Training Loss | Validation Loss |
|:-------------:|:----:|:---------------:|
| 1        | 0.099100  | 0.042086        |
| 2        | 0.030200  | 0.028448        |
| 3        | 0.017500  | 0.024397       |

## Model in Action 🚀
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
tokenizer = AutoTokenizer.from_pretrained("vikram71198/distilroberta-base-finetuned-fake-news-detection")
model = AutoModelForSequenceClassification.from_pretrained("vikram71198/distilroberta-base-finetuned-fake-news-detection")
#Following the same truncation & padding strategy used while training
encoded_input = tokenizer("Enter any news article to be classified. Can be a list of articles too.", truncation = True, padding = "max_length", max_length = 512, return_tensors='pt')
output = model(**encoded_input)["logits"]
#detaching the output from the computation graph
detached_output = output.detach()
#Applying softmax here for single label classification
softmax = nn.Softmax(dim = 1)
prediction_probabilities = list(softmax(detached_output).detach().numpy())
predictions = []
for x,y in prediction_probabilities:
   predictions.append("not_fake_news") if x > y else predictions.append("fake_news")
print(predictions)
```
Please note that if you're performing inference on a lengthy dataset, split it up into multiple batches, otherwise your RAM will overflow, unless you're using a really high end GPU/TPU setup. I'd recommend a batch length of 50, if you're working with a vanilla GPU setup.
### Framework versions

- Transformers 4.12.5
- Pytorch 1.11.0
- Datasets 1.17.0
- Tokenizers 0.10.3
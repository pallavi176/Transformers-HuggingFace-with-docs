# Processing the data

- Train a sequence classifier on one batch in TensorFlow:

``` py
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = dict(tokenizer(sequences, padding=True, truncation=True, return_tensors="tf"))
# This is new
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
labels = tf.convert_to_tensor([1, 1])
model.train_on_batch(batch, labels)
```
- Better accuracy required bigger dataset.
- Dataset used: MRPC (Microsoft Research Paraphrase Corpus) dataset, introduced in [paper](https://aclanthology.org/I05-5002.pdf) 
- The dataset consists of 5,801 pairs of sentences, with a label indicating if they are paraphrases or not (i.e., if both sentences mean the same thing).

## Loading a dataset from the Hub

- Huggingface dataset library is a library that provides an api to quickly download many public's datasets and preprocess them.
- Huggingface dataset library allows you to easily download and cache datasets from its identifier on dataset hub.
- MRPC dataset is one of the 10 datasets composing the [GLUE benchmark](https://gluebenchmark.com/), which is an academic benchmark that is used to measure the performance of ML models across 10 different text classification tasks.
- MRPC (Microsoft Research Paraphrase Corpus) dataset, introduced in [paper](https://aclanthology.org/I05-5002.pdf) 
- The dataset consists of 5,801 pairs of sentences, with a label indicating if they are paraphrases or not (i.e., if both sentences mean the same thing).

``` py
from datasets import load_dataset
raw_datasets = load_dataset("glue", "mrpc")
raw_datasets
```

- It returns a DatasetDict object which is a sort of dictionary containing each split(train,validation & test set) of the dataset.
- We can access any split of dataset by its key, then any element by index.

``` py
raw_datasets["train"]
```

- Each of the splits contains several columns (sentence1, sentence2, label, and idx) and a variable number of rows, which are the number of elements in each set (so, there are 3,668 pairs of sentences in the training set, 408 in the validation set, and 1,725 in the test set).
- We can access en element(s) by indexing:

``` py
raw_datasets["train"][6]
raw_datasets["train"][:5]
```

- We can also directly get a slice of your dataset.
- The features attributes gives us more information about each column. It gives corresponds between integer and names for tha labels.

``` py
raw_datasets["train"].features
```

- The map method allows you to apply a function over all the splits of a given dataset.

``` py
from transformers import AutoTokenizer
checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def tokenize_function(example):
    return tokenizer(
        example["sentence1"], example["sentence2"], padding="max_length", truncation=True, max_length=128
    )
tokenized_datasets = raw_datasets.map(tokenize_function)
print(tokenized_datasets.column_names)
```

- As long as the function returns a dictionary like object, the map() method will add new columns as needed or update existing ones.
- Result of tokenize_function(): ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids']
- You can preprocess faster by using the option batched=True. The applied function will then receive multiple examples at each call.

``` py
from transformers import AutoTokenizer
checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def tokenize_function(examples):
    return tokenizer(
        examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True, max_length=128
    )
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
```

- We can aslo use multiprocessing with a map method.
- With just a few last tweaks, the dataset is then ready for training!
- We just remove the columns we don't need anymore with the remove column methods, rename label to labels since the model from transformers library expect that and set the output format to desired backend torch, tensorflow or numpy.
- If needed we can also generate a short sample of dataset using the select method.

``` py
tokenized_datasets = tokenized_datasets.remove_columns(["idx", "sentence1", "sentence2"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets = tokenized_datasets.with_format("tensorflow")
tokenized_datasets["train"]
small_train_dataset = tokenized_datasets["train"].select(range(100))
```


## Preprocessing a dataset

## Dynamic padding


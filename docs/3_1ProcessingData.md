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

### Preprocessing sentence pairs (TensorFlow)

- We have seen before how to tokenize single sentences and batch them together.

``` py
from transformers import AutoTokenizer
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="tf")
```

- But text classification can also be applied on pairs of sentences. 
- In a problem called Natural Language Inference(NLI), were we check whether a pair of sentences are logically related or not.
- In fact, the GLUE benchmark, 8 of the 10 tasks concern pair of sentences.
    - Datasets with single sentences: COLA, SST-2
    - Datasets with pairs of sentences: MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI
- Models like BERT are pretrained to recognize relationships between 2 sentences.
- The tokenizers accept sentence pairs as well as single sentences.

``` py
from transformers import AutoTokenizer
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer("My name is Pallavi Saxena.", "I work at Avalara.")
```

- It returns a new field called "token_type_ids" which tells the model which tokens belong to the first sentence and which ones belong to the second sentence.
- The tokenizer adds special tokens for the corresponding model and prepares "token_type_ids" to indicate which part of the inputs correspond to which sentence.
- To process several pairs of sentences together, just pass the list of first sentences followed by the list of second sentences.

``` py
from transformers import AutoTokenizer
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer(
    ["My name is Pallavi Saxena.", "Going to the cinema."],
    ["I work at Avalara.", "This movie is great."],
    padding=True
)
```

- Tokenizers prepare the proper token type IDs and attention masks.
- Those inputs are then ready to go through a sequence classification model!

``` py
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
batch = tokenizer(
    ["My name is Pallavi Saxena.", "Going to the cinema."],
    ["I work at Avalara.", "This movie is great."],
    padding=True,
    return_tensors="tf",
)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**batch)
```

- All model checkpoint layers were used when initializing TFBertForSequenceClassification.
- Some layers of TFBertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier']
- You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

## Dynamic padding

- We need to pad sentences of different lengths to make batches.
- The first way to do this is to pad all the sentences in the whole datset to the maximum length in the dataset.
- Its downside is sentences with short sentences will have a lot of padding tokens, which will include more computations in the model which actually not needed.
- Another way is to pad the sentences at the batch creation, to the length of the longest sentence; a technique called dynamic padding.
    - Pros: All the batches will have the smallest size possible.
    - Con: dynamic shapes don't work well on the accelerators. All batches have different shapes slows down things on accelerators like TPUs.
- In practice, here is how we can preprocess the MRPC dataset with fixed padding:

``` py
from datasets import load_dataset
from transformers import AutoTokenizer
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def tokenize_function(examples):
    return tokenizer(
        examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True, max_length=128
    )
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["idx", "sentence1", "sentence2"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets = tokenized_datasets.with_format("torch")
```

- We can then use our dataset in a standard PyTorch DataLoader. As expected, we get batches of fixed shapes

``` py
from torch.utils.data import DataLoader
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=16, shuffle=True)
for step, batch in enumerate(train_dataloader):
    print(batch["input_ids"].shape)
    if step > 5:
        break
```

- To apply dynamic padding, we postpone the padding in the preprocessing function.

``` py
from datasets import load_dataset
from transformers import AutoTokenizer
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["idx", "sentence1", "sentence2"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets = tokenized_datasets.with_format("torch")
```

- Each batch then has a different size, but there is no needless padding.

``` py
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer)
train_dataloader = DataLoader(
    tokenized_datasets["train"], batch_size=16, shuffle=True, collate_fn=data_collator
)
for step, batch in enumerate(train_dataloader):
    print(batch["input_ids"].shape)
    if step > 5:
        break
```
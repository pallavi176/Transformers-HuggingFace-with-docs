# Handling multiple sequences

## Batching inputs together:
- Sentences we want to group inside a batch will often have different lengths

``` py
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
sentences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this.",
]
tokens = [tokenizer.tokenize(sentence) for sentence in sentences]
ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
print(ids[0])
print(ids[1])
#[1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012]
#[1045, 5223, 2023, 1012]
```

- You can't build a tensor with lists of different lengths

``` py
import tensorflow as tf
ids = [[1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012],
       [1045, 5223, 2023, 1012]]
input_ids = tf.convert_to_tensor(ids) # Error: not a rectangualr shape
```

- Generally, we only truncate sentences when they are longer than the maximum length the model can handle
- Which is why we usually pad the smaller sentences to the length of the longest one!

``` py
import tensorflow as tf
ids = [[1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012],
       [1045, 5223, 2023, 1012,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0]]
input_ids = tf.convert_to_tensor(ids)
input_ids
```

``` py
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token_id      # Applying padding here
```

- Now that we have padded our sentences we can make a batch with them
- But just passing this through a transformers model will not give the right results.

``` py
from transformers import TFAutoModelForSequenceClassification
ids1 = tf.convert_to_tensor(
    [[1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012]]
)
ids2 = tf.convert_to_tensor([[1045, 5223, 2023, 1012]])
all_ids = tf.convert_to_tensor(
    [[1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012],
     [1045, 5223, 2023, 1012,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0]]
)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
print(model(ids1).logits)
print(model(ids2).logits)
print(model(all_ids).logits)
"""
tensor([[-2.7276,  2.8789]], grad_fn=<AddmmBackward>)
tensor([[ 3.9497, -3.1357]], grad_fn=<AddmmBackward>)
tensor([[-2.7276,  2.8789],
        [ 1.5444, -1.3998]], grad_fn=<AddmmBackward>)
"""
```

- This is because the attention layers use the padding tokens in the context they look at for each token in the sentence.
    - Attention layers attend just the 4 tokens: [I, hate, this, !]
    - Attention layers attend the 4 tokens and all padding tokens: [I, hate, this, !, [PAD], [PAD], [PAD], [PAD]]

- To tell the attention layers to ignore the padding tokens, we need to pass them an attention mask.

``` py
all_ids = tf.convert_to_tensor(
    [[1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012],
     [1045, 5223, 2023, 1012,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0]]
)
# adding attention by creating attention mask
attention_mask = tf.convert_to_tensor(
    [[   1,    1,    1,    1,    1,    1,    1,     1,     1,    1,    1,    1,    1,    1],
     [   1,    1,    1,    1,    0,    0,    0,     0,     0,    0,    0,    0,    0,    0]]
)
```

- Here, attention layers will ignore the tokens marked with 0 in the attention mask.

- With the proper attention mask, predictions are the same for a given sentence, with or without padding.

``` py
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
output1 = model(ids1)
output2 = model(ids2)
print(output1.logits)
print(output2.logits)
output = model(all_ids, attention_mask=attention_mask)
print(output.logits)
```

- Using with padding=True, the tokenizer can directly prepare the inputs with padding and the proper attention mask:

``` py
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
sentences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this.",
]
print(tokenizer(sentences, padding=True))
```

## Preprocessing sentence pairs (TensorFlow)

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

- But text classification can also be applied on pairs of sentences
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


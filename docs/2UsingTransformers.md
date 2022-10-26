# Behind the pipeline

### Preprocessing with a tokenizer
    -> All preprocessing needs to be done in exactly the same way as when the model was pretrained.
    -> To do this, we use the AutoTokenizer class and its from_pretrained() method.
    ->  Using the checkpoint name of our model, it will automatically fetch the data associated with the model’s tokenizer and cache it.

``` py
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

    -> Next step is to convert the list of input IDs to tensors.
    -> To specify the type of tensors we want to get back (PyTorch, TensorFlow, or plain NumPy), we use the return_tensors argument:

``` py
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="tf")
print(inputs)
```

    -> The output itself is a dictionary containing two keys, input_ids and attention_mask. 
    -> input_ids contains two rows of integers (one for each sentence) that are the unique identifiers of the tokens in each sentence.

### Going through the model

# Tokenizers:
- In NLP, most of the data we handle is raw text.
- Tokenizers is used to transform raw text to numbers.
- Tokenizer's objective is to find a meaningful representation.
- 3 distict tokenizations:
    - Word-based
    - Character-based
    - Subword-based

## Word-based
- The text is split on spaces
- Other rules, such as punctuation, may be added
- In this algorithm, each word has a specific ID/number attributed to it
- Limits: 
    - very similar words have entirely different meanings. eg: dog vs dogs
    - the vocabulary can end up very large due to lot of different words
    - large vocabularies result in heavy models
    - loss of meaning across very similar words
    - large quantity of out-of-vocabulary tokens
- We can limit the amount of words we add to the vocabulary by ignoring certain words which are not necessary
- Out of vocabulary words result in a loss of information, since model will have the exact same representation for all words that it doesn't know, [UNK], unknown.

``` py
tokenized_text = "Jim Henson was a puppeteer".split()
print(tokenized_text)
# ['Jim', 'Henson', 'was', 'a', 'puppeteer']
```

## Character-based
- Splitting a raw text into characters
- Character-based vocabularies are slimmer
- There are much fewer out-of-vocabulary (unknown) tokens, since every word can be built from characters.
- Limits: 
    - intuitively, it’s less meaningful: each character doesn’t mean a lot on its own in compared to any word
    - Their sequences are translated into very large amounts of tokens to be processed by the model and this can have an impact on the size of the contants the model will carry around and it will reduce the size of the text we can use as input for a model which is often limited
    - very long sequences
    - less meaningful individual tokens

## Subword tokenization
- Splitting a raw text into subwords
- Finding a middle ground between word and character-based algorithms
- Subword-based tokenization lies between character and worf-based algorithm
    - Frequently used words should not be split into smaller subwords
    - Rare words should be decomposed into meaningful subwords
- Subwords help identify similar syntactic or semantic situations in text
- Subword tokenization algorithms can identify start of word tokens and which tokens complete start of words
- Most models obtaining state-of-the-art results in English today use some kind of subword-tokenization algorithm.
    - WordPiece: BERT, DistilBERT
    - Unigram: XLNet, ALBERT
    - Byte-Pair Encoding: GPT-2, RoBERTa

- These approaches help in reducing the vocabulary sizes by sharing information across different words having the ability to have prefixes and suffixes understood as such.
- They keep meaning across very similar words by recognizing similar tokens making them up.

## Tokenizer Pipeline
- A tokenizer takes texts as inputs and outputs numbers the associated model can make sense of

``` py
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
sequence = "Let's try to tokenize!"
inputs = tokenizer(sequence)
print(inputs)
#[101, 2292, 1005, 1055, 3046, 2000, 19204, 4697, 999, 102]
```

1. Tokenization pipeline: from input text to a list of numbers:

    1. Raw text: Let's try to tokenize!
    2. Tokens: [let,',s, try, to, token, ##ize,!]
    3. Special tokens: [[CLS],let,',s, try, to, token, ##ize,!,[SEP]]
    4. Input IDs: [101, 2292, 1005, 1055, 3046, 2000, 19204, 4697, 999, 102]







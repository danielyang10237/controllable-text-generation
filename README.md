# Controllable Text Generation: Word Length Complexity Manipulation

This project implements a controllable text generation system that can either **complicate** or **simplify** sentences by manipulating word length. The system uses fine-tuned GPT-2 and GPT-J models to generate text with controlled complexity based on word length preferences. Purpose is to deepen my knowledge in natural language processing

## Project Overview

The system works by:
1. **Training** utilizing existing language models with custom loss functions that incentivize longer or shorter words
2. **Generating** a series of output text with controllable complexity through specialized sampling strategies
3. **Evaluating** the generated text using multiple metrics including ROUGE, BERTScore, and word length analysis

## Key Features

- **Dual Complexity Control**: Can generate both more complex (longer words) and simpler (shorter words) versions of input text
- **Custom Loss Functions**: Implements word-length-aware training objectives
- **Multiple Model Support**: Works with both GPT-2 and GPT-J architectures
- **Comprehensive Evaluation**: Uses ROUGE, BERTScore, and word complexity metrics
- **Data Preprocessing**: Includes tools for creating training data from Wikipedia and paraphrase datasets

## Project Structure

```
├── model.py              # GPT-2 model training with custom loss functions
├── gptj.py               # GPT-J model implementation
├── generate.py           # Text generation with complexity control
├── evaluate.py           # Evaluation metrics (ROUGE, BERTScore, word length)
├── dataparser.py         # Wikipedia data preprocessing
├── paraphraseparser.py   # MSR Paraphrase dataset parsing
├── datasets/             # Training and evaluation data
│   ├── wikisent2.txt     # Wikipedia sentences
│   ├── scrambled.txt     # Preprocessed scrambled sentences
│   └── target.txt        # Target sentences
├── gen-outputs/          # Generated text outputs
└── paraphrased-text/     # Paraphrase evaluation data
```

**note that datasets and gen-outputs are not included in this repo, but datasets are taken on Wikipedia training corpus**

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Create a new conda environment (recommended)
conda create -n controllable-text python=3.9
conda activate controllable-text

# Or create a virtual environment with venv
python -m venv controllable-text-env
source controllable-text-env/bin/activate  # On Windows: controllable-text-env\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install manually
pip install torch torchvision torchaudio
pip install transformers
pip install nltk
pip install rouge-score
pip install bert-score
pip install tqdm
pip install numpy
```

### 3. Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage

### Training Models

1. **Train GPT-2 with complexity control:**
```bash
python model.py
```

2. **Train GPT-J model:**
```bash
python gptj.py
```

### Generating Text

```bash
python generate.py
```

The generation script supports different modes:
- `default_gen=True`: Standard generation
- `make_longer_gen=True`: Generate with longer words (more complex)
- `make_longer_gen=False`: Generate with shorter words (simpler)

### Evaluating Results

```bash
python evaluate.py
```

This will compute:
- **ROUGE scores** (ROUGE-1, ROUGE-2, ROUGE-L)
- **BERTScore** (Precision, Recall, F1)
- **Word length analysis** (average character count per word)
- **Vocabulary complexity** (ratio of uncommon words)

### Custom Loss Function

The system implements a word-length-aware loss function as the core mechanism for finetuning:

```python
def custom_loss_function(outputs, labels, tokenizer, incentive_threshold=4, penalty=0.01, make_longer_train=True):
    # Standard cross-entropy loss
    loss = outputs.loss
    
    # Word length reward/penalty
    if make_longer_train:
        # Reward longer words
        total_reward *= (math.sqrt(total_long_words) / math.sqrt(len(predicted_words)))
    else:
        # Penalize longer words
        total_reward -= penalty * len(word) * 0.1
    
    return loss * total_reward
```

## Data Processing

### Wikipedia Data (`dataparser.py`)
- Processes Wikipedia sentences
- Removes stopwords
- Applies word shuffling and synonym replacement
- Creates scrambled input and target sentence pairs

### Paraphrase Data (`paraphraseparser.py`)
- Parses MSR Paraphrase dataset
- Extracts high-quality paraphrase pairs
- Formats data for evaluation

## Evaluation Metrics

1. **ROUGE Scores**: Measures overlap between generated and reference text
2. **BERTScore**: Semantic similarity using BERT embeddings
3. **Word Length Analysis**: Average character count per word
4. **Vocabulary Complexity**: Ratio of uncommon vs. common words

## Dependencies

See `requirements.txt` for complete dependency list. Key packages:
- PyTorch
- Transformers (Hugging Face)
- NLTK
- rouge-score
- bert-score
- tqdm
- numpy

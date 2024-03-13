from rouge_score import rouge_scorer
from bert_score import score

# THE FOLLOWING FUNCTIONS ARE USED TO EVALUATE THE ACCURACY OF THE GENERATED TEXT
###########################################
# scoring based off rouge score
generated_dest = 'paraphrased-text/prompts.txt'
target_dest = 'paraphrased-text/generated.txt'

# rouge score evaluations
def get_all_rouge(generated_dest, target_dest):
    with open(generated_dest, 'r') as gen_file:
        generated = gen_file.readlines()
    with open(target_dest, 'r') as target_file:
        target = target_file.readlines()

    assert len(generated) == len(target), "not matching lengths for generated and target files"

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    scores = { "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0 }

    for gen_line, targ_line in zip(generated, target):
        score = scorer.score(gen_line, targ_line)
        scores['rouge1'] += score['rouge1'].fmeasure
        scores['rouge2'] += score['rouge2'].fmeasure
        scores['rougeL'] += score['rougeL'].fmeasure
    
    total_lines = len(generated)
    scores['rouge1'] /= total_lines
    scores['rouge2'] /= total_lines
    scores['rougeL'] /= total_lines

    print(scores)

    return scores

get_all_rouge(generated_dest, target_dest)
###########################################

###########################################
# bert score evaluation
generated_dest = 'paraphrased-text/prompts.txt'
target_dest = 'paraphrased-text/generated.txt'

def get_bert_score(generated_dest, target_dest):
    with open(generated_dest, 'r') as gen_file:
        generated = gen_file.readlines()
    with open(target_dest, 'r') as target_file:
        target = target_file.readlines()

    P, R, F1 = score(generated, target, lang='en', verbose=True)

    print(f"Precision: {P.mean()}")
    print(f"Recall: {R.mean()}")
    print(f"F1: {F1.mean()}")

    return P, R, F1

get_bert_score(generated_dest, target_dest)
###########################################

# THE FOLLOWING FUNCTIONS ARE USED TO EVALUATE THE COMPLEXITY OF THE GENERATED TEXT
###########################################
# scoring based off average word length
prompts_dest = 'gen-outputs/prompts.txt'
generated_dest = 'gen-outputs/prompts.txt'

def get_letter_count(prompts_dest, generated_dest):
    average_word_len_prompts = 0.0
    average_word_len_generated = 0.0

    with open(prompts_dest, 'r') as prompts, open(generated_dest, 'w') as generated:
        for prompt in prompts:
            prompt = prompt.strip()
            prompt_words = prompt.split()
            for word in prompt_words:
                average_word_len_prompts += len(word)
        
        average_word_len_prompts /= len(prompt_words)
        
        for generated_sentence in generated:
            generated_sentence = generated_sentence.strip()
            generated_words = generated_sentence.split()
            for word in generated_words:
                average_word_len_generated += len(word)

        average_word_len_generated /= len(generated_words)
    
    print(f"average word character length of prompts: {average_word_len_prompts}")
    print(f"average word character length of generated: {average_word_len_generated}")

get_letter_count(prompts_dest, generated_dest)
###########################################

###########################################
# scoring based off how frequently words are used
prompt_dest = 'gen-outputs/prompts.txt'
generated_dest = 'gen-outputs/generated.txt'

def get_comm1(prompt, generated):
    common_words = set()

    with open("eval-resources/common.txt", "r") as f:
        for line in f:
            common_words.add(line.strip())

    unique_words = 0.0
    total_words = 0.0
    
    with open(prompt, "r") as f:
        for line in f:
            words = line.strip().split()
            for word in words:
                total_words += 1
                if word not in common_words:
                    unique_words += 1

    print(f"unique words ratio in prompts: {unique_words / total_words}")

    unique_words2 = 0.0
    total_words2 = 0.0
    
    with open(generated, "r") as f:
        for line in f:
            words = line.strip().split()
            for word in words:
                total_words2 += 1
                if word not in common_words:
                    unique_words2 += 1
    
    print(f"unique words ratio in generated: {unique_words2 / total_words2}")

    return unique_words / total_words, unique_words2 / total_words2

get_comm1(prompt_dest, generated_dest)
###########################################


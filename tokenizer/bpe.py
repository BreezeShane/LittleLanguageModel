"""
BPE demo developed from scratch.
If wanna use reliable BPE, use [subword-nmt](https://github.com/rsennrich/subword-nmt).
Run `pip install subword-nmt` to install.
"""

import re
import json
import collections
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# import jsonlines
# def load_data(path: str):
#     """ Load data from the file path. """
#     corpus_text = ""
#     with jsonlines.open(path, "r") as f:
#         for data in f:
#             corpus_text += data["text"]
#     return corpus_text

# def build_vocabulary(corpus: str):
#     """ Build vocabulary from the corpus text. """
#     words = corpus.split(" ")
#     word_table = collections.Counter(words)
#     return word_table

def get_vocab(filename):
    """ Generate char table to build vocabulary. """
    __vocab = collections.defaultdict(int)
    with open(filename, 'r', encoding='utf-8') as fhand:
        for line in fhand:
            words = line.strip().split()
            for word in words:
                __vocab[' '.join(list(word)) + ' </w>'] += 1
    return __vocab

def get_stats(__vocab):
    """ Compute the frequency of char pairs. """
    __pairs = collections.defaultdict(int)
    for word, freq in __vocab.items():
        symbols = word.split()
        for __index in range(len(symbols)-1):
            __pairs[symbols[__index],symbols[__index+1]] += freq
    return __pairs

def merge_vocab(pair, v_in):
    """ Merge the vocabulary. """
    v_out = {}
    bigram = re.escape(' '.join(pair))
    escaped_text = re.escape(''.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(escaped_text, word)
        v_out[w_out] = v_in[word]
    return v_out

def get_tokens_from_vocab(__vocab):
    """ Get the frequencies of tokens merged from char pairs. """
    __tokens_frequencies = collections.defaultdict(int)
    __vocab_tokenization = {}
    for word, freq in __vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            __tokens_frequencies[token] += freq
        __vocab_tokenization[''.join(word_tokens)] = word_tokens
    return __tokens_frequencies, __vocab_tokenization

def measure_token_length(token):
    """ Mesure how long the token is, especially `</w>` is regarded as long as one char. """
    if token[-4:] == '</w>':
        return len(token[:-4]) + 1
    return len(token)

def tokenize_word(string, sorted_tokens, unknown_token='</u>'):
    """ Tokenize/Encode the word to ids stored in tokenizer. """
    if string == '':
        return []
    if sorted_tokens == []:
        return [unknown_token]

    string_tokens = []
    for ind, token in enumerate(sorted_tokens):
        token_reg = re.escape(token.replace('.', '[.]'))

        matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]
        if len(matched_positions) == 0:
            continue
        substring_end_positions = [matched_position[0] for matched_position in matched_positions]

        substring_start_position = 0
        for substring_end_position in substring_end_positions:
            substring = string[substring_start_position:substring_end_position]
            string_tokens += tokenize_word(
                string=substring, sorted_tokens=sorted_tokens[ind+1:], unknown_token=unknown_token)
            string_tokens += [token]
            substring_start_position = substring_end_position + len(token)
        remaining_substring = string[substring_start_position:]
        string_tokens += tokenize_word(
            string=remaining_substring,
            sorted_tokens=sorted_tokens[ind+1:],
            unknown_token=unknown_token
        )
        break
    return string_tokens

def save_bpe_vocab(save_path: str, __vocab: dict):
    """ Save BPE token vocabulary by json. """
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj=__vocab, fp=f)

def load_bpe_vocab(load_path: str):
    """ Load trained BPE vocabulary from json file. """
    with open(load_path, "r", encoding="utf-8") as f:
        result = json.load(f)
        return result

if __name__ == "__main__":
    X_AXIS, Y_AXIS = [], []
    fig, (ax_count, ax_prob) = plt.subplots(2)
    # ax_count = plt.subplot2grid((2, 2), (0, 0))
    # ax_prob = plt.subplot2grid((2, 2), (0, 1))
    plt.ion()

    vocab = get_vocab("/home/BreezeShane/PersonalProjects/lil_language_model/corpus/corpus.txt")

    NUM_MERGES = 50000
    for i in range(NUM_MERGES+1):
        tokens_frequencies, vocab_tokenization = get_tokens_from_vocab(vocab)

        X_AXIS.append(i)
        Y_AXIS.append(len(tokens_frequencies.keys()))

        ax_count.clear()
        ax_prob.clear()

        ax_count.set_title("The Number of Tokens")
        ax_count.set_xlabel("Iteration")
        ax_count.set_ylabel("Count Number")

        ax_count.plot(X_AXIS, Y_AXIS)
        log_probs = np.log10([
            1.0 * x / sum(tokens_frequencies.values()) + 1e-10
            for x in tokens_frequencies.values()
        ])
        kde = gaussian_kde(log_probs)
        x_vals = np.linspace(log_probs.min() - 1, log_probs.max() + 1, 1000)
        y_vals = kde(x_vals)
        ax_prob.plot(x_vals, y_vals, color='tab:red', linewidth=2.5, label='Probability Density')
        ax_prob.fill_between(x_vals, y_vals, color='tab:red', alpha=0.1)
        ax_prob.set_xlabel("log(Probability)", color='tab:red', fontsize=12)
        ax_prob.set_ylabel('Probability Density', color='tab:red', fontsize=12)
        ax_prob.tick_params(axis='y', labelcolor='tab:red')

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)

        if i % 1000 == 0:
            save_bpe_vocab(
                "/home/BreezeShane/PersonalProjects/lil_language_model/"+
                f"tokenizer/trained/bpe_{i}.json", vocab)

        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        print(f'Iter: {i} - Best pair: {best}')

    plt.ioff()
    plt.show()
    save_bpe_vocab(
        "/home/BreezeShane/PersonalProjects/lil_language_model/"+
        "tokenizer/trained/bpe_fin.json", vocab)

    # # Let's check how tokenization will be for a known word
    # word_given_known = 'newest</w>lower</w>'
    # word_given_unknown = 'Ilikeeatingapples!</w>'

    # sorted_tokens_tuple = sorted(
    #     tokens_frequencies.items(),
    #     key=lambda item: (measure_token_length(item[0]), item[1]), reverse=True)
    # sorted_tokens = [token for (token, freq) in sorted_tokens_tuple] # 用来解码

    # print(sorted_tokens)

    # word_given = word_given_known

    # print('Tokenizing word: {}...'.format(word_given))
    # if word_given in vocab_tokenization:
    #     print('Tokenization of the known word:')
    #     print(vocab_tokenization[word_given])
    #     print('Tokenization treating the known word as unknown:')
    #     print(tokenize_word(string=word_given, sorted_tokens=sorted_tokens, unknown_token='</u>'))
    # else:
    #     print('Tokenizating of the unknown word:')
    #     print(tokenize_word(
    #         string=word_given,
    #         sorted_tokens=sorted_tokens,
    #         unknown_token='</u>'
    #     )) # ['newest</w>', 'lower</w>']

    # word_given = word_given_unknown

    # print('Tokenizing word: {}...'.format(word_given))
    # if word_given in vocab_tokenization:
    #     print('Tokenization of the known word:')
    #     print(vocab_tokenization[word_given])
    #     print('Tokenization treating the known word as unknown:')
    #     print(tokenize_word(string=word_given, sorted_tokens=sorted_tokens, unknown_token='</u>'))
    # else:
    #     print('Tokenizating of the unknown word:')
    #     print(tokenize_word(string=word_given, sorted_tokens=sorted_tokens, unknown_token='</u>'))

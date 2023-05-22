from collections import defaultdict
from typing import Dict, Tuple, List
from operator import itemgetter

class BytePairEncoder:
    def __init__(self):
        pass

    def _get_pair_counts(
        self,
        word_count: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, str], int]:
        """
          Input:
            - word_count: a dictionary where the keys are all words (broken into
                          a tuple of characters) in the corpus and the values
                          are the counts

          Output:
            - pair_count: a dictionary where the keys are all pairs of
                          consecutive characters and the values are the counts
        """

        pair_count = defaultdict(int)
        for chars, count in word_count.items():

            # count occurrences of all consecutive pairs and add them to ``pair_count``
            for i in range(len(chars) - 1):
                pair_count[(chars[i], chars[i + 1])] = pair_count.get((chars[i], chars[i + 1]), 0) + count

        return pair_count

    def _merge_pair(
        self,
        word_count: Dict[Tuple[str, ...], int],
        pair: Tuple[str, str]
    ) -> Dict[Tuple[str, ...], int]:
        """
          Input:
            - word_count: a dictionary where the keys are all words (broken into
                          a tuple of characters) in the corpus and the values
                          are the counts
            - pair: a pair of characters to be merged

          Output:
            - word_count_new: updated dictionary according to the given pair to
                              to be merged
        """

        word_count_new = defaultdict(int)
        for chars, count in word_count.items():
            chars_new = []
            i = 0

            # merge all occurrences of ``pair`` in ``chars`` to get ``chars_new`` and add it to ``word_count_new``
            while i < len(chars):
                if i + 1 >= len(chars):
                    chars_new.append(chars[i])
                    break
                if (chars[i], chars[i + 1]) == pair:
                    chars_new.append(chars[i] + chars[i + 1])
                    i += 2
                else:
                    chars_new.append(chars[i])
                    i += 1

            word_count_new[tuple(chars_new)] = count

        return word_count_new

    def train(
        self,
        corpus: str,
        num_merges: int = 5
    ) -> Dict[str, int]:
        """
          TODO [part 1f]

          Input:
            - corpus: a string of text for training the BPE encoding
            - num_merges: number of new vocabularies obtained from the corpus via training

          Output:
            - self.vocabs: a dictionary of vocabularies obtained from the corpus via training
        """

        # step 1: normalization
        corpus = corpus.lower()
        corpus = corpus.replace(',', '')
        corpus = corpus.replace('!', '')
        corpus = corpus.replace(' ', '_')

        # step 2: pre-tokenization
        word_list = corpus.split('_')
        word_list = [w+'_' for w in word_list]

        # step 3: learning from the word list
        word_count = defaultdict(int)
        for w in word_list:
            word_count[tuple([*w])] += 1

        print('='*100)
        print('Your BPE learning')
        print('='*100)

        self.vocab = {c: i for i, c in enumerate(set(corpus))}
        n = len(self.vocab)
        for i in range(num_merges):
            # count the number of occurrences of each pair of characters
            pair_count = self._get_pair_counts(word_count)
            if not pair_count:
                break

            # select the most frequent pair in pair_count
            most_frequent_pair = max(pair_count, key = pair_count.get)

            # add the most frequent pair to the vocabulary
            self.vocab[most_frequent_pair] = n + i

            # print info
            print('iteration:', i)
            print('vocabulary: ', self.vocab)
            print('most frequent pair:', most_frequent_pair)
            print('')

            # merge the most frequent pair
            word_count = self._merge_pair(word_count, most_frequent_pair)

        print('final vocabulary: ', self.vocab)
        print('')

        return self.vocab

    def _get_pairs(
        self,
        text: List[str]
    ) -> List[Tuple[str, str]]:
        """
          Input:
            - text: a list of strings for BPE encoding

          Output:
            - pairs: a list of consecutive pairs of characters
        """

        pairs = []
        prev_char = text[0]
        for char in text[1:]:
            # recursively add a tuple of consecutive characters to ``pairs``
            pairs.append((prev_char, char))
            prev_char = char

        return pairs

    def _merge_pair_for_text(
        self,
        text: List[str],
        pair_to_merge: Tuple[str, str]
    ) -> List[str]:
        """
          Input:
            - text: a list of strings
            - pair_to_merge: a tuple of characters to be merged

          Output:
            - new_text: a new list of strings where the given pair of characters
                        is merged
        """

        new_text = []

        # merge all occurrences of ``pair_to_merge`` in ``text`` to get ``new_text`` and return it
        i = 0
        while i < len(text):
            if i + 1 >= len(text):
                new_text.append(text[i])
                break
            if (text[i], text[i + 1]) == pair_to_merge:
                new_text.append(text[i] + text[i + 1])
                i += 2
            else:
                new_text.append(text[i])
                i += 1

        return new_text

    def encode(
        self,
        raw_text: str
    ) -> Tuple[List[str], List[int]]:
        """

          [part 1h]

          Input: 
            - raw_text: given text in string
            - pair_to_merge: a tuple of characters to be merged

          Output:
            - text: a list of strings where each string is vocabulary
            - encoding: a list of integer ids of the strings in text
        """

        # step 1: normalization
        raw_text = raw_text.lower()
        raw_text = raw_text.replace(' ', '_')

        # step 2: pre-tokenization
        text = list(raw_text)
        text.append('_')

        # step 3: encode the given text
        print('='*100)
        print('Your BPE encoding')
        print('='*100)

        i = 0
        while True:
            # get all pairs of charactors
            pairs = self._get_pairs(text)
            vocab_pairs = [(pair, self.vocab[pair]) for pair in pairs if pair in self.vocab]
            if not vocab_pairs:
                break

            # select the next pair for merging
            pair_to_merge = min(vocab_pairs, key=itemgetter(1))[0]
            text = self._merge_pair_for_text(text, pair_to_merge)

            # print info
            print('iteration:', i)
            print('pair to merge:', pair_to_merge)
            print('text: ', text)
            print('')

            i += 1

        # get a list of indices for the tokens
        vocab_short = {}
        for key, value in self.vocab.items():
            if isinstance(key, tuple):
                vocab_short[str(key[0])+str(key[1])] = value
            else:
                vocab_short[key] = value
        encoding = [vocab_short[token] for token in text]

        return text, encoding

from collections import defaultdict
from email.policy import default
from typing import List, NamedTuple

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        self.EMPTY_TOK = "^"
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        raw = ''.join([self.ind2char[ind] for ind in inds])
        correct = ""
        prev = None
        for symb in raw:
            if symb != '^':
                if prev is not None:
                    if prev != symb:
                        correct += symb
                        prev = symb
                else:
                    prev = symb
                    correct += symb
        # print(correct)
        return correct



    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
                        # beam size надо взять меньше иначе это 100 лет.
                        
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []
    
        # TODO: your code here
        dp = {('', self.EMPTY_TOK): 1.0}

        for prob in probs:
            dp = self.extend_and_merge(dp, prob, self.ind2char)
            dp = self.cut_beams(dp, beam_size)

        list_probs =  list(sorted([((res + last_char).strip().replace(self.EMPTY_TOK, ''),  proba) for (res, last_char), proba in dp.items()], key=lambda x: -x[1]))
        df = dict()
        for (sent, prob) in list_probs:
            if sent not in df:
                df[sent] = prob
            else:
                df[sent] += prob
        # print(df)
        return [(k, df[k]) for k in df]


    def extend_and_merge(self, dp, proba, ind2char):
        new_dp = defaultdict(float)
        for (res, last_char), v in dp.items():
            for i in range(len(proba)):
                if ind2char[i] == last_char:
                    new_dp[(res, last_char)] += v * proba[i]
                else:
                    new_dp[((res+last_char).replace(self.EMPTY_TOK, ''), ind2char[i])] += v * proba[i]
        return new_dp

    def cut_beams(self, dp: dict, beam_size: int) -> dict:
        return dict(list(sorted(dp.items(), key=lambda x: x[1]))[-beam_size:])
        # return sorted(hypos, key=lambda x: x.prob, reverse=True)

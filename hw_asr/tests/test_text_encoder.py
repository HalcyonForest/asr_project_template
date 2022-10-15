import unittest

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder

import torch 

class TestTextEncoder(unittest.TestCase):
    def test_ctc_decode(self):
        
        text = "i^^ ^w^i^sss^hhh^   i ^^^s^t^aaaar^teee^d " \
               "dddddd^oooo^in^g tttttttth^iiiis h^^^^^^^^w^ e^a^r^li^er"
        true_text = "i wish i started doing this hw earlier"
        text_encoder = CTCCharTextEncoder(list(text))
        inds = [text_encoder.char2ind[c] for c in text]
        decoded_text = text_encoder.ctc_decode(inds)
        print("Decoded CTC text: ", decoded_text)
        self.assertIn(decoded_text, true_text)

    def test_beam_search(self):
        # TODO: (optional) write tests for beam search
        # TODO добавить еще 1-2 тестика для красоты и полноты.
        hypothesises = ["^a","a^","ab", "aa"]
        probs = torch.tensor([0.3, 0.20, 0.34, 0.16])

        probs = torch.tensor([
            [0.4,0.5, 0.1],
            [0.65,0.15, 0.2],
            [0.1, 0.8, 0.1]
        ])
        text_encoder = CTCCharTextEncoder(['a', 'b'])
        true_text = "a"
        # inds = [text_encoder.char2ind[c] for c in hypothesises]
        decoded_text = text_encoder.ctc_beam_search(probs, probs.shape[0], 10) #??? )
        print(decoded_text)
        decoded_text = sorted(decoded_text, key=lambda x: x[1], reverse=True)[0][0]
        print("Decoced BeamSearch text: ", decoded_text)

        self.assertIn(decoded_text, true_text)
        # return

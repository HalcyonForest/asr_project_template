import logging
from typing import List
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    # print(dataset_items)
    max_audio = 0
    max_spectro = 0 
    max_texte = 0
    max_text=0


    audios = torch.tensor([])
    specs = torch.tensor([])
    texts_e = torch.tensor([])
    audio_paths = []
    text_encoded_lens = []
    spectrogram_lens = []
    texts = []
    # print(dataset_items[0]['text'].dtype)
    for item in dataset_items:
        audio_paths.append(item['audio_path'])
        # for i in item:
        #     print(i)
        max_audio = max(max_audio, item['audio'].shape[1])
        max_spectro = max(max_spectro, item['spectrogram'].shape[2])
        max_texte = max(max_texte, item['text_encoded'].shape[1])
        max_text = max(max_text, len(item['text']))
    
    for item in dataset_items:
        audio = item['audio']
        audio_shape = audio.shape

        spectrogram = item['spectrogram']
        spectrogram_shape = spectrogram.shape

        text_encoded = item['text_encoded']
        text_encoded_shape = text_encoded.shape

        text_encoded_lens.append(len(text_encoded))
        spectrogram_lens.append(len(spectrogram))

        #   item['text'] += "^" * (max_text - len(item['text']))
        # texts_e.append(item['text_encoded'])
        texts.append(item['text'])
        # if texts_e is not None:
        #     texts_e = torch.cat([texts_e, torch.tensor(item['text_encoded'])], dim=0)
        # else:
        #     texts_e = torch.tensor(item['text_encoded'])
        
        

        audios = torch.cat([audios, (torch.cat((audio, torch.zeros((audio_shape[0],max_audio - audio_shape[1]))), dim=1))], dim=0)
        specs = torch.cat([specs, (torch.cat((spectrogram, torch.zeros((spectrogram_shape[0], spectrogram_shape[1], max_spectro - spectrogram_shape[2]))), dim=2))], dim=0)

        texts_e = torch.cat([texts_e, (torch.cat((text_encoded, torch.zeros((text_encoded_shape[0], max_texte - text_encoded_shape[1]))), dim=1))], dim=0)

        # if text_encoded_lens is not None:
        #     text_encoded_lens = torch.cat([text_encoded_lens, (torch.tensor([text_encoded_shape[1]]))], dim=0)
        # else:
        #     text_encoded_lens = torch.tensor([text_encoded_shape[1]])
    # print(audios.shape, specs.shape, texts_e.shape, text_encoded_lens.shape, len(texts))




        # print(item['audio'].shape, item['spectrogram'].shape, item['text_encoded'].shape)

    #keys: audio, spectogram, duration, text, text_encoded, audio_path
    # Падим аудио? шпектограму? текст_энкодед? 
    
    # print(dataset_items[0])
    # print("TEXT ENCODED SHAPE: ", texts_e.shape)
    result_batch = {
        'audio_path':audio_paths,
        'audio': audios,
        'spectrogram': specs,
        'spectrogram_length':torch.tensor(spectrogram_lens),
        'text_encoded': texts_e,
        'text_encoded_length':torch.tensor(text_encoded_lens),
        'text': texts
    }

    return result_batch
    
    # raise NotImplementedError

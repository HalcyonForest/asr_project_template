# Don't forget to support cases when target_text == ''
import editdistance

def calc_cer(target_text, predicted_text) -> float:
    return editdistance.distance(target_text, predicted_text) / len(target_text)
    

def calc_wer(target_text, predicted_text) -> float:
    splitted_target = target_text.split(' ')
    splitted_predict = predicted_text.split(' ')
    return editdistance.distance(splitted_target, splitted_predict) / len(splitted_target)
    

    

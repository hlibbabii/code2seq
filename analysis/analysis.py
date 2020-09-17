from collections import Counter
from dataclasses import dataclass
from typing import List, Set, Tuple, Dict

from pathlib import Path
from pprint import pprint

PROJECT_DIR = Path(__file__).parent.parent

FILE=PROJECT_DIR/'data/java-small/java-small.test.c2s'
PREDICTIONS=PROJECT_DIR/'models/java-large-model/pred.txt'
REFERENCE=PROJECT_DIR/'models/java-large-model/ref.txt'


@dataclass
class MethodVocab(object):
    method_name_subwords: List[str]
    identifiers_subwords: Set[Tuple[str]]

    def normalize_body_subwords(self):
        return {subword for identifier in self.identifiers_subwords for subword in identifier}

    def get_new_name_subwords(self):
        normalized_subwords = self.normalize_body_subwords()
        s = {subword for subword in self.method_name_subwords if subword not in normalized_subwords}
        return s


def extract(s: str):
    identifiers_subwords: Set[Tuple[str]] = set()
    split_string: List[str] = s.split(' ')
    method_name: str = split_string[0]
    paths: List[str] = split_string[1:]
    method_name_subwords = method_name.split('|')
    for path in paths:
        terminal1, _, terminal2 = path.split(',')
        identifiers_subwords.add(tuple(terminal1.split('|')))
        identifiers_subwords.add(tuple(terminal2.split('|')))
    return MethodVocab(method_name_subwords, identifiers_subwords)


def extract_methods_and_predicitions() -> List[Tuple[MethodVocab, str]]:
    methodVocabs: List[Tuple[MethodVocab, str]] = []
    with open(FILE, 'r') as bodies, open(PREDICTIONS, 'r') as preds, open(REFERENCE, 'r') as refs:
        for body_line, pred, ref in zip(bodies, preds, refs):
            extracted = extract(body_line.rstrip(' \n'))
            assert " ".join(extracted.method_name_subwords) == ref.rstrip(' \n')
            methodVocabs.append((extracted, pred.rstrip(' \n')))
    return methodVocabs


def combine_dicts(dict1: Dict[str, int], dict2: Dict[str, int]) -> Dict[str, Tuple[int, int]]:
    combined: Dict[str, Tuple[int, int]] = {}
    all_keys = set(dict1.keys())
    all_keys.update(set(dict2.keys()))
    for key in all_keys:
        value1 = dict1[key] if key in dict1 else 0
        value2 = dict2[key] if key in dict2 else 0
        combined[key] = (value1, value2)
    return combined


def calc_stats(methods_and_predicitions: List[Tuple[MethodVocab, str]]) -> Tuple[List[Tuple[str, Tuple[int, int]]], int, int]:
    correct_predictions_vocab = Counter()
    incorrect_predictions_vocab = Counter()
    for method, prediction in methods_and_predicitions:
        new_name_subwords = method.get_new_name_subwords()
        if new_name_subwords:
            predicted_sub_tokens = prediction.split(' ')
            non_predicted_words = set(new_name_subwords).difference(predicted_sub_tokens)
            predicted_words = set(new_name_subwords).intersection(predicted_sub_tokens)

            correct_predictions_vocab.update(predicted_words)
            incorrect_predictions_vocab.update(non_predicted_words)

    combined: Dict[str, Tuple[int, int]] = combine_dicts(correct_predictions_vocab, incorrect_predictions_vocab)
    combined_sorted: List[Tuple[str, Tuple[int, int]]] = sorted(combined.items(), key=lambda x: float(x[1][1]+1) / (x[1][0]+x[1][1]+2))
    return combined_sorted


if __name__ == '__main__':
    methods_and_predicitions = extract_methods_and_predicitions()
    combined_sorted = calc_stats(methods_and_predicitions)
    # print(f'Correctly predicted names where at least one word had to be invented: {new_correct}')
    # print(f'INcorrectly predicted names where at least one word had to be invented: {new_incorrect}')
    n=300
    pprint(f'{n} most easily invented words: \n')
    pprint(combined_sorted[:n])
    pprint(f'{n} most difficultly invented words: \n')
    pprint(combined_sorted[-n:])

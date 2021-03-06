from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass
from typing import List, Set, Tuple, Dict, Iterable

from pathlib import Path
from pprint import pprint
import csv

PROJECT_DIR = Path(__file__).parent.parent

FILE=PROJECT_DIR/'data/java-small/java-small.test.c2s'
PREDICTIONS=PROJECT_DIR/'models/java-large-model/pred.txt'
REFERENCE=PROJECT_DIR/'models/java-large-model/ref.txt'


@dataclass
class MethodVocab(object):
    method_name_subwords: List[str]
    identifiers_subwords: Set[Tuple[str]]

    def normalize_body_subwords(self) -> Set[str]:
        return {subword for identifier in self.identifiers_subwords for subword in identifier}

    def get_new_name_subwords(self) -> Set[str]:
        return self.get_invented_subwords_from_list(self.method_name_subwords)

    def get_invented_subwords_from_list(self, lst: Iterable[str]) -> Set[str]:
        normalized_subwords = self.normalize_body_subwords()
        s = {subword for subword in lst if subword not in normalized_subwords}
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


def classify_methods(methodVocabAndPredList: List[Tuple[MethodVocab, str]]) -> Dict[str, List[Tuple[MethodVocab, str]]]:
    vocab = {
        'setgetters_2_words': [],
        'setgetters_3_words': [],
        'setgetters_more_words': [],
        'handlers': [],
        'main': [],
        '1words_0invented': [],
        '1words_1invented': [],
        '2words_0invented': [],
        '2words_1invented': [],
        '2words_2invented': [],
        '3words_0invented': [],
        '3words_1invented': [],
        '3words_2invented': [],
        '3words_3invented': [],
        '4+words_0invented': [],
        '4+words_1invented': [],
        '4+words_2invented': [],
        '4+words_3invented': [],
        '4+words_4+invented': [],
    }
    for methodVocab, pred in methodVocabAndPredList:
        name_subwords = methodVocab.method_name_subwords
        new_name_subwords = methodVocab.get_new_name_subwords()
        is_setter = name_subwords[0] in ['get', 'set', 'is']
        if is_setter and len(name_subwords) == 2:
            vocab['setgetters_2_words'].append((methodVocab, pred))
        elif is_setter and len(name_subwords) == 3:
            vocab['setgetters_3_words'].append((methodVocab, pred))
        elif is_setter and len(name_subwords) > 3:
            vocab['setgetters_more_words'].append((methodVocab, pred))
        elif name_subwords[0] == 'on':
            vocab['handlers'].append((methodVocab, pred))
        elif name_subwords[0] == 'main' and len(name_subwords) == 1:
            vocab['main'].append((methodVocab, pred))
        elif len(new_name_subwords) == 0 and len(name_subwords) == 1:
            vocab['1words_0invented'].append((methodVocab, pred))
        elif len(new_name_subwords) == 1 and len(name_subwords) == 1:
            vocab['1words_1invented'].append((methodVocab, pred))
        elif len(new_name_subwords) == 0 and len(name_subwords) == 2:
            vocab['2words_0invented'].append((methodVocab, pred))
        elif len(new_name_subwords) == 1 and len(name_subwords) == 2:
            vocab['2words_1invented'].append((methodVocab, pred))
        elif len(new_name_subwords) == 2 and len(name_subwords) == 2:
            vocab['2words_2invented'].append((methodVocab, pred))
        elif len(new_name_subwords) == 0 and len(name_subwords) == 3:
            vocab['3words_0invented'].append((methodVocab, pred))
        elif len(new_name_subwords) == 1 and len(name_subwords) == 3:
            vocab['3words_1invented'].append((methodVocab, pred))
        elif len(new_name_subwords) == 2 and len(name_subwords) == 3:
            vocab['3words_2invented'].append((methodVocab, pred))
        elif len(new_name_subwords) == 3 and len(name_subwords) == 3:
            vocab['3words_3invented'].append((methodVocab, pred))
        elif len(new_name_subwords) == 0 and len(name_subwords) >= 4:
            vocab['4+words_0invented'].append((methodVocab, pred))
        elif len(new_name_subwords) == 1 and len(name_subwords) >= 4:
            vocab['4+words_1invented'].append((methodVocab, pred))
        elif len(new_name_subwords) == 2 and len(name_subwords) >= 4:
            vocab['4+words_2invented'].append((methodVocab, pred))
        elif len(new_name_subwords) == 3 and len(name_subwords) >= 4:
            vocab['4+words_3invented'].append((methodVocab, pred))
        elif len(new_name_subwords) >= 4 and len(name_subwords) >= 4:
            vocab['4+words_4+invented'].append((methodVocab, pred))
        else:
            raise AssertionError()
    return vocab


def combine_dicts(dict1: Dict[str, int], dict2: Dict[str, int]) -> Dict[str, Tuple[int, int]]:
    combined: Dict[str, Tuple[int, int]] = {}
    all_keys = set(dict1.keys())
    all_keys.update(set(dict2.keys()))
    for key in all_keys:
        value1 = dict1[key] if key in dict1 else 0
        value2 = dict2[key] if key in dict2 else 0
        combined[key] = (value1, value2)
    return combined


@dataclass(frozen=True)
class Stats:
    word: str
    first_subtoken: bool
    invented_subtoken: bool
    full_word_guessed: bool
    all_subtokens_mentioned: bool
    current_subtoken_mentioned: bool

    def __post_init__(self):
        if self.full_word_guessed:
            assert self.all_subtokens_mentioned
        if self.all_subtokens_mentioned:
            assert self.current_subtoken_mentioned

@dataclass
class OverGuessed:
    overinvented: int = 0
    overcopied: int = 0


def calc_stats(methods_and_predicitions: List[Tuple[MethodVocab, str]]) -> Tuple[Dict[Stats, int], Dict[str, OverGuessed]]:
    """
    >>> calc_stats([\
(MethodVocab(['get', 'name'], {('name',)}), 'get name'), \
(MethodVocab(['get', 'name'], {('name',)}), 'return name'), \
(MethodVocab(['invent', 'name'], {('name',)}), 'invent name')])
    ({Stats(word='get', first_subtoken=True, invented_subtoken=True, full_word_guessed=True, all_subtokens_mentioned=True, current_subtoken_mentioned=True): 1, Stats(word='name', first_subtoken=False, invented_subtoken=False, full_word_guessed=True, all_subtokens_mentioned=True, current_subtoken_mentioned=True): 2, Stats(word='get', first_subtoken=True, invented_subtoken=True, full_word_guessed=False, all_subtokens_mentioned=False, current_subtoken_mentioned=False): 1, Stats(word='name', first_subtoken=False, invented_subtoken=False, full_word_guessed=False, all_subtokens_mentioned=False, current_subtoken_mentioned=True): 1, Stats(word='invent', first_subtoken=True, invented_subtoken=True, full_word_guessed=True, all_subtokens_mentioned=True, current_subtoken_mentioned=True): 1}, {'return': OverGuessed(overinvented=1, overcopied=0)})
    """
    result: Dict[Stats, int] = defaultdict(int)
    over_guessed: Dict[str, OverGuessed] = defaultdict(OverGuessed)
    for method, prediction in methods_and_predicitions:
        new_name_subwords = method.get_new_name_subwords()
        predicted_sub_tokens = prediction.split(' ')
        predicted_sub_token_set = set(predicted_sub_tokens)
        identifier_guessed = (method.method_name_subwords == predicted_sub_tokens)
        all_subtokens_mentioned = identifier_guessed or not set(method.method_name_subwords).difference(predicted_sub_token_set)
        overguessed_tokens = predicted_sub_token_set.difference(method.method_name_subwords)
        overinvented_tokens = method.get_invented_subwords_from_list(overguessed_tokens)
        overcopied_tokens = overguessed_tokens.difference(overinvented_tokens)
        for o in overinvented_tokens:
            over_guessed[o].overinvented += 1
        for o in overcopied_tokens:
            over_guessed[o].overcopied += 1
        for i, subtoken in enumerate(method.method_name_subwords):
            result[Stats(subtoken, i == 0, subtoken in new_name_subwords, identifier_guessed, all_subtokens_mentioned, subtoken in predicted_sub_token_set)] += 1
    return dict(result), dict(over_guessed)


def get_correct_predictions(dct: Dict[Stats, int]) -> int:
    return sum([count for stats, count in dct.items() if stats.first_subtoken and stats.full_word_guessed])


def get_permuted_predictions(dct: Dict[Stats, int]) -> int:
    return sum([count for stats, count in dct.items() if stats.first_subtoken and not stats.full_word_guessed and stats.all_subtokens_mentioned])


def get_not_guessed_predictions(dct: Dict[Stats, int]) -> int:
    return sum([count for stats, count in dct.items() if stats.first_subtoken and not stats.all_subtokens_mentioned])


@dataclass
class InventedCopiedStats:
    _invented: int = 0
    _to_invent_total: int = 0
    _copied: int = 0
    _to_copy_total: int = 0
    overcopied: int = 0
    overinvented: int = 0

    def invented(self, count: int):
        self._invented += count
        self._to_invent_total += count

    def not_invented(self, count: int):
        self._to_invent_total += count

    def copied(self, count: int):
        self._copied += count
        self._to_copy_total += count

    def not_copied(self, count: int):
        self._to_copy_total += count

    def total_occured(self):
        return self._to_copy_total + self._to_invent_total

    def __repr__(self):
        return f'(invented:{self._invented}/{self._to_invent_total}, copied: {self._copied}/{self._to_copy_total}, over-invented/copied: {self.overinvented}--{self.overcopied})'


def get_subword_stats(dct: Dict[Stats, int], overinvented_copied: Dict[str, OverGuessed]) -> Dict[str, InventedCopiedStats]:
    """
    >>> get_subword_stats({Stats(word='get', first_subtoken=True, invented_subtoken=True, full_word_guessed=True, all_subtokens_mentioned=True, current_subtoken_mentioned=True): 100, Stats(word='name', first_subtoken=False, invented_subtoken=False, full_word_guessed=True, all_subtokens_mentioned=True, current_subtoken_mentioned=True): 2, Stats(word='get', first_subtoken=True, invented_subtoken=True, full_word_guessed=False, all_subtokens_mentioned=False, current_subtoken_mentioned=False): 1, Stats(word='name', first_subtoken=False, invented_subtoken=False, full_word_guessed=False, all_subtokens_mentioned=False, current_subtoken_mentioned=True): 1, Stats(word='invent', first_subtoken=True, invented_subtoken=True, full_word_guessed=True, all_subtokens_mentioned=True, current_subtoken_mentioned=True): 1})
    {'get': (invented:100/101, copied: 0/0), 'name': (invented:0/0, copied: 3/3), 'invent': (invented:1/1, copied: 0/0)}
    """
    invented_copied_stats = defaultdict(InventedCopiedStats)
    for stats, count in dct.items():
        if stats.invented_subtoken:
            if stats.current_subtoken_mentioned:
                invented_copied_stats[stats.word].invented(count)
            else:
                invented_copied_stats[stats.word].not_invented(count)
        else:
            if stats.current_subtoken_mentioned:
                invented_copied_stats[stats.word].copied(count)
            else:
                invented_copied_stats[stats.word].not_copied(count)
    for word, c in overinvented_copied.items():
        invented_copied_stats[word].overinvented = c.overinvented
        invented_copied_stats[word].overcopied = c.overcopied
    return dict(invented_copied_stats)


if __name__ == '__main__':
    all_methods_and_predicitions = extract_methods_and_predicitions()
    method_and_predictions_groups = classify_methods(all_methods_and_predicitions)
    analysis_ouput_dir = PROJECT_DIR / 'analysis-output'
    if not analysis_ouput_dir.exists():
        analysis_ouput_dir.mkdir()
    predictions_file = analysis_ouput_dir / 'predictions.csv'
    with open(predictions_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Group", "Correctly predicted", "Wrong order or with additional words", "Missing words"])

        for group, methods_and_predicitions in method_and_predictions_groups.items():
            stats, overinvented_copied = calc_stats(methods_and_predicitions)
            writer.writerow([group, get_correct_predictions(stats), get_permuted_predictions(stats), get_not_guessed_predictions(stats)])
    print(f"Predictions stats is saved to {predictions_file}")

    per_word_prediction_file = analysis_ouput_dir/'per-word-predictions.csv'
    with open(per_word_prediction_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Word", "Correctly invented", "To be invented",
                         "Copied", "To be copied", "Shouldn't have invented", "Shouldn't have copied"])
        stats, overinvented_copied = calc_stats(all_methods_and_predicitions)
        subword_stats: Dict[str, InventedCopiedStats] = get_subword_stats(stats, overinvented_copied)
        subword_stats_sorted: List[Tuple[str, InventedCopiedStats]] = sorted(subword_stats.items(), key=lambda x: x[1].total_occured(), reverse=True)

        for row in subword_stats_sorted:
            writer.writerow([row[0], row[1]._invented, row[1]._to_invent_total,
                             row[1]._copied, row[1]._to_copy_total, row[1].overinvented, row[1].overcopied])
        print(f"Per-word prediction stats is saved to {per_word_prediction_file}")

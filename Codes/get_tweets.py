import twint
import nest_asyncio
from tqdm import tqdm

import itertools
from argparse import ArgumentParser
from typing import List

plural_postfixes = ['leri', 'ları', 'lari']
singular_postfixes = ['ı', 'i', 'ü', 'u']

letter_postfix = {
    'a': 'sı',
    'e': 'si',
    'ı': 'sı',
    'i': 'si',
    'o': 'su',
    'ö': 'sü',
    'u': 'su',
    'ü': 'sü'
}

def read_texts(filepath: str) -> List[str]:
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    return [l.rstrip() for l in lines]


def singular_postfix(txt):
    vowels = set('aeıioöuü')
    last_vowel = next((l for l in reversed(txt) if l in vowels), None)
    if last_vowel:
        return letter_postfix[last_vowel][-1]
    else:
        return letter_postfix['e'][-1]


def get_queries(entities: List[str] = [], offensives: List[str] = []) -> List[str]:
    queries = []
    plural_combins = list(itertools.product(offensives, plural_postfixes))

    for e in entities:
        queries.extend([f'{bad_word} {e}{postfix}' for bad_word, postfix in plural_combins])
        if e[-1] in letter_postfix:
            queries.extend([f'{bad_word} {e}{letter_postfix[e[-1]]}' for bad_word in offensives])
        else:
            queries.extend([f'{bad_word} {e}{singular_postfix(e)}' for bad_word in offensives])
    
    return queries

if __name__ == '__main__':
    parser = ArgumentParser(description='Retrieves tweets with the entities as direct objects and the offensives in them.')

    parser.add_argument('--entities', default='entities.txt', help='Path to the text file containing entities seperated by newline', type=str)
    parser.add_argument('--offensives', default='offensives.txt', help='Path to the text file containing offensive texts seperated by newline', type=str)
    parser.add_argument('--output', default='tweets.csv', help='Path to save tweets', type=str)
    parser.add_argument('--tweetslimit', default=10, help='Maximum number of tweets to retrieve', type=int)

    args = parser.parse_args()

    # Read entiteis and offensive texts files
    entities = read_texts(args.entities)
    offensives = read_texts(args.offensives)

    queries = get_queries(entities=entities, offensives=offensives)


    # Twint config
    config = twint.Config()
    config.Limit = args.tweetslimit
    config.Store_csv = True
    config.Output = args.output
    config.Lang = 'tr'

    nest_asyncio.apply()

    for q in tqdm(queries):
        config.Search = 'lang:tr ' + q
        twint.run.Search(config)
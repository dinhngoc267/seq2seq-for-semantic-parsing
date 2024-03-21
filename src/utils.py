import json
import re
import random
import string


def normalize_cased_token(token):
    flag = False
    for c in token:
        if c.isupper():
            flag = True
            break
    if flag:
        token = token.lower()
        token = token[0].upper() + token[1:]

    return token


def tokenize_sequence(text):
    tokens = re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)
    # normalized_tokens = [normalize_cased_token(token) for token in tokens]

    return tokens


def compact_query(query):
    if 'match (e)-[:SubEvent|mo_ta]->(e1)' in query:
        query = query.split('match (e)-[:SubEvent|mo_ta]->(e1)')[0]
        query = query + 'return dien_bien'
    if '[:NextEvent|Result|Causal]-(e1) where (e)-[:NextEvent|Result]->(e1) or (e)<-[:Causal]-(e1)' in query:
        query = query.replace(
            '[:NextEvent|Result|Causal]-(e1) where (e)-[:NextEvent|Result]->(e1) or (e)<-[:Causal]-(e1)',
            '[:dan_den]-(e1)')
    if 'match (e)-[:Causal|nguyen_nhan|boi_canh]->(e1)' in query:
        query = query.split('match (e)-[:Causal|nguyen_nhan|boi_canh]->(e1)')[0]
        query = query + 'return nguyen_nhan'
    if 'match (e)-[:Result|ket_qua]->(e1)' in query:
        query = query.split('match (e)-[:Result|ket_qua]->(e1)')[0]
        query = query + 'return ket_qua'
    if '(e)-[:SubEvent]-(e1:TranChien|ChienDich) where not (e1)<-[:NextEvent]-() and not (e1)-[:Causal]->()' in query:
        query = query.replace(
            '(e)-[:SubEvent]-(e1:TranChien|ChienDich) where not (e1)<-[:NextEvent]-() and not (e1)-[:Causal]->()',
            '[:mo_dau]-(e1)')
    if 'match path=(e)-[*0..1]-(y:Entity) return path'.replace(' ', '') in query.replace(' ', ''):
        query = query.split('match path')[0]
        query = query + 'return e'
    if 'match path=(e1)-[*0..1]-(y:Entity) return path'.replace(' ', '') in query.replace(' ', ''):
        query = query.split('match path')[0]
        query = query + 'return e1'

    query = query.replace(".name", "").replace("(", "").replace(")", "")

    return query

def extract_entity_in_query(query):

    quoted = re.compile('"[^"]*"')

    entities = []
    for entity in quoted.findall(query):
        entity = entity.replace('"','')
        entities.append(entity)

    return entities


def create_vocab(train_questions: list,
                 train_queries: list,
                 keywords_path: str) -> dict:
    """
    :param train_questions: list of questions in train set
    :param train_queries: list of queries in train set
    :param keywords_path: list of keywords which must have in vocabulary
    :return: vocab dictionary
    """

    question_tokens = []
    for item in train_questions:
        question_tokens += tokenize_sequence(item)

    query_tokens = []
    entity_tokens = []
    for item in train_queries:
        query_tokens += tokenize_sequence(item)
        entities = extract_entity_in_query(item)

        for entity in entities:
            entity_tokens += tokenize_sequence(entity)
    query_tokens = list(set(query_tokens))

    keywords = json.load(open(keywords_path, 'r'))
    keyword_tokens = []

    for item in keywords:
        keyword_tokens.extend(tokenize_sequence(item))
    key_tokens = list(set(keyword_tokens))

    mask_entity_tokens = random.choices(list(set([token for token in entity_tokens if token not in string.punctuation
                                                  and token not in key_tokens])),
                                        k=int(0.4 * len(set(entity_tokens))))

    mask_normal_tokens = random.choices(list(set([token for token in question_tokens if token not in string.punctuation
                                                  and token not in key_tokens]).difference(set(entity_tokens))),
                                        k=int(0.008 * len(set(question_tokens))))

    # mask_normal_tokens.extend(["em", "ạ"])
    mask = mask_entity_tokens + mask_normal_tokens
    print(set(mask))
    print(f'Mask {len(mask_entity_tokens)} entity tokens and {len(mask_normal_tokens)} normal tokens')

    vocab = {'<pad>': 0, '<unk>': 1, '</s>': 2, '<s>': 3}
    count = 4

    tokens = query_tokens + question_tokens
    for token in set(tokens):
        if token not in vocab and token not in mask:  # mask:#(normal_entities_mask + entities_mask):
            vocab[token] = count
            count += 1

    return vocab

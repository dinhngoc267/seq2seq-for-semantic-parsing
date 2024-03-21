import torch
import pandas as pd
import pytorch_lightning as pl
import random
import string
from utils import tokenize_sequence
from model import CopySeq2Seq
from copynet import ParserDataset
from logger import HistoryLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import re
import copy

from model import TRAIN_BATCH_SIZE, VAL_BATCH_SIZE

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_float32_matmul_precision("medium")
pl.seed_everything(42)


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


def compact_role(query):

    role_pattern = re.compile('\[[^\[]*\]')
    tmp = copy.deepcopy(query)
    for value in role_pattern.findall(query):
        if '_' in value:
            role_tokens = re.findall(r"\w+|[^\w\s]", value.lower(), re.UNICODE)
            if value == '[:dien_ra_vao|thoi_gian_ket_thuc]':
                replace_string = 'thoi_gian_ket_thuc'
            elif value == '[:dien_ra_vao|thoi_gian_bat_dau]':
                replace_string = 'thoi_gian_bat_dau'
            else:
                replace_string = [token for token in role_tokens if '_' in token][0]
            tmp = tmp.replace(value, replace_string)

    return tmp


def process(query):
    query = query.replace(".name", "").replace("(", "").replace(")", "")

    return query


if __name__ == "__main__":

    train_data = pd.read_csv("/home/nld/kg-reasoning/my_notebooks/semantic_parsing/augmented_data.csv")

    train_questions = [x.strip().replace("–", "-").replace('\n', ' ').replace('\r', '').replace('\t', '') for x in train_data['Questions'].dropna().tolist()]
    train_queries = [x.strip().replace("–", "-").replace('\n', ' ').replace('\r', '').replace('\t', '') for x in train_data['Queries'].dropna().tolist()]

    quoted = re.compile('"[^"]*"')
    entities = []
    for idx, q in enumerate(train_queries):
        for value in quoted.findall(q):
            value = value.replace('"', '')
            entities.append(value)

    train_queries = [compact_query(x) for x in train_queries]
    train_data = pd.DataFrame(data=zip(train_questions, train_queries), columns=['inputs', 'outputs'])
    train_data.drop_duplicates(inplace=True)

    print(f"Train Dataset size: {len(train_data)}")

    max_input_length = max([len(tokenize_sequence(item)) for item in train_questions]) + 2
    max_output_length = max([len(tokenize_sequence(item)) for item in train_queries]) + 3

    print(max_input_length, max_output_length)

    valid_data = pd.read_csv("/home/nld/kg-reasoning/my_notebooks/semantic_parsing/val.csv")
    valid_questions = [x.strip().replace("–", "-").replace('\n', ' ') for x in valid_data['Questions'].dropna().tolist()]
    valid_queries = [x.strip().replace("–", "-").replace('\n', ' ') for x in valid_data['Cypher Query'].dropna().tolist()]
    valid_queries = [compact_query(x) for x in valid_queries]
    valid_data = pd.DataFrame(data=zip(valid_questions, valid_queries), columns=['inputs', 'outputs'])

    print(max([len(tokenize_sequence(item)) for item in valid_questions]))
    print(max([len(tokenize_sequence(item)) for item in valid_queries]))

    question_tokens = []

    for item in train_questions:
        question_tokens += tokenize_sequence(item)

    queries_tokens = []

    for item in train_queries:
        queries_tokens += tokenize_sequence(item)

    key_words = ["phong trào", "cuộc khởi nghĩa", "của", "hội nghị", "cuộc tiến công", "liên hiệp", "liên minh", "ở",
                 "diễn ra ở", "vào", "hãy nêu", "trình bày", "mục đích là gì", "tại sao", "nguyên nhân", "chính sách",
                 "khi nào", "ngày tháng năm thế kỉ đêm sáng", "đại hội", "thành lập", "thời gian nào", "từ đến",
                 "hiệp ước", "nhà nước", "ra đời", "chống", "ngày tháng năm sáng đêm những", "thực dân",
                 "chính sách", "hiệp định", "kế hoạch", "ban hành", "chiến lược", "cách mạng", "cuối thế kỉ sông thống",
                 "giang trên thị xã đi tham gia", "quân và vào hiến pháp kế hoạch đội tham dự theo bị ám sát hại",
                 "hội nghị họp lần cuộc", "ném", "đảng sau cùng", "quốc" "e e1 a b c d f n e g 0 1", "chiếm trong",
                 "thông trong không cho", "hòa ước kí kết bởi thúc chiến thắng đầu hàng thất bại chiến dịch phong",
                 "trào đấu tranh đánh dấu lãnh đạo", "triều đình", "nhà", "ám sát thập niên",
                 "phái giữa đầu cuối đêm sáng tướng nội chiến tỉnh", '(', ')', '*', '-', '.', ':', '<', '=', '>', '[',
                 ']', 'and', 'as', 'b', 'ban_hanh', 'boi_canh', 'c', 'causal', 'chiendich', 'd', 'dien_ra_o',
                 'dien_ra_vao', 'dienracuochop', 'doi_tuong_ban_hanh', 'doi_tuong_bi_chiem', 'doi_tuong_bi_giet',
                 'doi_tuong_bi_lat_do', 'doi_tuong_bi_phe_truat', 'doi_tuong_bi_tan_cong', 'doi_tuong_bi_thay_the',
                 'doi_tuong_bo_nhiem', 'doi_tuong_chiem', 'giet', 'in', 'ket_qua', 'kiket', 'l', 'match',  'not', 'or',
                 'p', 'path', 'result', 'return', 'subevent', 'ten_goi', 'thanh_lap_boi', 'thanhlap', "giặc", "miền",
                 "người", "ném bom", "phe", "vùng", "nửa", "khôi", "hạm đội", "viết", "lệnh", "lãnh thổ", "luật",
                 "căn cứ", "hòa hoãn", "huyện tỉnh thị xã", "điểm", "tây", "”", "“", "thuộc", "mô tả",
                 'thoi_gian_bat_dau', 'thuoc', 'tranchien', 'vai_tro', 'where', 'with', 'x', 'y', 'y_nghia', '|']

    entity_tokens = []
    for item in entities:
        entity_tokens.extend(tokenize_sequence(item))

    key_tokens = []
    for item in key_words:
        key_tokens.extend(tokenize_sequence(item))
    key_tokens = list(set(key_tokens))
    print(key_tokens)

    mask_entity_tokens = random.choices(list(set([token for token in entity_tokens if token not in string.punctuation
                                                  and token not in key_tokens])),
                                        k=int(0.4 * len(set(entity_tokens))))

    mask_normal_tokens = random.choices(list(set([token for token in question_tokens if token not in string.punctuation
                                                  and token not in key_tokens]).difference(set(entity_tokens))),
                                        k=int(0.008 * len(set(question_tokens))))
    mask_normal_tokens.extend(["em", "ạ"])
    mask = mask_entity_tokens + mask_normal_tokens
    print(set(mask))
    print(f'Mask {len(mask_entity_tokens)} entity tokens and {len(mask_normal_tokens)} normal tokens')

    vocab = {'<pad>': 0, '<unk>': 1, '</s>': 2, '<s>': 3}
    count = 4

    tokens = queries_tokens + question_tokens
    for token in set(tokens):
        if token not in vocab and token not in mask:  # mask:#(normal_entities_mask + entities_mask):
            vocab[token] = count
            count += 1
    print(len(vocab))
    # import json
    # vocab = json.load(open("vocab.json", "r"))
    # print(list(vocab.keys())[5:10])

    id2token = {}
    for token, id in vocab.items():
        id2token[id] = token

    model = CopySeq2Seq(embedding_dim=300,
                        hidden_dim=512,
                        vocab_size=len(vocab),
                        num_layers=3,
                        dropout=0.2,
                        input_len=max_input_length,
                        sos_token_id=vocab['<s>'],
                        vocab=vocab,
                        id2token=id2token,
                        max_output_length=max_output_length).to(device)

    train_dataset = ParserDataset(data=train_data,
                                  max_output_length=max_output_length,
                                  max_input_length=max_input_length,
                                  device=device,
                                  vocab=vocab)

    val_dataset = ParserDataset(data=valid_data,
                                max_output_length=max_output_length,
                                max_input_length=max_input_length,
                                device=device,
                                vocab=vocab)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=TRAIN_BATCH_SIZE, num_workers=63)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=VAL_BATCH_SIZE, num_workers=2)
    logger = HistoryLogger()
    checkpoint = ModelCheckpoint(dirpath="ckpts", save_top_k=1, monitor="val_bleu_score", mode="max")
    trainer = pl.Trainer(accelerator="gpu", min_epochs=0, max_epochs=250, logger=logger,
                         log_every_n_steps=0, callbacks=[checkpoint])
    trainer.fit(model, train_dataloader, val_dataloader)

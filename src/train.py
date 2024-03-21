import torch
import pytorch_lightning as pl

from copynet.model import CopySeq2Seq
from copynet.dataset import ParserDataset
from logger import HistoryLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision("medium")
pl.seed_everything(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


TRAIN_BATCH_SIZE=32
VAL_BATCH_SIZE=128


if __name__ == "__main__":

    train_dataset = ParserDataset(data_path="data/augmented_data.csv",
                                  device=device,
                                  keywords_path="data/keywords.json")

    vocab = train_dataset.get_vocab()
    id2word_mapping = train_dataset.get_id2word_mapping()
    max_input_length = train_dataset.get_max_input_length()
    max_output_length = train_dataset.get_max_output_length()

    val_dataset = ParserDataset(data_path="data/val.csv",
                                device=device,
                                max_output_length=max_output_length,
                                max_input_length= max_input_length,
                                vocab=vocab)

    model = CopySeq2Seq(embedding_dim=300,
                        hidden_dim=512,
                        vocab_size=len(vocab),
                        num_layers=3,
                        dropout=0.2,
                        input_len=max_input_length,
                        sos_token_id=vocab['<s>'],
                        vocab=vocab,
                        id2token=id2word_mapping,
                        max_output_length=max_output_length).to(device)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=TRAIN_BATCH_SIZE, num_workers=63)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=VAL_BATCH_SIZE, num_workers=2)

    logger = HistoryLogger()
    checkpoint = ModelCheckpoint(dirpath="ckpts", save_top_k=1, monitor="val_bleu_score", mode="max")
    trainer = pl.Trainer(accelerator="gpu",
                         min_epochs=0,
                         max_epochs=250,
                         logger=logger,
                         log_every_n_steps=0,
                         callbacks=[checkpoint])

    trainer.fit(model, train_dataloader, val_dataloader)

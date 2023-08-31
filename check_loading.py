from sparseml.pytorch.optim import ScheduledModifierManager
from fastprogress import master_bar, progress_bar
from fastai.vision.all import SimpleNamespace, set_seed
import wandb
import whisper
import torch
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from dataset import JvsSpeechDataset, WhisperDataCollatorWhithPadding
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from utils import (
    create_dirs_if_not_exist,
    set_weight_decay,
    define_metrics,
    compute_metrics,
)
import os
import evaluate

models_dir = os.getenv("MODELS")
model_size = "small"
run_name = f"{model_size}_check_vanilla"
save_dir = f"{models_dir}/checkpoints/FinetuneWhisper/{model_size}/"
model_to_load = "small_10e_full_data_(9).tar"

create_dirs_if_not_exist(save_dir)

config = SimpleNamespace(
    seed=42,
    lr=0.0005,
    batch_size=2,
    epochs=10,
    dropout=0.2,
    weight_decay=0.01,
    acu_steps=128,
    sample_rate=16000,
)

run = wandb.init(
    project="finetune-whisper",
    entity="ludeksvoboda",
    config=config,
    job_type=run_name,
    name=run_name,
)

set_seed(config.seed)

config = wandb.config

common_voice = DatasetDict()

common_voice["train"] = load_dataset(
    "mozilla-foundation/common_voice_13_0", "cs", split="train[:1%]", token=True
)
common_voice["test"] = load_dataset(
    "mozilla-foundation/common_voice_13_0", "cs", split="test", token=True
)
common_voice = common_voice.remove_columns(
    [
        "accent",
        "age",
        "client_id",
        "down_votes",
        "gender",
        "locale",
        "path",
        "segment",
        "up_votes",
    ]
)

feature_extractor = WhisperFeatureExtractor.from_pretrained(
    f"openai/whisper-{model_size}"
)
tokenizer = WhisperTokenizer.from_pretrained(
    f"openai/whisper-{model_size}", language="cs", task="transcribe"
)
processor = WhisperProcessor.from_pretrained(
    f"openai/whisper-{model_size}", language="cs", task="transcribe"
)

common_voice = common_voice.cast_column(
    "audio", Audio(sampling_rate=config.sample_rate)
)


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = common_voice.map(
    prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4
)

woptions = whisper.DecodingOptions(language="cs", without_timestamps=True)
model = whisper.load_model(model_size)

# checkpoint = torch.load(f"{save_dir}{model_to_load}")
# model.load_state_dict(checkpoint["model_state_dict"])

# dataset = JvsSpeechDataset(common_voice["train"])
# loader = torch.utils.data.DataLoader(
#     dataset, batch_size=config.batch_size, collate_fn=WhisperDataCollatorWhithPadding()
# )

test_dataset = JvsSpeechDataset(common_voice["test"])
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    collate_fn=WhisperDataCollatorWhithPadding(),
    shuffle=False,
)

loss_fn = CrossEntropyLoss(ignore_index=-100)

# no_decay = ["bias", "LayerNorm.weight"]
# optimizer_grouped_parameters = set_weight_decay(model, config.weight_decay, no_decay)

# optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr)

metric_information = {
    "val_loss": "val_step",
    "val_wer": "val_step",
    "val_cer": "val_step",
    "train_loss": "train_step",
    "train_wer": "train_step",
    "train_cer": "train_step",
}

define_metrics(metric_information)

metrics_wer = evaluate.load("wer")
metrics_cer = evaluate.load("cer")

###Cut subset of data before testing
mb = master_bar(range(config.epochs))
train_step = 0
val_step = 0
acu_wer = 0
acu_cer = 0
accumulated_loss = 0
idx = 0
for epoch in mb:
    for batch in progress_bar(test_loader, len(test_loader), parent=mb):
        input_ids = batch["input_ids"].cuda()

        labels = batch["labels"].long().cuda()
        dec_input_ids = batch["dec_input_ids"].long().cuda()

        with torch.no_grad():
            audio_features = model.encoder(input_ids)

            out = model.decoder(dec_input_ids, audio_features)
            val_loss = loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

            val_cer, val_wer = compute_metrics(out, labels, tokenizer, metrics_cer, metrics_wer)

        wandb.log(
            {
                "val_loss": val_loss,
                "val_wer": val_wer,
                "val_cer": val_cer,
                "val_step": val_step,
            }
        )
        val_step += 1
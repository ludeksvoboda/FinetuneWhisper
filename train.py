from fastprogress import master_bar, progress_bar
from fastai.vision.all import SimpleNamespace, set_seed
import wandb
import whisper
import torch
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from dataset import JvsSpeechDataset, WhisperDataCollatorWhithPadding
from torch.nn import CrossEntropyLoss
import evaluate
from torch.optim import AdamW
from utils import create_dirs_if_not_exist, set_weight_decay, define_metrics
import os

models_dir = os.getenv('MODELS')
model_size = 'medium'
run_name = f'{model_size}_10e_full_data'
save_dir = f'{models_dir}/checkpoints/FinetuneWhisper/{model_size}/'

create_dirs_if_not_exist(save_dir)

config = SimpleNamespace(
    seed = 42,
    lr = 0.0005,
    batch_size = 2,
    epochs = 10,
    dropout = 0.2,
    weight_decay = 0.01,
    acu_steps = 128
)

SAMPLE_RATE = 16000
BATCH_SIZE = 2
TRAIN_RATE = 0.8

AUDIO_MAX_LENGTH = 480000
TEXT_MAX_LENGTH = 120
run = wandb.init(project="finetune-whisper",entity="ludeksvoboda", config=config, job_type=run_name, name=run_name)

set_seed(config.seed)

config = wandb.config

common_voice = DatasetDict()

# common_voice["train"] = load_dataset("mozilla-foundation/common_voice_13_0", "cs", split="train+validation", use_auth_token=True)
common_voice["train"] = load_dataset("mozilla-foundation/common_voice_13_0", "cs", split="train+validation", token=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_13_0", "cs", split="test", token=True)
common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{model_size}")
tokenizer = WhisperTokenizer.from_pretrained(f"openai/whisper-{model_size}", language="cs", task="transcribe")
processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}", language="cs", task="transcribe")

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4)

woptions = whisper.DecodingOptions(language="cs", without_timestamps=True)
model = whisper.load_model(model_size)

dataset = JvsSpeechDataset(common_voice['train'])
loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, collate_fn=WhisperDataCollatorWhithPadding())

test_dataset = JvsSpeechDataset(common_voice['test'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=WhisperDataCollatorWhithPadding(), shuffle=False)

loss_fn = CrossEntropyLoss(ignore_index=-100)
metrics_wer = evaluate.load("wer")
metrics_cer = evaluate.load("cer")

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = set_weight_decay(model, config.weight_decay, no_decay)

optimizer = AdamW(optimizer_grouped_parameters, 
                          lr=config.lr)

metric_information = {
    'val_loss': 'val_step',
    'val_wer': 'val_step',
    'val_cer': 'val_step',
    'train_loss': 'train_step',
    'train_wer': 'train_step',
    'train_cer': 'train_step'
}

define_metrics(metric_information)

###Cut subset of data before testing
mb = master_bar(range(config.epochs))
train_step = 0
val_step = 0
acu_wer = 0
acu_cer = 0
accumulated_loss = 0
idx = 0
for epoch in mb:
    for batch in progress_bar(loader, len(loader), parent=mb):
        input_ids = batch["input_ids"].cuda()

        labels = batch["labels"].long().cuda()
        dec_input_ids = batch["dec_input_ids"].long().cuda()

        with torch.no_grad():
            audio_features = model.encoder(input_ids)

        out = model.decoder(dec_input_ids, audio_features)
        loss = loss_fn(out.view(-1, out.size(-1)), labels.view(-1)) / config.acu_steps
        accumulated_loss += loss.item()
        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            o_list.append(tokenizer.decode(o, skip_special_tokens=True))
            l_list.append(tokenizer.decode(l, skip_special_tokens=True))
        cer = metrics_cer.compute(references=l_list, predictions=o_list)
        wer = metrics_wer.compute(references=l_list, predictions=o_list)

        acu_wer += wer
        acu_cer += cer

        loss.backward()

        if ((idx + 1) % config.acu_steps == 0) or (idx + 1 == len(loader)):
            optimizer.step()
            optimizer.zero_grad()
            wandb.log({"train_loss": accumulated_loss,"train_wer": wer, 
                   "train_cer": cer, "train_step": train_step})
            acu_wer = 0
            acu_cer = 0
            accumulated_loss = 0
            train_step += 1        
        idx += 1
    torch.save({'descripiton': """Full run
                    """,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, f'{save_dir}{run_name}_({str(epoch)}).tar')

    for batch in progress_bar(test_loader, len(test_loader), parent=mb):
        input_ids = batch["input_ids"].cuda()

        labels = batch["labels"].long().cuda()
        dec_input_ids = batch["dec_input_ids"].long().cuda()

        with torch.no_grad():
            audio_features = model.encoder(input_ids)

            out = model.decoder(dec_input_ids, audio_features)
            val_loss = loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
            o_list, l_list = [], []
            for o, l in zip(out, labels):
                o = torch.argmax(o, dim=1)
                o_list.append(tokenizer.decode(o, skip_special_tokens=True))
                l_list.append(tokenizer.decode(l, skip_special_tokens=True))
            val_cer = metrics_cer.compute(references=l_list, predictions=o_list)
            val_wer = metrics_wer.compute(references=l_list, predictions=o_list)

        wandb.log({"val_loss": val_loss, "val_wer": val_wer,
                "val_cer": val_cer, "val_step": val_step})
        val_step += 1
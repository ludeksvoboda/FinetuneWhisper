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
from utils import create_dirs_if_not_exist

##Save commonvoice mapped dataset before going full so you do not have to do it every time

model_size = 'medium'
run_name = 'wholedata'
save_dir = '/home/lu/Models/checkpoints/FinetuneWhisper/medium/'

create_dirs_if_not_exist(save_dir)

config = SimpleNamespace(
    seed = 42,
    lr = 0.0005,
    batch_size = 1,
    epochs = 50,
    dropout = 0.2,
    weight_decay = 0.01
)

SAMPLE_RATE = 16000
BATCH_SIZE = 2
TRAIN_RATE = 0.8

AUDIO_MAX_LENGTH = 480000
TEXT_MAX_LENGTH = 120
run = wandb.init(project="finetune-whisper",entity="ludeksvoboda", config=config, job_type=run_name)

set_seed(config.seed)

config = wandb.config

common_voice = DatasetDict()

common_voice["train"] = load_dataset("mozilla-foundation/common_voice_13_0", "cs", split="train+validation", use_auth_token=True)
# common_voice["train"] = load_dataset("mozilla-foundation/common_voice_13_0", "cs", split="train", use_auth_token=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_13_0", "cs", split="test", use_auth_token=True)
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
model = whisper.load_model("medium")

dataset = JvsSpeechDataset(common_voice['train'])
loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=WhisperDataCollatorWhithPadding())

test_dataset = JvsSpeechDataset(common_voice['test'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=WhisperDataCollatorWhithPadding())

loss_fn = CrossEntropyLoss(ignore_index=-100)
metrics_wer = evaluate.load("wer")
metrics_cer = evaluate.load("cer")

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

optimizer = AdamW(optimizer_grouped_parameters, 
                          lr=config.lr)

###Cut subset of data before testing
mb = master_bar(range(config.epochs))
for epoch in mb:
    for batch in progress_bar(loader, len(loader), parent=mb):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].cuda()

        labels = batch["labels"].long().cuda()
        dec_input_ids = batch["dec_input_ids"].long().cuda()

        with torch.no_grad():
            audio_features = model.encoder(input_ids)

        out = model.decoder(dec_input_ids, audio_features)
        loss = loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            o_list.append(tokenizer.decode(o, skip_special_tokens=True))
            l_list.append(tokenizer.decode(l, skip_special_tokens=True))
        cer = metrics_cer.compute(references=l_list, predictions=o_list)
        wer = metrics_wer.compute(references=l_list, predictions=o_list)
        loss.backward()
        optimizer.step()
        ##Make wandb log

        wandb.log({"train_loss": loss, "train_wer": wer, "train_cer": cer})
    torch.save({'descripiton': """Full run
                    """,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, f'{save_dir}{run_name}_({str(epoch)}).tar')

    optimizer.zero_grad()
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
        ##Make wandb log

        wandb.log({"val_loss": val_loss, "wr": val_wer, "val_cer": val_cer})
import torch
import numpy as np

class JvsSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict) -> None:
        super().__init__()

        self.data_dict = data_dict

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, id):
        data_row = self.data_dict[id]

        # audio
        mel = torch.tensor(data_row['input_features'])

        text = data_row['labels']
        labels = text[1:]
        dec_in_ids = text[:-1]

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": dec_in_ids
        }
    
class WhisperDataCollatorWhithPadding:
    def __call__(sefl, features):
        input_ids, labels, dec_input_ids = [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])
        
        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)

        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)] # 50257 is eot token id

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_ids"] = input_ids

        return batch
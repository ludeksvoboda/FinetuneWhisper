from sparseml.pytorch.utils import ModuleExporter
import torch._C._onnx as _C_onnx
import torch
from os.path import isfile
import os
import whisper

models_dir = os.getenv("MODELS")
model_size = "small"
save_dir = f"{models_dir}/checkpoints/FinetuneWhisper/{model_size}/"
model_to_load = "small_sparsify_full_data_(9).tar"

if not isfile(f"{save_dir}{model_to_load}"):
    raise ValueError("Model to load does not exist")

model = whisper.load_model(model_size)

checkpoint = torch.load(f"{save_dir}{model_to_load}")
model.load_state_dict(checkpoint["model_state_dict"])
device = torch.device("cpu")
model = model.eval().to(device)

# model = model.decoder
exporter = ModuleExporter(model, output_dir=save_dir)

###Decoder features torch.Size([2, 1500, 768]), decoded input ids torch.Size([2, 31])
exporter.export_onnx((torch.rand((1, 80, 3000)), (torch.randint(20, (1, 31)))), name="sparse-model_full.onnx", training=_C_onnx.TrainingMode.EVAL, do_constant_folding=False)
import torch
from torch.utils.data import DataLoader
import pandas as pd
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification, DebertaV2TokenizerFast, DebertaV2ForSequenceClassification, EvalPrediction, DataCollatorWithPadding
from datasets import Dataset, DatasetDict
from peft import get_peft_model, LoraConfig, PeftModel
import evaluate
from tqdm import tqdm

from pathlib import Path
import yaml
import os
from argparse import ArgumentParser
import gc

import utils


def preprocess(examples):
    text_encoding = tokenizer(
        examples["text_concat"],
        padding=True,
        truncation=True,
        max_length=512,
        # return_tensors="pt",
    )
    text_encoding["labels"] = examples["label"]

    return text_encoding


def get_model(num_labels: int):
    if "deberta" in args.model_name:
        model = DebertaV2ForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)
    else: 
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    if args.finetune_new_model:
        lora_config = LoraConfig(
            task_type="SEQ_CLS", 
            inference_mode=False, 
            r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print(f"Loading existing {args.model_name} adapter.")
        model = PeftModel.from_pretrained(model, f'{output_dir}/best_model_adapter')

    return model


def compute_metrics(eval_pred: EvalPrediction):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = logits.argmax(-1)
    
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    project_root: Path = utils.get_project_root()
    os.chdir(project_root)
    with open(str(project_root / "config/experiments_config.yaml")) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    with open(str(project_root / "config/data_generation_config.yaml")) as f:
        params_data_gen = yaml.load(f, Loader=yaml.FullLoader)
    
    # should move these to a yaml
    parser = ArgumentParser()
    parser.add_argument("--random_state", type=int, default=1911)
    parser.add_argument("--model_name", type=str, default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--inf_save_dir", type=str, default="/nlp/edgehetero-nodeproppred/data/embeddings/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup_ratio", type=float, default=0.6)
    parser.add_argument("--label_smoothing", type=float, default=0.3)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--finetune_new_model", action="store_true", default=False)
    parser.add_argument("--perform_inference", action="store_true", default=True)
    args = parser.parse_args("")

    output_dir = str(project_root) + "/models/" + args.model_name.split("/")[-1] # for LM.
    path_to_data = str(Path(params["data"]["graph_dataset"][params["dataset"]]))
    data = torch.load(path_to_data).to("cpu")
    num_labels = data["paper"].y.unique().size()[0]

    path_to_data = str(Path(params_data_gen["data"]["path_to_metadata"][params["dataset"]]))
    df = pd.read_parquet(path_to_data)[["label", "text_concat"]] # assuming pubmed for now.

    dataset = DatasetDict({
        "train": Dataset.from_pandas(df.iloc[data["paper"].train_idx], preserve_index=True), # __index_level_0__
        "test": Dataset.from_pandas(df.iloc[data["paper"].test_idx], preserve_index=True),
        "validation": Dataset.from_pandas(df.iloc[data["paper"].val_idx], preserve_index=True),
    })

    inf_dataset = Dataset.from_pandas(df, preserve_index=True)

    if "deberta" in args.model_name:
        tokenizer = DebertaV2TokenizerFast.from_pretrained(args.model_name, do_lower_case=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=True)

    model = get_model(num_labels)

    if args.finetune_new_model:
        encoded_dataset = dataset.map(preprocess, batched=True)

        warmup_steps = int(args.warmup_ratio * (len(encoded_dataset["train"]) // args.batch_size + 1))
        training_args = TrainingArguments(
                seed=args.random_state,
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=1,
                learning_rate=args.lr,
                load_best_model_at_end=True,
                metric_for_best_model="eval_accuracy",
                label_smoothing_factor=args.label_smoothing,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                warmup_steps=warmup_steps,
                lr_scheduler_type="linear",
                num_train_epochs=args.num_epochs,
                fp16=args.fp16,
            )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["validation"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        model.save_pretrained(f'{output_dir}/best_model_adapter')
        gc.collect()
        torch.cuda.empty_cache()
    
    if args.perform_inference:
        encoded_inf_dataset = inf_dataset.map(preprocess, batched=True, remove_columns=inf_dataset.column_names)
        encoded_inf_dataset.set_format("torch")

        # can't access last hidden state if we use Trainer.
        emb = torch.empty(size=(0, 768), dtype=torch.float16, device=args.device)
        logits = torch.empty(size=(0, num_labels), dtype=torch.float16, device=args.device)

        eval_dataloader = DataLoader(encoded_inf_dataset, shuffle=False, batch_size=args.batch_size*8)
        model.to(dtype=torch.float16, device=args.device)
        model.eval()
        for batch in tqdm(eval_dataloader):
            for k, v in batch.items():
                batch[k] = v.to(args.device)
            with torch.no_grad():
                out = model(**batch, output_hidden_states=True, return_dict=True)
            temp_embs = out.hidden_states[-1].permute(1, 0, 2)[0]
            temp_logits = out.logits
            emb = torch.cat([emb, temp_embs], 0)
            logits = torch.cat([logits, temp_logits], 0)

        torch.save(emb, args.inf_save_dir + f"{params['dataset']}_simtg_x_embs.pt")
        torch.save(logits, args.inf_save_dir + f"{params['dataset']}_simtg_logits.pt")
        gc.collect()
        torch.cuda.empty_cache()
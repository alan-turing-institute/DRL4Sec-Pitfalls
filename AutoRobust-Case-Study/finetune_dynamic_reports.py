import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # Force single GPU usage
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import torch
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    Trainer, 
    TrainingArguments,
    EvalPrediction
)

@dataclass
class ClassificationExample:
    text: str
    label: int  # 0 for goodware, 1 for malware
    label_name: str  # "goodware" or "malware"
    report_hash: str
    source_path: Path

@dataclass 
class ClassificationDataset:
    # Container for the prepared classification dataset.
    examples: List[ClassificationExample]
    
    def get_class_distribution(self) -> Dict[str, int]:
        dist = {"goodware": 0, "malware": 0}
        for ex in self.examples:
            dist[ex.label_name] += 1
        return dist
    
    def __len__(self) -> int:
        return len(self.examples)


def load_label_map(labels_csv: Path) -> Dict[str, str]:
    # Load hash to label mapping from labels.csv.
    label_map: Dict[str, str] = {}
    with labels_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "hash" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("labels.csv must contain 'hash' and 'label' columns")
        for row in reader:
            label_map[row["hash"].strip().lower()] = row["label"].strip().lower()
    return label_map


def load_classification_dataset(
    goodware_dir: Path,
    malware_dir: Path,
    label_map: Dict[str, str],
    seed: int = 42,
) -> ClassificationDataset:
    """Load and prepare classification dataset from JSON reports."""
    
    examples: List[ClassificationExample] = []
    
    # Collect all report paths with their labels
    all_paths = []
    for path in goodware_dir.rglob("*.json"):
        all_paths.append((path, "goodware"))
    for path in malware_dir.rglob("*.json"):
        all_paths.append((path, "malware"))
    
    rng = random.Random(seed)
    rng.shuffle(all_paths)
    
    for path, fallback_label in all_paths:
        report_hash = path.stem.lower()
        # Enforce binary labeling by directory
        label_name = fallback_label
        
        # Load JSON report as text
        try:
            with path.open("r", encoding="utf-8") as f:
                report_data = json.load(f)
        except (IOError, OSError, json.JSONDecodeError):
            continue
        summary_dict = report_data.get("summary") if isinstance(report_data, dict) else None
        flat_report = summary_dict if isinstance(summary_dict, dict) else report_data
        report_text = json.dumps(flat_report, ensure_ascii=False)
        
        label_id = 0 if label_name == "goodware" else 1
        
        examples.append(
            ClassificationExample(
                text=report_text,
                label=label_id,
                label_name=label_name,
                report_hash=report_hash,
                source_path=path,
            )
        )
    
    return ClassificationDataset(examples=examples)


def train_eval_split(dataset: ClassificationDataset, eval_ratio: float, seed: int) -> Tuple[ClassificationDataset, ClassificationDataset]:
    """Split dataset into train and evaluation sets."""

    rng = random.Random(seed)
    examples = dataset.examples.copy()
    rng.shuffle(examples)
    
    split_index = int(len(examples) * (1.0 - eval_ratio))
    split_index = max(1, split_index)
    
    train_examples = examples[:split_index]
    eval_examples = examples[split_index:] if split_index < len(examples) else []
    
    train_dataset = ClassificationDataset(examples=train_examples)
    eval_dataset = ClassificationDataset(examples=eval_examples) if eval_examples else None
    
    return train_dataset, eval_dataset


# Chunking utilities and data collator

def chunk_text(text: str, tokenizer: AutoTokenizer, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks based on token count."""
    # Tokenize the full text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += chunk_size - overlap 
    return chunks


@dataclass
class DataCollator:
    """Data collator that splits long texts into chunks for classification."""
    tokenizer: AutoTokenizer
    chunk_size: int = 512
    max_chunks_per_sample: int = 8  # Limit chunks to avoid memory issues

    def __call__(self, features: List[Dict[str, any]]):
        all_chunks = []
        all_labels = []
        sample_indices = []  # Track which sample each chunk belongs to
        
        for idx, feature in enumerate(features):
            text = feature["text"]
            label = feature["labels"]
            
            # Split text into chunks
            chunks = chunk_text(text, self.tokenizer, self.chunk_size)

            # Limit number of chunks per sample (deterministic: take first N)
            if len(chunks) > self.max_chunks_per_sample:
                chunks = chunks[:self.max_chunks_per_sample]
            
            # Add chunks and their labels
            all_chunks.extend(chunks)
            all_labels.extend([label] * len(chunks))
            sample_indices.extend([idx] * len(chunks))
        
        # Tokenize all chunks
        encoded = self.tokenizer(
            all_chunks,
            truncation=True,
            padding=True,
            max_length=self.chunk_size,
            return_tensors="pt"
        )
        
        encoded["labels"] = torch.tensor(all_labels, dtype=torch.long)
        return encoded


def compute_classification_metrics(eval_pred: EvalPrediction):
    """Compute accuracy, precision, recall, and F1 for binary classification."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# Model setup and training functions

def setup_model_for_classification(
    model_name_or_path: str,
    bnb_compute_dtype: str,
    gradient_checkpointing: bool,
    target_modules: Optional[Sequence[str]],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    bias: str,
    use_lora: bool = True,
) -> AutoModelForSequenceClassification:
    """Set up a model for binary classification with LoRA."""
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise SystemExit(
            "The peft package is required for LoRA fine-tuning. Install it via 'pip install peft'."
        ) from exc

    compute_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[bnb_compute_dtype]

    quant_config = None

    # Load model on a single device to avoid LoRA multi-GPU placement issues
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=2,  # Binary classification
        quantization_config=quant_config,
        trust_remote_code=True,
    )

    if hasattr(model, "gradient_checkpointing_enable") and gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Move base model to single device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=list(target_modules) if target_modules else None,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type="SEQ_CLS",  # Sequence classification
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("[INFO] LoRA disabled: full model fine-tuning (higher memory usage).")
    return model


def train_classifier(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    train_dataset: ClassificationDataset,
    eval_dataset: Optional[ClassificationDataset],
    training_args: TrainingArguments,
    chunk_size: int = 512,
    max_chunks_per_sample: int = 4,
) -> Trainer:
    """Train the classification model using chunked inputs."""
    
    # Convert to HuggingFace dataset format
    train_hf_dataset = Dataset.from_list([
        {"text": ex.text, "labels": ex.label} 
        for ex in train_dataset.examples
    ])
    
    eval_hf_dataset = None
    if eval_dataset is not None:
        eval_hf_dataset = Dataset.from_list([
            {"text": ex.text, "labels": ex.label} 
            for ex in eval_dataset.examples
        ])

    collator = DataCollator(
        tokenizer=tokenizer, 
        chunk_size=chunk_size,
        max_chunks_per_sample=max_chunks_per_sample
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf_dataset,
        eval_dataset=eval_hf_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_classification_metrics,
    )

    trainer.train()
    return trainer


def test_classifier(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    test_dataset: ClassificationDataset,
    chunk_size: int = 512,
    batch_size: int = 16,
) -> Dict[str, any]:
    """Test the trained classifier using majority voting over chunks."""
    
    model.eval()
    device = next(model.parameters()).device
    
    all_predictions = []
    all_true_labels = []
    
    # Process each example individually for majority voting
    for example in test_dataset.examples:
        # Split into chunks
        chunks = chunk_text(example.text, tokenizer, chunk_size)
        
        chunk_predictions = []
        
        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            # Tokenize batch
            encoded = tokenizer(
                batch_chunks,
                truncation=True,
                padding=True,
                max_length=chunk_size,
                return_tensors="pt"
            ).to(device)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**encoded)
                logits = outputs.logits
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                chunk_predictions.extend(batch_preds)
        
        # Majority voting: most common prediction across chunks
        if chunk_predictions:
            final_prediction = max(set(chunk_predictions), key=chunk_predictions.count)
        else:
            final_prediction = 0  # Default to goodware if no chunks
        
        all_predictions.append(final_prediction)
        all_true_labels.append(example.label)
    
    # Convert to numpy arrays
    pred_labels = np.array(all_predictions)
    true_labels = np.array(all_true_labels)
    
    # Compute metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='binary'
    )
    
    # Generate detailed classification report
    label_names = ["goodware", "malware"]
    detailed_report = classification_report(
        true_labels, pred_labels, target_names=label_names, output_dict=True
    )
    
    metrics = {
        "eval_accuracy": accuracy,
        "eval_f1": f1,
        "eval_precision": precision,
        "eval_recall": recall,
    }
    
    return {
        "metrics": metrics,
        "detailed_report": detailed_report,
        "predictions": pred_labels,
        "true_labels": true_labels,
    }


# Configuration and argument parsing

CONFIG = {
    "data": {
        "goodware_dir": Path("data/goodware_dataset/reports_summary"),
        "malware_dir": Path("data/malware_dataset"),
        "labels_csv": Path("data/labels.csv"),
        "eval_ratio": 0.1,
        "test_ratio": 0.1,
        "seed": 42,
    },
    "model": {
        "name": "distilbert-base-uncased",
        "target_modules": "q_lin,k_lin,v_lin,out_lin",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_bias": "none",
        "bnb_compute_dtype": "bfloat16",
    },
    "training": {
        "epochs": 4.0,
        "learning_rate": 2e-4,
        "weight_decay": 0.0,
        "warmup_ratio": 0.03,
        "gradient_accumulation_steps": 8,
        "logging_steps": 200,
        "eval_steps": 200,
        "save_steps": 200,
        "save_total_limit": 3,
        "chunk_size": 512,
        "max_chunks_per_sample": 20,
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a model for malware classification")
    parser.add_argument("--output-dir", type=Path, default=Path("models/distilbert"), help="Output directory for model and results")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size per device")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Evaluation batch size per device")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size (tokens)")
    parser.add_argument("--max-chunks-per-sample", type=int, default=8, help="Max chunks per sample")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument("--fp16", action="store_true", help="Use float16 precision")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA (full fine-tuning uses more memory)")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Pipeline-only mode: subsample each split to 8 examples, "
             "do 1 optimizer step, skip the post-training test eval. "
             "Used by reproducibility/smoke-test.sh to confirm the script "
             "can be imported, data loads, the model runs forward+backward, "
             "and Trainer wires up — without doing meaningful training."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 or --fp16")

    # Parse target modules from config
    target_modules = [fragment.strip() for fragment in CONFIG["model"]["target_modules"].split(",")]

    # Resolve data paths relative to this script
    script_dir = Path(__file__).resolve().parent
    goodware_dir = script_dir / CONFIG["data"]["goodware_dir"]
    malware_dir = script_dir / CONFIG["data"]["malware_dir"]
    labels_csv = script_dir / CONFIG["data"]["labels_csv"]

    # Load data
    label_map = load_label_map(labels_csv)
    full_dataset = load_classification_dataset(
        goodware_dir=goodware_dir,
        malware_dir=malware_dir,
        label_map=label_map,
        seed=CONFIG["data"]["seed"],
    )

    print(f"Total examples: {len(full_dataset)}")
    print(f"Class distribution: {full_dataset.get_class_distribution()}")

    # Three-way split: train, eval, test
    remaining_dataset, test_dataset = train_eval_split(full_dataset, CONFIG["data"]["test_ratio"], CONFIG["data"]["seed"])
    train_dataset, eval_dataset = train_eval_split(remaining_dataset, CONFIG["data"]["eval_ratio"], CONFIG["data"]["seed"] + 1)

    if args.smoke_test:
        # Pipe-through verification: cap every split to a tiny size so the
        # whole train+eval cycle finishes in ~10s rather than minutes.
        train_dataset.examples = train_dataset.examples[:8]
        if eval_dataset is not None:
            eval_dataset.examples = eval_dataset.examples[:4]
        if test_dataset is not None:
            test_dataset.examples = test_dataset.examples[:4]

    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset) if eval_dataset else 0}")
    print(f"Test examples: {len(test_dataset) if test_dataset else 0}")

    # Determine effective chunk parameters (CLI overrides if provided)
    effective_chunk_size = args.chunk_size or CONFIG["training"]["chunk_size"]
    effective_max_chunks = args.max_chunks_per_sample or CONFIG["training"]["max_chunks_per_sample"]
    print(f"Using chunk_size={effective_chunk_size}, max_chunks_per_sample={effective_max_chunks}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"]["name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Set up model
    model = setup_model_for_classification(
        model_name_or_path=CONFIG["model"]["name"],
        bnb_compute_dtype=CONFIG["model"]["bnb_compute_dtype"],
        gradient_checkpointing=args.gradient_checkpointing,
        target_modules=target_modules,
        lora_r=CONFIG["model"]["lora_r"],
        lora_alpha=CONFIG["model"]["lora_alpha"],
        lora_dropout=CONFIG["model"]["lora_dropout"],
        bias=CONFIG["model"]["lora_bias"],
        use_lora=not args.no_lora,
    )

    # Training arguments
    smoke = bool(getattr(args, "smoke_test", False))
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=(1 if smoke else CONFIG["training"]["gradient_accumulation_steps"]),
        num_train_epochs=(1 if smoke else CONFIG["training"]["epochs"]),
        max_steps=(1 if smoke else -1),
        save_strategy=("no" if smoke else "steps"),
        eval_strategy=("no" if smoke else ("steps" if eval_dataset is not None else "no")),
        learning_rate=CONFIG["training"]["learning_rate"],
        weight_decay=CONFIG["training"]["weight_decay"],
        warmup_ratio=CONFIG["training"]["warmup_ratio"],
        logging_steps=CONFIG["training"]["logging_steps"],
        eval_steps=CONFIG["training"]["eval_steps"],
        save_steps=CONFIG["training"]["save_steps"],
        save_total_limit=CONFIG["training"]["save_total_limit"],
        bf16=args.bf16,
        fp16=args.fp16,
        optim="adamw_torch",
        report_to="none",
        ddp_find_unused_parameters=False,
        load_best_model_at_end=(False if smoke else (True if eval_dataset is not None else False)),
        metric_for_best_model=(None if smoke else ("f1" if eval_dataset is not None else None)),
        greater_is_better=True,
        remove_unused_columns=False,
        label_names=["labels"],
        logging_strategy="steps",
    )

    # Train the model
    trainer = train_classifier(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=training_args,
        chunk_size=effective_chunk_size,
        max_chunks_per_sample=effective_max_chunks,
    )

    # Save model and tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")

    # Test model
    if test_dataset and not smoke:
        print("Evaluating on test set...")
        test_results = test_classifier(
            model=model,
            tokenizer=tokenizer,
            test_dataset=test_dataset,
            chunk_size=effective_chunk_size,
            batch_size=args.eval_batch_size,
        )
        
        # Save test results
        test_results_path = args.output_dir / "test_results.json"
        json_safe_results = {
            "metrics": test_results["metrics"],
            "detailed_report": test_results["detailed_report"],
        }
        with test_results_path.open("w", encoding="utf-8") as handle:
            json.dump(json_safe_results, handle, indent=2)
        
        print(f"Test results:")
        print(f"  Accuracy: {test_results['metrics']['eval_accuracy']:.4f}")
        print(f"  F1: {test_results['metrics']['eval_f1']:.4f}")
        print(f"  Precision: {test_results['metrics']['eval_precision']:.4f}")
        print(f"  Recall: {test_results['metrics']['eval_recall']:.4f}")
        print(f"Detailed test results saved to {test_results_path}")


if __name__ == "__main__":
    main()
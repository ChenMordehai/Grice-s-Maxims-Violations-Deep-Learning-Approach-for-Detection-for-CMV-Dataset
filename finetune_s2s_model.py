import pandas as pd

import csv
import evaluate
import nltk
import numpy as np
from typing import List, Tuple
from nltk.tokenize import sent_tokenize
from datasets import Dataset, concatenate_datasets
from huggingface_hub import HfFolder
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import classification_report, roc_auc_score

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer


INPUT_FILE_PATH = "csv_data.csv"
NEW_CSV_PATH = "manner_violation_csv_data.csv"
comments_data = pd.read_csv(INPUT_FILE_PATH)


def get_parent_text(parent_id):
    if str(parent_id) == "-1":
        return ""
    return comments_data.iloc[int(parent_id)]["text"]


def get_manner_class(row):
    if row["Sarcasm"] == "1":
        return True
    elif row["Ridicule"] == "1":
        return True
    elif row["Aggressive"] == "1":
        return True
    elif row["BAD"] == "1":
        return True
    return False


def get_relevance_class(row):
    if row["Irrelevance"] == "1":
        return True
    return False


def create_new_csv(input_csv_path, output_csv_path):
    with open(input_csv_path, "r") as input_file, open(
        output_csv_path, "w", newline=""
    ) as output_file:
        reader = csv.DictReader(input_file)
        fieldnames = ["conversation", "manner_violation"]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            parent_text = get_parent_text(row["parent"])
            conversation = (
                f"first_comment: {parent_text} ; second_comment: {row['text']}"
            )
            manner_violation = get_manner_class(row)

            writer.writerow(
                {"conversation": conversation, "manner_violation": manner_violation}
            )


create_new_csv(INPUT_FILE_PATH, NEW_CSV_PATH)


def load_dataset():
    # Read the dataset from CSV file
    dataset_ecommerce_pandas = pd.read_csv(NEW_CSV_PATH)
    # print(type(dataset_ecommerce_pandas))

    # Create a new column for labels and text
    dataset_ecommerce_pandas["label"] = dataset_ecommerce_pandas[
        "manner_violation"
    ].astype(str)
    dataset_ecommerce_pandas["text"] = dataset_ecommerce_pandas["conversation"].astype(
        str
    )
    # print(type( dataset_ecommerce_pandas['label']))

    # Split to train & test
    dataset = Dataset.from_pandas(dataset_ecommerce_pandas)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=0.3)

    # # Uncomment to apply under\over sampling
    # dataset_ecommerce_pandas = dataset["train"]

    # # Separate features (text) and target (labels)
    # X = pd.Series(dataset_ecommerce_pandas["text"])
    # y = pd.Series(dataset_ecommerce_pandas["label"])

    # # Fit and apply the under-sampling technique
    # undersampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
    # X_resampled, y_resampled = undersampler.fit_resample(X.values.reshape(-1,1), y)

    # # Fit and apply the over-sampling technique
    # ros = RandomOverSampler(sampling_strategy="minority", random_state=42)
    # X_resampled, y_resampled = ros.fit_resample(X.values.reshape(-1, 1), y)

    # # Convert the resampled data back to DataFrame
    # resampled_df = pd.DataFrame(
    #     {"text": [t[0] for t in X_resampled], "label": y_resampled}
    # )

    # # Convert the resampled DataFrame to a Hugging Face Dataset
    # resampled_dataset = Dataset.from_pandas(resampled_df)

    # dataset["train"] = resampled_dataset

    return dataset


# Model to fine-tune
MODEL_ID = "google/flan-t5-base"
REPOSITORY_ID = f"manner_violation-text-classification-all-data"

# Load dataset
dataset = load_dataset()

# Load tokenizer of FLAN-t5
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Metric
metric = evaluate.load("f1")

# The maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(
    lambda x: tokenizer(x["text"], truncation=True),
    batched=True,
    remove_columns=["text", "label"],
)
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(
    lambda x: tokenizer(x["label"], truncation=True),
    batched=True,
    remove_columns=["text", "label"],
)
max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
print(f"Max target length: {max_target_length}")


def preprocess_function(sample: Dataset, padding: str = "max_length") -> dict:
    """Preprocess the dataset."""

    # add prefix to the input for t5
    inputs = [item for item in sample["text"]]

    # tokenize inputs
    model_inputs = tokenizer(
        inputs, max_length=max_source_length, padding=padding, truncation=True
    )

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(
        text_target=sample["label"],
        max_length=max_target_length,
        padding=padding,
        truncation=True,
    )

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def postprocess_text(
    preds: List[str], labels: List[str]
) -> Tuple[List[str], List[str]]:
    """helper function to postprocess text"""
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, average="macro"
    )
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return result


def train(training_args, location=REPOSITORY_ID) -> None:
    """Train the model."""

    tokenized_dataset = dataset.map(
        preprocess_function, batched=True, remove_columns=["text", "label"]
    )
    print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

    # load model from the hub
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    nltk.download("punkt")

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
    )

    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["test"]

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # TRAIN
    trainer.train()

    # SAVE
    model.save_pretrained(location)


def classify(texts_to_classify, manner_model):
    """Classify a batch of texts using the model."""
    inputs = tokenizer(
        texts_to_classify,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        outputs = manner_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=2,
            early_stopping=True,
        )

    predictions = [
        tokenizer.decode(output, skip_special_tokens=True) for output in outputs
    ]
    return predictions


def evaluate(model_):
    """Evaluate the model on the test dataset."""
    predictions_list, labels_list = [], []

    batch_size = 16  # Adjust batch size based GPU capacity
    num_batches = len(dataset["test"]) // batch_size + (
        0 if len(dataset["test"]) % batch_size == 0 else 1
    )
    progress_bar = tqdm(total=num_batches, desc="Evaluating")

    for i in range(0, len(dataset["test"]), batch_size):
        batch_texts = dataset["test"]["text"][i : i + batch_size]
        batch_labels = dataset["test"]["label"][i : i + batch_size]

        batch_predictions = classify(batch_texts, model_)

        predictions_list.extend(batch_predictions)
        labels_list.extend([str(label) for label in batch_labels])

        progress_bar.update(1)

    progress_bar.close()
    report = classification_report(labels_list, predictions_list)
    print(report)

    labels_list_i = [int(value == "True") for value in labels_list]
    predictions_list_i = [int(value == "True") for value in predictions_list]
    auc_roc = roc_auc_score(labels_list_i, predictions_list_i)
    print("AUC-ROC Score:", auc_roc)


if __name__ == "__main__":
    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=REPOSITORY_ID,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        fp16=False,  # Overflows with fp16
        learning_rate=3e-4,
        num_train_epochs=3,
        logging_dir=f"{REPOSITORY_ID}/logs",  # logging & evaluation strategies
        logging_strategy="epoch",
        evaluation_strategy="no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="tensorboard",
        push_to_hub=False,
        hub_strategy="every_save",
        hub_model_id=REPOSITORY_ID,
        hub_token=HfFolder.get_token(),
    )

    lr = 3e-4
    batch_size = 16

    # Train model
    location = f"s2s_standart_{lr}_{batch_size}"
    training_args.per_device_train_batch_size = batch_size
    training_args.learning_rate = lr
    print(location)
    train(training_args, location)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    manner_model_ = AutoModelForSeq2SeqLM.from_pretrained(location)
    manner_model_.to("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate
    evaluate(manner_model_)

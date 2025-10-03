import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def concat_list(original_list):
    concatenated_result = []

    for i in range(0, len(original_list), 2):
        # Check if there's a second list to concatenate
        if i + 1 < len(original_list):
            concatenated_result.append(original_list[i] + original_list[i+1])
        else:
            # If there's an odd number of lists, append the last one as is
            concatenated_result.append(original_list[i])
    return concatenated_result


def remove_all_subsequences(main_list, sub_list):
    new_list = list(main_list) # Create a copy to modify
    while True:
        found = False
        n = len(new_list)
        m = len(sub_list)
        for i in range(n - m + 1):
            if new_list[i:i+m] == sub_list:
                new_list = new_list[:i] + new_list[i+m:]
                found = True
                break # Restart search from the beginning of the modified list
        if not found:
            break
    return new_list


def split_list_by_subsequence(main_list, sub_list):
    result = []
    for i in range( len(main_list) - len(sub_list) + 1):
        # Check if the subsequence matches at the current position
        if main_list[i:i + len(sub_list)] == sub_list:
            return [0 for _ in range(i)] + main_list[i + len(sub_list):]
    return result


def filter_function(text):
    return (text["chosen"].count("Human:") == 1) and (text["rejected"].count("Human:") == 1)


def prepare_dataset(dataset, tokenizer, num_examples=3000, batch_size=100):
    """Подготовка датасета Anthropic HH-RLHF"""
    try:
        train_dataset = dataset["train"].filter(filter_function)
        test_dataset = dataset["test"].filter(filter_function)

        train_dataset = train_dataset.select(range(min(num_examples, len(train_dataset))))
        test_dataset = test_dataset.select(range(min(num_examples//15, len(test_dataset))))

        def tokenize_function(examples):
            chosen_texts = []
            for chosen in examples["chosen"]:
                text = chosen.replace("Human: ", "")
                # .replace("Assistant: ", "")
                chosen_texts.append(text)

            rejected_texts = []
            for rejected in examples["rejected"]:
                text = rejected.replace("Human: ", "")
                # .replace("Assistant: ", "")
                rejected_texts.append(text)

            # Токенизация
            chosen_tokenized = tokenizer(
                chosen_texts,
                truncation=True,
                padding="max_length",
                max_length=128,  # Уменьшил для экономии памяти
                return_tensors=None  # Возвращаем списки, а не тензоры
            )
            rejected_tokenized = tokenizer(
                rejected_texts,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors=None
            )

            # TODO Заполнить левуб половину + 2 0ми, чтобы сохранить верную длину
            # TODO Скорее всего нужно захендлить длину с attention mask и ids

            chosen_labels = [split_list_by_subsequence(x, [48902, 25]) for x in chosen_tokenized["input_ids"]]
            rejected_labels = [split_list_by_subsequence(x, [48902, 25]) for x in rejected_tokenized["input_ids"]]

            chosen_input_ids = [remove_all_subsequences(x, [48902, 25]) for x in chosen_tokenized["input_ids"]]
            rejected_input_ids = [remove_all_subsequences(x, [48902, 25]) for x in rejected_tokenized["input_ids"]]

            return {
                "chosen_input_ids": chosen_input_ids,
                "chosen_attention_mask": [el[2:] for el in chosen_tokenized["attention_mask"]],
                "rejected_input_ids": rejected_input_ids,
                "rejected_attention_mask": [el[2:] for el in rejected_tokenized["attention_mask"]],
                "chosen_labels": chosen_labels,
                "rejected_labels": rejected_labels
            }

        tokenized_train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=batch_size,
            remove_columns=train_dataset.column_names
        )
        tokenized_test_dataset = test_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=batch_size,
            remove_columns=test_dataset.column_names
        )
        return tokenized_train_dataset, tokenized_test_dataset

    except Exception as e:
        print(f"Ошибка загрузки датасета: {e}")

def collate_batch(batch):
    """Кастомная функция для коллации батча"""
    max_len = np.max([len(item["chosen_input_ids"]) for item in batch])
    add_pad = [50256] * max_len
    chosen_input_ids = torch.stack(
        [torch.tensor(item["chosen_input_ids"] + [50256] * (max_len - len(item["chosen_input_ids"]))) for item in batch])
    chosen_attention_mask = torch.stack(
        [torch.tensor(item["chosen_attention_mask"] + [1] * (max_len - len(item["chosen_attention_mask"]))) for item in batch])
    chosen_labels = torch.stack(
        [torch.tensor(item["chosen_labels"] + [0] * (max_len - len(item["chosen_labels"]))) for item in batch])

    rejected_input_ids = torch.stack(
        [torch.tensor(item["rejected_input_ids"] + [50256] * (max_len - len(item["rejected_input_ids"]))) for item in batch])
    rejected_attention_mask = torch.stack(
        [torch.tensor(item["rejected_attention_mask"] + [1] * (max_len - len(item["rejected_attention_mask"]))) for item in batch])
    rejected_labels = torch.stack(
        [torch.tensor(item["rejected_labels"] + [0] * (max_len - len(item["rejected_labels"]))) for item in batch])

    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": chosen_attention_mask,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_input_ids,
        "rejected_attention_mask": rejected_attention_mask,
        "rejected_labels": rejected_labels,
    }

def extract_qa(eval_dataset):
    # разделить вопросы и ответы
    questions = []
    answers = []
    for sample in eval_dataset:
        parts = sample.split("Assistant:")
        questions.append(parts[0].split("Human:")[-1].strip())
        answers.append(parts[-1].strip())
    return questions, answers


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

from src.data.load_llama import load_llama
from src.data.prepare_dataset import *
from src.utils.lr_scheduler import create_warmup_scheduler
from src.utils.run import run_training
from src.model.peft import make_adapter_model
from src.utils.win_rating import compete_dpo

def get_basic_parts(model_name = "gpt2",
                    dataset="Anthropic/hh-rlhf",
                    load_path=None
                    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = load_dataset(dataset)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # Уменьшить размер модели
    print(
        f"Number of trainable params in original model:      {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model = make_adapter_model(model).to(device)
    print(
        f"Number of trainable params in model with adapters:  {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if load_path:
        model.load_state_dict(torch.load(load_path, map_location=device)['model_state_dict'])

    ref_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Замораживаем reference model
    for param in ref_model.parameters():
        param.requires_grad = False

    return device, dataset, tokenizer, model, ref_model

def run_train(batch_size = 5,
              beta = 0.1,
              model_name = "gpt2",
              dataset="Anthropic/hh-rlhf",
              num_examples=3000,
              lr = 1e-6,
              warmup_steps = 150,
              num_epochs=300, early_stop_patience=50, num_exp=2, checkpoint_dir = './checkpoints/', load_path=None
        ):
    device, dataset, tokenizer, model, ref_model = get_basic_parts(model_name, dataset, load_path)

    # Prepare dataset
    train_data, test_data = prepare_dataset(dataset, tokenizer, num_examples=num_examples, batch_size=batch_size)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )

    # Prepare env
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    scheduler = create_warmup_scheduler(optimizer, warmup_steps=warmup_steps, target_lr=lr)

    # Run training
    run_training(model, ref_model, train_loader, test_loader, optimizer, scheduler, device, batch_size,
                 checkpoint_dir=checkpoint_dir, num_epochs=num_epochs, early_stop_patience=early_stop_patience, num_exp=num_exp)

    # Save weights
    last_checkpoint_path = os.path.join("./checkpoints/", f"last_model_final.pt")
    torch.save(model.state_dict(), last_checkpoint_path)
    return model

def run_eval(model_name = "gpt2",
                    dataset="Anthropic/hh-rlhf",
             num_test_questions = 5, load_path="./checkpoints/best_model.pth"):
    device, dataset, tokenizer, model, ref_model = get_basic_parts(model_name, dataset, load_path)

    # Evaluate
    eval_dataset = dataset["test"].filter(filter_function)['chosen']
    test_questions, correct_answers = extract_qa(eval_dataset)
    test_questions, correct_answers = test_questions[:num_test_questions], correct_answers[:num_test_questions]

    llm_evaluator = load_llama()
    # Compete with ref model
    compete_answers = False
    base_wins, dpo_wins, wins_history = compete_dpo(test_questions, tokenizer, llm_evaluator, model, device, ref_model, correct_answers,
                                                    compete_answers)
    print("Number of wins of the base model:", base_wins)
    print("Number of wins of the DPO model:", dpo_wins)
    print("Winner:\n", "DPO" if dpo_wins > base_wins else ("Base" if dpo_wins != base_wins else "Both"))

    # Compete with answers
    compete_answers = True
    base_wins, dpo_wins, wins_history = compete_dpo(test_questions, tokenizer, model, ref_model, correct_answers,
                                                    compete_answers)
    print("Number of wins of the chosen in dataset:", base_wins)
    print("Number of wins of the DPO model:", dpo_wins)
    print("Winner:\n", "DPO" if dpo_wins > base_wins else ("Answers" if dpo_wins != base_wins else "Both"))



if __name__ == "__main__":
    run_train()
    run_eval()
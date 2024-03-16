import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartPretrainedModel, BartModel, BartConfig
from transformers import get_linear_schedule_with_warmup
from torch.nn import functional as F
from argparse import ArgumentParser
from tqdm import tqdm
from src.bart_dialog import DialogueModel
from src.dataset import DialogueDataset, collate_fn, read_data
import datetime
import os
import logging
import numpy as np
from time import sleep
torch.autograd.set_detect_anomaly(True)

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# create main save directory
def create_save_directory(save_path, batch_size, num_epochs):
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_dir = f"{save_path}/bs_{batch_size}_epochs_{num_epochs}_{timestamp}"
    
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    return main_dir

# loss function for auxiliary strategy representation task
def auxiliary_loss_function(strategy_logits, true_strategy_ids):
    return nn.functional.cross_entropy(strategy_logits, true_strategy_ids)

# training function
def train_epoch(model, dataloader, optimizer, device, config, l_strategy):
    model.train()
    total_loss = 0
    total_loss_gen = 0
    total_loss_aux = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        decoder_attention_mask = batch['decoder_attention_mask'].to(device)
        labels = batch['targets'].to(device)
        
        strategy_ids = batch['strategy_ids']
        response_memory = batch['response_memory']
        strategy_memory = batch['strategy_memory']
        motivation_ids = batch['motivation_ids'].to(device)
        motivation_mask = batch['motivation_mask'].to(device)
        
        logits, strategy_logits, memory_strategy_ids = model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, strategy_ids, response_memory, strategy_memory, motivation_ids, motivation_mask, stage='train')
        
        loss_gen = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=config.pad_token_id)
        loss_aux = auxiliary_loss_function(strategy_logits.view(-1, strategy_logits.size(-1)), torch.tensor(memory_strategy_ids, dtype=torch.long, device=device).view(-1))
        
        loss = loss_gen + l_strategy * loss_aux 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_loss_gen += loss_gen.item()
        total_loss_aux += loss_aux.item()
        
    return total_loss / len(dataloader), total_loss_gen / len(dataloader), total_loss_aux / len(strategy_ids)

# validation function
def val_epoch(model, dataloader, device, config):
    model.eval()
    total_loss = 0
    total_words = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating Perplexity"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            decoder_attention_mask = batch['decoder_attention_mask'].to(device)
            labels = batch['targets'].to(device)
            
            strategy_ids = batch['strategy_ids']
            response_memory = batch['response_memory']
            strategy_memory = batch['strategy_memory']
            motivation_ids = batch['motivation_ids'].to(device)
            motivation_mask = batch['motivation_mask'].to(device)
            
            logits = model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, strategy_ids, response_memory, strategy_memory, motivation_ids, motivation_mask, stage='val')
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=config.pad_token_id)
        #     total_loss += loss.item()
        # return total_loss / len(dataloader)
            
            total_loss += loss.item() * labels.size(0)
            total_words += torch.sum(labels != config.pad_token_id).item()
    perplexity = np.exp(total_loss / total_words)
    return perplexity.item()
            


def save_model(model, tokenizer, main_dir, epoch, val_ppl):
    # create epoch directory
    epoch_dir = f"{main_dir}/epoch_{epoch}_val_ppl_{round(val_ppl,2)}"
    
    if not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir)
    
    # save model, tokenizer, and memory bank
    torch.save(model.state_dict(), epoch_dir + "/dialogue_model.pth")
    tokenizer.save_pretrained(epoch_dir)
    torch.save(model.memory_bank, epoch_dir + "/memory_bank.pth")
    
    print(f"Model saved to {epoch_dir}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--train_data", type=str, default='data/train.json', help="Path to training data")
    parser.add_argument("--val_data", type=str, default='data/valid.json', help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default='out_model/', help="Output directory for model and tokenizer")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="learning_rate")
    parser.add_argument("--warmup_steps", type=int, default=0, help="warmup_steps")
    parser.add_argument("--bart_base_dir", type=str, default='src/bart_base/', help="bart_base_dir")
    parser.add_argument("--l_strategy", type=float, default=0.2, help="l_strategy")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--strategy_memory_max", type=int, default=128, help="Max num in the strategy memory")
    parser.add_argument("--max_length_res_memory", type=int, default=64, help="Max sequence length for response in strategy memory")
    args = parser.parse_args()
    
    main_save_dir = create_save_directory(args.output_dir, args.batch_size, args.num_epochs)
    
    # set log_file directory
    log_file = main_save_dir + '/experiments.log'
    setup_logging(log_file)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BartTokenizer.from_pretrained(args.bart_base_dir)
    
    
    config = BartConfig.from_pretrained(args.bart_base_dir)
    
    train_data = read_data(args.train_data)
    val_data = read_data(args.val_data)
    
    train_dataset = DialogueDataset(train_data, tokenizer, args.max_length)
    val_dataset = DialogueDataset(val_data, tokenizer, args.max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    
    bart_model = BartModel.from_pretrained(args.bart_base_dir)
    #model
    model = DialogueModel(config, bart_model.config, tokenizer, args.max_length, args.strategy_memory_max, args.max_length_res_memory, device)
    #load pretrained bart checkpoints
    model.encoder.load_state_dict(bart_model.encoder.state_dict(), strict=False)
    model.decoder.load_state_dict(bart_model.decoder.state_dict(), strict=False)
    model.strategy_encoder.load_state_dict(bart_model.encoder.state_dict(), strict=False)
    
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=len(train_dataloader) * args.num_epochs)

    logging.info(f"Model: {model}")
    logging.info(f"train_data: {args.train_data}")
    logging.info(f"val_data: {args.val_data}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Main_save_dir: {main_save_dir}")
    logging.info(f"learning_rate: {args.learning_rate}")
    logging.info(f"warmup_steps: {args.warmup_steps}")
    logging.info(f"l_strategy: {args.l_strategy}")
    logging.info(f"max_length: {args.max_length}")
    logging.info(f"strategy_memory_max: {args.strategy_memory_max}")
    logging.info(f"max_length_res_memory: {args.max_length_res_memory}")
    logging.info(f"Size-train_data: {len(train_data)}")
    logging.info(f"Size-val_data: {len(val_data)}")
    
    best_val_ppl = float('inf')
    patience = 0

    for epoch in range(args.num_epochs):
        logging.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        
        train_loss,train_loss_gen, train_loss_aux = train_epoch(model, train_dataloader, optimizer, device, config, args.l_strategy)
        logging.info(f"Training loss: {train_loss}")
        logging.info(f"Training loss_gen: {train_loss_gen}")
        logging.info(f"Training loss_aux: {train_loss_aux}")
        
        val_perplexity = val_epoch(model, val_dataloader, device, config)
        logging.info(f"Validation perplexity: {val_perplexity}")
        
        if val_perplexity < best_val_ppl:
            best_val_ppl = val_perplexity
            save_model(model, tokenizer, main_save_dir, epoch, best_val_ppl)
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                logging.info("Early stopping")
                break
        scheduler.step()

if __name__ == "__main__":
    main()

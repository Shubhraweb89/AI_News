import torch
import torch.nn as nn  # This is the missing import
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from pathlib import Path
import logging
from tqdm import tqdm
from datetime import datetime

# --------------------------
# 🛠️ CONFIGURATION
# --------------------------
MODEL_PATH = "bart_summarizer_with_rl"
FEEDBACK_CSV = "feedback_data/feedback_log.csv"
RETRAINED_MODEL_PATH = "bart_summarizer_with_rl_retrained"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 5e-5
PPO_CLIP = 0.2
ENTROPY_COEF = 0.01
FEEDBACK_THRESHOLD = 10

# Setup logging
logging.basicConfig(
    filename='retraining.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --------------------------
# 🧠 REWARD MODEL
# --------------------------
class RewardModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))

# --------------------------
# 🔄 RETRAINING UTILITIES
# --------------------------
def _init_weights(module):
    """Initialize weights for the reward model"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()

def train_basic_reward_model():
    """Train a simple initial reward model if none exists"""
    try:
        df = pd.read_csv(FEEDBACK_CSV)
        if len(df) < 10:  # Minimum samples needed
            return False
            
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        reward_model = RewardModel().to(DEVICE)
        optimizer = AdamW(reward_model.parameters(), lr=1e-4)
        
        # Simple training loop
        for epoch in range(3):  # Fewer epochs for initial training
            for _, row in df.iterrows():
                inputs = tokenizer(
                    f"{row['article']} [SEP] {row['summary']}", 
                    return_tensors='pt'
                ).to(DEVICE)
                target = torch.tensor([row['feedback']], dtype=torch.float).to(DEVICE)
                
                outputs = reward_model(inputs['input_ids'].float())
                loss = F.binary_cross_entropy(outputs, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        torch.save(reward_model.state_dict(), "reward_model.pth")
        return True
    except Exception as e:
        logging.error(f"Basic reward model training failed: {str(e)}")
        return False

# --------------------------
# 🔄 RETRAINING FUNCTION
# --------------------------
def retrain_model():
    """Main retraining function with PPO"""
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
        
        # Initialize reward model with proper weight initialization
        reward_model = RewardModel().to(DEVICE)
        if os.path.exists("reward_model.pth"):
            try:
                reward_model.load_state_dict(torch.load("reward_model.pth", map_location=DEVICE))
                logging.info("Successfully loaded reward model")
            except Exception as e:
                logging.warning(f"Failed to load reward model: {str(e)}. Initializing new one.")
                reward_model.apply(_init_weights)
        else:
            logging.warning("No reward_model.pth found. Initializing new reward model.")
            reward_model.apply(_init_weights)
            train_basic_reward_model()  # Train initial reward model
        
        # Load feedback data
        if not os.path.exists(FEEDBACK_CSV):
            logging.error("Feedback file not found")
            return False
            
        df = pd.read_csv(FEEDBACK_CSV)
        positive_feedback = df[df['feedback'] == 1]
        if len(positive_feedback) < FEEDBACK_THRESHOLD:
            logging.info(f"Not enough positive feedbacks ({len(positive_feedback)}/{FEEDBACK_THRESHOLD})")
            return False
            
        # Prepare dataloader
        dataset = FeedbackDataset(tokenizer, positive_feedback)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Setup optimizer
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=len(dataloader) * EPOCHS
        )
        
        # Training loop
        model.train()
        reward_model.train()
        logging.info("Starting retraining process...")
        
        for epoch in range(EPOCHS):
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            
            for batch in progress_bar:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                
                try:
                    # PPO training step would go here
                    # For now using basic training
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    loss = outputs.loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    epoch_loss += loss.item()
                    progress_bar.set_postfix(loss=loss.item())
                except Exception as e:
                    logging.error(f"Error in batch processing: {str(e)}")
                    continue
            
            # Save checkpoint
            torch.save(reward_model.state_dict(), f"reward_model_epoch_{epoch+1}.pth")
            logging.info(f"Epoch {epoch+1} completed. Avg loss: {epoch_loss/len(dataloader)}")
        
        # Save final models
        model.save_pretrained(RETRAINED_MODEL_PATH)
        tokenizer.save_pretrained(RETRAINED_MODEL_PATH)
        torch.save(reward_model.state_dict(), "reward_model.pth")
        
        # Archive feedback data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = Path("feedback_data/archive")
        archive_dir.mkdir(exist_ok=True)
        os.rename(FEEDBACK_CSV, archive_dir / f"feedback_{timestamp}.csv")
        
        logging.info("Retraining completed successfully!")
        return True
        
    except Exception as e:
        logging.error(f"Retraining failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    if os.path.exists(FEEDBACK_CSV):
        df = pd.read_csv(FEEDBACK_CSV)
        positive_count = len(df[df['feedback'] == 1])
        
        if positive_count >= FEEDBACK_THRESHOLD:
            print(f"Starting retraining with {positive_count} positive feedbacks")
            if retrain_model():
                print("✅ Retraining successful!")
            else:
                print("❌ Retraining failed - check retraining.log")
        else:
            print(f"Need {FEEDBACK_THRESHOLD} positive feedbacks (have {positive_count})")
    else:
        print("No feedback data found at:", FEEDBACK_CSV)
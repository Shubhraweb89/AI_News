# File: train_reward_model.py
import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import logging

# Configuration
FEEDBACK_CSV = "feedback_data/feedback_log.csv"
MODEL_SAVE_PATH = "reward_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-3

# Setup logging
logging.basicConfig(
    filename='reward_model_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RewardDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = f"{row['article']} [SEP] {row['summary']}"
        label = row['feedback']
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze().to(torch.long),
            'attention_mask': inputs['attention_mask'].squeeze().to(torch.long),
            'label': torch.tensor(label, dtype=torch.float32)
        }

class RewardModel(nn.Module):
    def __init__(self, base_model_name="bert-base-uncased", hidden_size=128):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.fc = nn.Linear(self.base_model.config.hidden_size, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask):
        # Get embeddings from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Pass through classifier
        x = self.dropout(pooled_output)
        return torch.sigmoid(self.fc(x))

def train_reward_model():
    try:
        # Load data
        df = pd.read_csv(FEEDBACK_CSV)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Prepare dataset
        dataset = RewardDataset(df, tokenizer)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # Initialize model
        model = RewardModel().to(DEVICE)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Training loop
        model.train()
        logging.info("Starting reward model training...")
        
        for epoch in range(EPOCHS):
            epoch_loss = 0
            for batch in dataloader:
                inputs = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                
                # Forward pass
                outputs = model(inputs, attention_mask)
                loss = criterion(outputs.squeeze(), labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            logging.info(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss/len(dataloader)}")
        
        # Save model
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        logging.info(f"Reward model saved to {MODEL_SAVE_PATH}")
        return True
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    if Path(FEEDBACK_CSV).exists():
        print("Training reward model...")
        if train_reward_model():
            print("✅ Reward model trained successfully!")
        else:
            print("❌ Reward model training failed - check reward_model_training.log")
    else:
        print(f"Error: Feedback file not found at {FEEDBACK_CSV}")
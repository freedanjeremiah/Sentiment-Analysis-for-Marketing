import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

# Load the Twitter airline sentiment dataset
df = pd.read_csv("/content/Tweets.csv")

# Map the sentiment labels to numerical values (e.g., 'positive' to 2, 'neutral' to 1, 'negative' to 0)
sentiment_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
df['label'] = df['airline_sentiment'].map(sentiment_mapping)

# Split the dataset into training, validation, and test sets
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Initialize the BERT tokenizer and encode the text data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def encode_text(texts, tokenizer, max_length):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
                            text,                      # Text to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_length,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks

max_length = 128
train_input_ids, train_attention_masks = encode_text(train_df['text'], tokenizer, max_length)
val_input_ids, val_attention_masks = encode_text(val_df['text'], tokenizer, max_length)
test_input_ids, test_attention_masks = encode_text(test_df['text'], tokenizer, max_length)

# Create PyTorch DataLoader for training, validation, and test datasets
batch_size = 32
train_labels = torch.tensor(train_df['label'].values)
val_labels = torch.tensor(val_df['label'].values)
test_labels = torch.tensor(test_df['label'].values)

train_data = TensorDataset(train_input_ids, train_attention_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_input_ids, val_attention_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

test_data = TensorDataset(test_input_ids, test_attention_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# Initialize the BERT model for sentiment classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
num_epochs = 1
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Fine-tune the model
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}'):
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        model.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Average training loss for epoch {epoch + 1}: {avg_train_loss:.4f}')

# Evaluate the model on the validation set
model.eval()
val_preds, val_true_labels = [], []

for batch in tqdm(val_dataloader, desc='Validation'):
    input_ids, attention_mask, labels = batch
    input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
    true_labels = labels.detach().cpu().numpy()

    val_preds.extend(preds)
    val_true_labels.extend(true_labels)

val_accuracy = accuracy_score(val_true_labels, val_preds)
print(f'Validation Accuracy: {val_accuracy:.4f}')
print(classification_report(val_true_labels, val_preds))

# Evaluate the model on the test set
test_preds, test_true_labels = [], []

for batch in tqdm(test_dataloader, desc='Testing'):
    input_ids, attention_mask, labels = batch
    input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
    true_labels = labels.detach().cpu().numpy()

    test_preds.extend(preds)
    test_true_labels.extend(true_labels)

test_accuracy = accuracy_score(test_true_labels, test_preds)
print(f'Test Accuracy: {test_accuracy:.4f}')
print(classification_report(test_true_labels, test_preds))

import torch
import transformers

# Set device to GPU if available, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load XLM-Roberta pre-trained model and tokenizer
model_name = 'xlm-roberta-base'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# Define training data
train_texts = ['text 1', 'text 2' ]  # list of training texts
train_labels = [0, 1 ]  # list of binary labels corresponding to each text

# Tokenize training texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)

# Convert encoded inputs and labels into PyTorch tensors
train_inputs = torch.tensor(train_encodings['input_ids']).to(device)
train_masks = torch.tensor(train_encodings['attention_mask']).to(device)
train_labels = torch.tensor(train_labels).to(device)

# Define optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_inputs)*10)

# Define loss function and evaluation metric
loss_fn = torch.nn.CrossEntropyLoss()
metric_fn = torch.nn.functional.binary_cross_entropy_with_logits

# Train model
model.train()
for epoch in range(10):
    total_loss = 0
    for i in range(0, len(train_inputs), 32):
        optimizer.zero_grad()
        outputs = model(input_ids=train_inputs[i:i+32], attention_mask=train_masks[i:i+32], labels=None)
        logits = outputs.logits
        loss = loss_fn(logits.view(-1, 2), train_labels[i:i+32])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    avg_loss = total_loss / (len(train_inputs) / 32)
    print(f'Epoch {epoch+1} loss: {avg_loss:.3f}')

# Save trained model
torch.save(model.state_dict(), 'xlmr_binary_classification.pth')

# Load saved model
model.load_state_dict(torch.load('xlmr_binary_classification.pth', map_location=device))

# Define test data
test_text = 'this is a test text'
test_encoding = tokenizer(test_text, truncation=True, padding=True, max_length=512, return_tensors='pt').to(device)

# Predict label and probability
model.eval()
with torch.no_grad():
    outputs = model(**test_encoding)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    pred_label = torch.argmax(logits, dim=1).item()
    proba = probs[0][pred_label].item()

# Print prediction and probability
print(f'Test text: {test_text}')
print(f'Predicted label: {pred_label}')
print(f'Probability: {proba:.3f}')

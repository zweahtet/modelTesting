import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Define constants
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 1e-5
MAX_LENGTH = 512
MODEL_NAME = 'gpt2'
TRAIN_FILE = './data/calregs.txt'

# Define dataset class
class RegulationsDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.tokenizer = tokenizer
        self.input_ids = []
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            # Split text into sentences
            sentences = [s.strip() for s in text.split('.') if len(s) > 0]
            for sentence in sentences:
                # Encode sentence as input_ids and truncate to max length
                encoded = tokenizer.encode(sentence, max_length=MAX_LENGTH, truncation=True)
                self.input_ids.append(torch.tensor(encoded))
    
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx]

# Define collate function
def collate_fn(batch):
    # Pad batch to max length
    input_ids = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    # Create attention mask
    attention_mask = torch.where(input_ids != 0, torch.tensor(1), torch.tensor(0))
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# Freeze all layers except the last one
# for param in model.parameters():
#     param.requires_grad = False
# model.transformer.h[-1].requires_grad = True
# model.lm_head.requires_grad = True

# Prepare data
train_dataset = RegulationsDataset(TRAIN_FILE, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()


# specify device to use a GPU if you have access to one. Otherwise, 
# training on a CPU may take several hours instead of a couple of minutes.
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Train the model
for epoch in range(EPOCHS):
    running_loss = 0.0
    for batch in train_loader:
        # optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = loss_fn(outputs.logits.view(-1, tokenizer.vocab_size), input_ids.view(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * input_ids.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch {epoch+1}/{EPOCHS}: loss={epoch_loss:.4f}')

# Define function to generate responses
def generate_response(text):
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    output = model.generate(input_ids=input_ids, max_length=MAX_LENGTH, do_sample=True, temperature=0.7)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

from accelerate import Accelerator


def training_loop(mixed_precision="fp16", seed: int = 42, batch_size: int = 64):
    set_seed(seed)
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision=mixed_precision)

    # Freeze the base model
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True

    # You can normalize the batches of images to be a bit faster
    mean = torch.tensor(model.default_cfg["mean"])[None, :, None, None]
    std = torch.tensor(model.default_cfg["std"])[None, :, None, None]

    # To make these constants available on the active device, set it to the accelerator device
    mean = mean.to(accelerator.device)
    std = std.to(accelerator.device)

    # Intantiate the optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-2 / 25)

    # Instantiate the learning rate scheduler
    # lr_scheduler = OneCycleLR(optimizer=optimizer, max_lr=3e-2, epochs=5, steps_per_epoch=len(data_loader))

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the objects in the same order you gave them to the
    # prepare method.
    model, optimizer, data_loader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, data_loader, eval_dataloader, lr_scheduler
    )

    # Now you train the model
    for epoch in range(5):
        model.train()
        for batch in data_loader:
            inputs = (batch["image"] - mean) / std
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, batch["label"])
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # model.eval()
        # accurate = 0
        # num_elems = 0
        # for batch in eval_dataloader:
        #     inputs = (batch["image"] - mean) / std
        #     with torch.no_grad():
        #         outputs = model(inputs)
        #     predictions = outputs.argmax(dim=-1)
        #     accurate_preds = accelerator.gather(predictions) == accelerator.gather(batch["label"])
        #     num_elems += accurate_preds.shape[0]
        #     accurate += accurate_preds.long().sum()

        # eval_metric = accurate.item() / num_elems
        # # Use accelerator.print to print only on the main process.
        # accelerator.print(f"epoch {epoch}: {100 * eval_metric:.2f}")

# Test the model
while True:
    text = input('User: ')
    response = generate_response(text)
    print(f'Bot:', response)

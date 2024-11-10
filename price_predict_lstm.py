import pandas as pd
import torch
import data_loader
import torch.utils.data as data_utils
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

combined_data = pd.read_parquet("combined_processed_data.parquet")

num_ticker = 9

# this function return the num_ticker rows after the start_row, where start_row - num_ticker : start_row is the input
def get_label_rolling(start_row):
    # if starting from start_row, there are not num_ticker rows left, then return None
    if start_row + num_ticker >= len(combined_data):
        return None
    
    clipped_data = combined_data.iloc[start_row:start_row + num_ticker]

    # if starting from start_row, the period is 1, but 2*num_ticker rows later, the period is 0, then return None
    if clipped_data['period'].iloc[0] == 1 and clipped_data['period'].iloc[-1] == 0:
        return None
    
    # drop the session_start
    clipped_data = clipped_data.drop(columns=['session_start'])

    # # since news is always n elements, we can expand it into n columns
    # clipped_data = clipped_data.join(pd.DataFrame(clipped_data['news'].tolist())).drop(columns=['news'])

    # drop the news column since we are not insterested in predicting the news
    clipped_data = clipped_data.drop(columns=['news'])

    # drop the period and ticker columns
    clipped_data = clipped_data.drop(columns=['period', 'ticker'])

    # labal is the news, period, ticker, ask_price of each security
    label = torch.tensor(clipped_data.values, dtype=torch.float32, device=device)

    # flatten the tensor
    label = label.reshape(-1)

    return label


# the input tensor is simply the same data as the label
# input: start_row - num_ticker : start_row
# output: start_row : start_row + num_ticker
def make_input_tensor(trainable_data_list, prediction = False):
    if prediction:
        df = pd.DataFrame(trainable_data_list)
        # apply the news tensor function to the news column
        df['news'] = df['news'].apply(data_loader.get_news_tensor)
    else:
        df = trainable_data_list
    # drop the session_start column
    df = df.drop(columns=['session_start'])

    # if the length of the dataframe is less than num_ticker, then return None
    if len(df) < num_ticker:
        return None
    
    # if the period is 1 at the start but 0 at the end, then return None
    if df['period'].iloc[0] == 1 and df['period'].iloc[-1] == 0:
        return None

    # for news, since it is always n elements, we can expand it into n columns
    df = df.join(pd.DataFrame(df['news'].tolist())).drop(columns=['news'])

    # drop the period and ticker columns
    df = df.drop(columns=['period', 'ticker'])

    # turn the dataframe into a tensor
    input_tensor = torch.tensor(df.values, dtype=torch.float32, device=device)
    # flatten the tensor
    input_tensor = input_tensor.reshape(-1)
    return input_tensor


inputs = []
labels = []


# iterate every row in the combined data, and get the input tensor and label tensor
for i in tqdm(range(num_ticker, 2000)):
    input_tensor = make_input_tensor(combined_data.iloc[i - num_ticker:i])
    label = get_label_rolling(i)
    if input_tensor is not None and label is not None:
        inputs.append(input_tensor)
        labels.append(label)


# use torch stack
inputs = torch.stack(inputs)
labels = torch.stack(labels)


dataset = data_utils.TensorDataset(inputs, labels)

train_data, valid_data = data_utils.random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])

train_loader = data_utils.DataLoader(train_data, batch_size=32, shuffle=True)

input_dim = inputs.shape

output_dim = labels.shape

# 3537, 54
# the input dimension is too much, we cannot process without reducing the dimension
print(input_dim, output_dim)


class DeepLSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256, 128, 64], num_layers=4, dropout=0.2):
        super(DeepLSTMPredictor, self).__init__()
        
        # Calculate features per timestep (total input size / num_ticker)
        self.features_per_step = input_size // num_ticker
        self.num_ticker = num_ticker
        
        # Stack of LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=self.features_per_step,
            hidden_size=hidden_sizes[0],
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers with batch normalization and dropout
        self.fc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                nn.BatchNorm1d(num_ticker),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for i in range(len(hidden_sizes)-1)
        ])
        
        # Final output layer
        self.fc_out = nn.Linear(hidden_sizes[-1], self.features_per_step)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape input to (batch, timesteps, features_per_step)
        x = x.view(batch_size, self.num_ticker, self.features_per_step)
        
        # First LSTM layer
        lstm_out, _ = self.lstm1(x)
        
        # Pass through all FC layers
        out = lstm_out
        for fc_layer in self.fc_layers:
            out = fc_layer(out)
            
        # Final output layer
        out = self.fc_out(out)
        
        # Reshape output to match target shape
        out = out.reshape(batch_size, -1)
        
        return out

# Initialize model with proper input size
model = DeepLSTMPredictor(
    input_size=inputs.shape[1],  # Total input size
    hidden_sizes=[2048, 4096, 4096, 2048],
    num_layers=4,
    dropout=0.2
).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=5, 
    verbose=True
)

# Training function with validation
def train_model(model, train_loader, valid_data, criterion, optimizer, scheduler, num_epochs=50):
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        # display the average training loss
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')
        
        # Validation phase
        model.eval()
        valid_loss = 0
        valid_loader = data_utils.DataLoader(valid_data, batch_size=32, shuffle=False)
        
        with torch.no_grad():
            for data, target in valid_loader:
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()
        
        avg_valid_loss = valid_loss / len(valid_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Valid Loss: {avg_valid_loss:.4f}')
        
        # Learning rate scheduling
        scheduler.step(avg_valid_loss)
        
        # Save best model
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {avg_train_loss:.4f}')
            print(f'Valid Loss: {avg_valid_loss:.4f}')
            print('-' * 50)

# Train the model
train_model(model, train_loader, valid_data, criterion, optimizer, scheduler)

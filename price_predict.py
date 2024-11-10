import pandas as pd
import torch
import data_loader
import torch.utils.data as data_utils
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

combined_data = pd.read_parquet("combined_processed_data.parquet")

# get all non-None news row indices
news_row_indices = combined_data[combined_data['news'].notnull()].index

# this helper function returns the label of a news tensor at a given index
# the label is the tensor of 6 securities price change for the next n periods
def get_label(news_tensor_row_index, num_ticker):
    # if starting from the new_tensor_row_index, there are not num_ticker rows left, then return None
    if news_tensor_row_index + num_ticker >= len(combined_data):
        return None
    
    # clip the combined data to the next num_ticker rows starting from the news_tensor_row_index
    clipped_data = combined_data.iloc[news_tensor_row_index:news_tensor_row_index+num_ticker]

    # if the clipped data starts with a period being 1 but changed to 0 at the end, then return None
    if clipped_data['period'].iloc[0] == 1 and clipped_data['period'].iloc[-1] == 0:
        return None
    
    # drop the news, period and ticker columns
    clipped_data = clipped_data.drop(columns=['session_start', 'news', 'period', 'ticker'])

    # calculate the accumulated returns by compounding the percentage changes
    accumulated_return = (clipped_data + 1).prod() - 1

    # turn the accumulated return into a tensor
    accumulated_return = torch.tensor(accumulated_return.values, dtype=torch.float32, device=device)

    # turn the price change of 6 securities into a tensor
    label = torch.tensor(clipped_data.values, dtype=torch.float32, device=device)

    # flatten the tensor
    label = label.reshape(-1)

    # attach the accumulated return tensor to the label tensor
    label = torch.cat([label, accumulated_return])

    return label



def make_input_tensor(period, ticker, news):
    input_tensor = torch.tensor([period, ticker], dtype=torch.float32, device=device)
    news = torch.tensor(news, dtype=torch.float32, device=device)
    input_tensor = torch.cat([input_tensor, news])

    return input_tensor


tickers_period = 15

inputs = torch.Tensor([]).to(device)
labels = torch.Tensor([]).to(device)

# the training data is the period + thicker + news tensor: label
for news_row_index in news_row_indices:
    # get the label for the news tensor at the news_row_index
    label = get_label(news_row_index, tickers_period)

    # get the period, ticker and news tensor at the news_row_index
    period = combined_data['period'].iloc[news_row_index]
    ticker = combined_data['ticker'].iloc[news_row_index]
    news = combined_data['news'].iloc[news_row_index]
    # append the period, ticker, news tensor and label to the training data

    input_tensor = make_input_tensor(period, ticker, news)
    
    # append the input tensor to the inputs tensor, do not attah, I want tensor inside tensor
    inputs = torch.cat([inputs, input_tensor.unsqueeze(0)])
    labels = torch.cat([labels, label.unsqueeze(0)])



test_data = data_utils.TensorDataset(inputs, labels)

train_loader = data_utils.DataLoader(test_data, batch_size=32, shuffle=True)

input_dim = inputs.shape[1]

output_dim = labels.shape[1]


class PricePredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): 
        super(PricePredictionModel, self).__init__()
        
        self.network = nn.Sequential(
            # First layer
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Second layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Third layer
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output layer
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)


def train_model(model, train_loader, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'epoch_loss': [],
        'best_loss': float('inf')
    }
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        # Progress bar for batches
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch in progress_bar:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            batch_count += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / batch_count
        history['epoch_loss'].append(avg_epoch_loss)
        
        # Print epoch results
        print(f'\nEpoch {epoch + 1}/{num_epochs}:')
        print(f'Average Loss: {avg_epoch_loss:.6f}')
    
    return history


def get_prediction(model, period, ticker, news):
    model.eval()

    news = data_loader.get_news_tensor(news)
    
    # Prepare input tensor
    input_tensor = make_input_tensor(period, ticker, news)
    
    # Forward pass
    output = model(input_tensor).detach().cpu().numpy()

    # only keep the last 6 elements of the output tensor
    output = output[-6:].reshape(1, -1)

    security_columns = ['CorpBondA', 'CorpBondB', 'CorpBondC', 'GovtBondY2', 'GovtBondY5', 'GovtBondY10']

    # turn the output into a dataframe
    output = pd.DataFrame(output, columns=security_columns)
    
    return output


def get_model():
    model = PricePredictionModel(input_dim=input_dim, hidden_dim=1024, output_dim=output_dim).to(device)

    model.load_state_dict(torch.load('price_prediction_model.pth'))

    return model


# Create and train the model
def main():
    # Model initialization

    model = PricePredictionModel(input_dim=input_dim, hidden_dim=1024, output_dim=output_dim).to(device)
    
    # Print model summary
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    
    # Train the model
    history = train_model(model, train_loader, num_epochs=50, learning_rate=0.001)

    # Save the model
    torch.save(model.state_dict(), 'price_prediction_model.pth')

    


if __name__ == "__main__":
    main()




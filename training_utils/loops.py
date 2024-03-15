import torch
from tqdm import tqdm
import wandb

def run_testing(model, test_loader, criterion, unsqueeze=False):

    model.eval().to(model.device)

    with torch.no_grad():
        num_correct = 0
        num_samples = 0

        for batch in test_loader:

            inputs, labels = batch
            inputs = inputs.float()
            labels = labels.long()

            inputs = inputs.to(model.device)
            labels = labels.to(model.device)
            
            if unsqueeze:
                outputs = model.forward(inputs.unsqueeze(1))
            else:
                outputs = model.forward(inputs)

            _, predictions = torch.max(outputs, dim=1)
            num_correct += (predictions == labels).sum().item()
            num_samples += len(inputs)

    accuracy = num_correct / num_samples * 100
    return accuracy

def run_eval(model, train_loader, val_loader, criterion, unsqueeze=False):

        model.eval().to(model.device)

        with torch.no_grad():
            val_loss = 0.0
            num_correct_val = 0
            num_samples_val = 0
            num_correct_train = 0
            num_samples_train = 0

            for batch in val_loader:

                inputs, labels = batch
                inputs = inputs.float()
                labels = labels.long()

                inputs = inputs.to(model.device)
                labels = labels.to(model.device)
                
                if unsqueeze:
                    outputs = model.forward(inputs.unsqueeze(1))
                else:
                    outputs = model.forward(inputs)


                val_loss += criterion(outputs, labels)

                _, predictions = torch.max(outputs, dim=1)
                num_correct_val += (predictions == labels).sum().item()
                num_samples_val += len(inputs)

            for batch in train_loader:
                    
                inputs, labels = batch
                inputs = inputs.float()
                labels = labels.long()

                inputs = inputs.to(model.device)
                labels = labels.to(model.device)

                if unsqueeze:
                    outputs = model.forward(inputs.unsqueeze(1))
                else:
                    outputs = model.forward(inputs)

                _, predictions = torch.max(outputs, dim=1)
                num_correct_train += (predictions == labels).sum().item()
                num_samples_train += len(inputs)

        avg_loss = val_loss / len(val_loader)
        accuracy_val = num_correct_val / num_samples_val * 100
        accuracy_train = num_correct_train / num_samples_train * 100

        return avg_loss, accuracy_val, accuracy_train

def run_train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, unsqueeze=False, progress_bar=True, progress=True):
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        train_accuracies = []

        for epoch in range(num_epochs):

            model.train().to(model.device)

            if progress_bar:
                with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}',
                    position=0, leave=True) as pbar:
                    
                    avg_loss = 0
                    for batch in train_loader:
                        inputs, labels = batch
                        inputs = inputs.float()
                        labels = labels.long()

                        inputs = inputs.to(model.device)
                        labels = labels.to(model.device)

                        optimizer.zero_grad()
                        if unsqueeze:
                            outputs = model.forward(inputs.unsqueeze(1))
                        else:
                            outputs = model.forward(inputs)

                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        avg_loss += loss

                        pbar.update(1)
                        pbar.set_postfix(loss=loss.item())

                    train_loss = avg_loss / len(train_loader)
                    train_losses.append(train_loss.to('cpu').detach().numpy())
                    if progress:
                        print(f"Epoch {epoch + 1} - Avg Train Loss: {train_loss:.4f}")
            else:
                avg_loss = 0
                for batch in train_loader:
                    inputs, labels = batch
                    inputs = inputs.float()
                    labels = labels.long()

                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)

                    optimizer.zero_grad()
                    if unsqueeze:
                        outputs = model.forward(inputs.unsqueeze(1))
                    else:
                        outputs = model.forward(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss


                train_loss = avg_loss / len(train_loader)
                train_losses.append(train_loss.to('cpu').detach().numpy())
                
                if progress:
                    print(f"Epoch {epoch + 1} - Avg Train Loss: {train_loss:.4f}")
            
            val_loss, val_accuracy, train_accuracy = run_eval(model, train_loader, val_loader, criterion, unsqueeze=unsqueeze)

            # wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 
            #     'val_accuracy': val_accuracy, 'train_accuracy': train_accuracy})
            
            val_losses.append(val_loss.to('cpu'))
            val_accuracies.append(val_accuracy)
            train_accuracies.append(train_accuracy)
            
            if progress:
                print(f"Epoch {epoch + 1} - Avg Val Loss: {val_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")
        scheduler.step()

        return train_losses, val_losses, train_accuracies, val_accuracies
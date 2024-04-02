def train(model, train_loader, criterion, optimizer, device, num_epochs, start_epoch=0):
    model.to(device)
    model.train()
    losses, epochs, accuracies = [], [], []
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        correct, total = 0, 0
        cool_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}. Training {model.__class__.__name__}', unit='batch')
        for inputs, labels in cool_progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            cool_progress_bar.set_postfix(loss=running_loss / len(train_loader.dataset), acc=correct / total)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        epochs.append(epoch + 1)

        os.makedirs('trained_models/', exist_ok=True)
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': epoch_loss, 'accuracy': epoch_acc}, f'trained_models/{model.__class__.__name__}_epoch_{epoch + 1}.pth')
    return losses, epochs, accuracies

def continue_training(model, train_loader, criterion, optimizer, device, num_epochs, filepath):
    loaded_model = torch.load(filepath)
    model.load_state_dict(loaded_model["model_state_dict"])
    optimizer.load_state_dict(loaded_model["optimizer_state_dict"])
    start_epoch = loaded_model['epoch'] + 1

    losses, epochs, accuracies = train(model, train_loader, criterion, optimizer, device, num_epochs, start_epoch)

    return losses, epochs, accuracies

def validate(model, train_loader, criterion, device, num_epochs, start_epoch=0):
    model.to(device)
    model.eval()
    val_losses, val_epochs, val_accuracies = [], [], []
    with torch.no_grad():
      for epoch in range(start_epoch, num_epochs):
          running_loss = 0.0
          correct, total = 0, 0
          cool_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}. Validating {model.__class__.__name__}', unit='batch')
          for inputs, labels in cool_progress_bar:
              inputs, labels = inputs.to(device), labels.to(device)
              outputs = model(inputs)
              loss = criterion(outputs, labels)
              running_loss += loss.item() * inputs.size(0)
              _, predicted = torch.max(outputs, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()
              cool_progress_bar.set_postfix(loss=running_loss / len(train_loader.dataset), acc=correct / total)
          val_epoch_loss = running_loss / len(train_loader.dataset)
          val_epoch_acc = correct / total
          val_losses.append(val_epoch_loss)
          val_accuracies.append(val_epoch_acc)
          val_epochs.append(1)
    return val_losses, val_epochs, val_accuracies

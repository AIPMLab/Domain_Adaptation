def test(model, target_test_loader, args):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        target_features = []
        true_labels = []
        predicted_labels = []
        for data, target in target_test_loader:
            data, target = data.to(args.device), target.to(args.device)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
            target_features_batch = model.bottleneck_layer(model.base_network(data))
            target_features.append(target_features_batch.cpu().numpy())
            true_labels.extend(target.detach().cpu().numpy())
            predicted_labels.extend(pred.detach().cpu().numpy())

    target_features = np.concatenate(target_features)
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    
    acc = 100. * correct / len_target_dataset
    f1 = f1_score(true_labels, predicted_labels, average="weighted")
    
    return acc, test_loss.avg, f1, target_features

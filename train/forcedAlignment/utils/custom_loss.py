import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, lambbda=0.01, reduction='none', tolerance_window=3, penalized=False, regularization=False):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lambbda = lambbda
        self.reduction = reduction
        self.tolerance_window = tolerance_window  # Tolerance for nearby tokens
        self.penalized_loss = penalized
        self.regularization = regularization

    def forward(self, inputs, targets, masks, model_type=None):
        # Assuming inputs are the raw logits and not softmax probabilities
        if self.regularization:
            l1_reg = self.custom_L1_regularization(inputs, targets, masks, model_type)
        # Convert inputs to probabilities using sigmoid
        inputs = inputs.view(-1) 
        inputs = torch.sigmoid(inputs)
        targets = targets.view(-1)

        #masking
        masks = masks.view(-1)
        inputs = inputs[masks.bool()]
        targets = targets[masks.bool()]

        loss_targets = targets
        bce_loss = F.binary_cross_entropy(inputs, loss_targets.float(), reduction='none')
                
        # Calculate pt: the model's estimated probability for the true class
        pt = torch.where(loss_targets == 1, inputs, 1 - inputs)
        
        # Compute the focal loss
        loss = ((1-pt) ** self.gamma) * bce_loss #instead of 1-pt we give more attention to 1 class
        focal_loss = torch.where(loss_targets == 1, self.alpha*loss, (1-self.alpha)*loss)
        
        # Apply reduction
        if self.reduction == 'mean':
            if self.regularization:
                return focal_loss.mean() + self.lambbda*l1_reg
            else:
                return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        

    def custom_L1_regularization(self, inputs, targets, masks, model_type=None, window_size=7, distance_value=10.0):
        #print("inputs type: ", type(inputs))
        if model_type != 'vgg':
            inputs = inputs.squeeze(-1)
        inputs = torch.sigmoid(inputs)
        
        l1_distances_tot = 0.0
        num_labels = 0
        batch_size, seq_len = inputs.shape

        # Loop over each element in the batch
        for i in range(batch_size):
            # Apply the mask to inputs and targets
            inputs_masked = inputs[i][masks[i].bool()]
            targets_masked = targets[i][masks[i].bool()]

            # Convert inputs_masked to binary (1 if > 0.5)
            inputs_binary = (inputs_masked > 0.5).float()

            # Initialize a tensor to store the L1 distances for the current sequence
            l1_distances = torch.zeros(targets_masked.shape[0])

            # Loop over each target index to find the nearest predicted index
            for j in range(targets_masked.shape[0]):
                if targets_masked[j] == 1.0:  # Only consider targets that are 1
                    num_labels += 1
                    # Define the window range (7 indices to the left and 7 to the right)
                    start_idx = max(0, j - window_size)  # Prevent out-of-bounds access
                    end_idx = min(seq_len, j + window_size + 1)  # Prevent out-of-bounds access

                    # Find the indices within the window where inputs are > 0.5 (binary)
                    input_indices_in_window = torch.where(inputs_binary[start_idx:end_idx] == 1)[0]

                    if len(input_indices_in_window) > 0:
                        # Find the closest input index
                        min_distance = float('inf')
                        for input_idx in input_indices_in_window:
                            distance = abs(j - (start_idx + input_idx))  # L1 distance
                            min_distance = min(min_distance, distance)
                        l1_distances[j] = min_distance
                    else:
                        # No prediction in range, assign a large distance (10)
                        l1_distances[j] = distance_value

            # Sum all L1 distances for the current batch
            l1_distances_tot += l1_distances.sum()
            if num_labels > 0:
                l1_distances_tot = l1_distances_tot/num_labels

        # Return the total loss for the batch
        return l1_distances_tot
    

class CustomRankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        """
        Initialize the custom ranking loss with a margin.
        Args:
            margin (float): Margin value for ranking loss.
        """
        super(CustomRankingLoss, self).__init__()
        self.margin = margin  # margin for the ranking loss
        self.sigmoid = nn.Sigmoid()

    def forward(self, predictions, labels):
        """
        Compute the ranking loss for a batch of predictions and labels.
        Args:
            predictions (Tensor): Model output scores (logits), shape (batch_size,).
            labels (Tensor): Ground truth binary labels, shape (batch_size,).
        Returns:
            Tensor: The computed ranking loss for the batch.
        """
        # Apply sigmoid to get probabilities
        probs = self.sigmoid(predictions)

        # Separate positive and negative samples
        pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
        neg_indices = (labels == 0).nonzero(as_tuple=True)[0]

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            # If no positive or negative samples in the batch, return zero loss
            return torch.tensor(0.0, requires_grad=True).to(predictions.device)

        # Generate all possible positive-negative pairs
        pos_scores = probs[pos_indices]  # Scores for positive samples
        neg_scores = probs[neg_indices]  # Scores for negative samples

        # Create pairs: each positive paired with each negative (broadcasting)
        pos_scores = pos_scores.unsqueeze(1)  # Shape: (num_pos, 1)
        neg_scores = neg_scores.unsqueeze(0)  # Shape: (1, num_neg)

        # Compute pairwise margin loss: max(0, margin - (pos_score - neg_score))
        pairwise_loss = torch.clamp(self.margin - (pos_scores - neg_scores), min=0)

        # Average the loss over all pairs
        loss = pairwise_loss.mean()
        return loss
    


    
# Example Usage
if __name__ == "__main__":
    # Example model predictions and labels
    predictions = torch.tensor([
        [0.2, 0.8, 0.4, 0.9, 0.3],  # Batch 1
        [0.1, 0.5, 0.6, 0.7, 0.2],  # Batch 2
        [0.3, 0.4, 0.1, 0.2, 0.8],  # Batch 3
        [0.9, 0.7, 0.5, 0.3, 0.4],  # Batch 4
        [0.8, 0.1, 0.6, 0.2, 0.7]   # Batch 5
    ], requires_grad=True)

    # Example: 5 batches, each with 5 ground-truth labels
    labels = torch.tensor([
        [0, 1, 0, 1, 0],  # Batch 1
        [0, 0, 1, 1, 0],  # Batch 2
        [0, 0, 0, 0, 1],  # Batch 3
        [1, 1, 0, 0, 0],  # Batch 4
        [1, 0, 1, 0, 1]   # Batch 5
    ])

    # Initialize the custom ranking loss
    loss_fn = CustomRankingLoss(margin=0.5)

    # Compute the loss
    loss = loss_fn(predictions, labels)
    print(f"Ranking Loss: {loss.item()}")
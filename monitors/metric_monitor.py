import torch

class MetricsMonitor:
    def __init__(self, threshold):
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.threshold = threshold

    def accumulate_metrics(self, logits, target):
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).long()
        target = target.view(-1).long()

        self.true_positives += ((preds == 1) & (target == 1)).sum().item()
        self.true_negatives += ((preds == 0) & (target == 0)).sum().item()
        self.false_positives += ((preds == 1) & (target == 0)).sum().item()
        self.false_negatives += ((preds == 0) & (target == 1)).sum().item()
        
    def print_metrics(self):
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        f1 = self.calculate_f1(precision, recall)
        accuracy = self.calculate_accuracy()

        print(f" Precision : {precision} ; Recall: {recall} ; F1 score: {f1} ; Accuracy: {accuracy}")

        self.clear_vals()
    
    def calcualte_accuracy(self):
        total = self.true_negatives + self.true_positives + self.false_positives + self.false_negatives
        correct = self.true_positives + self.true_negatives

        return correct/total

    def calculate_precision(self):
        denom = (self.true_positives + self.false_positives)
        if denom == 0:
            return 0.0
        return self.true_positives/denom

    def calculate_recall(self):
        denom = (self.true_positives + self.false_negatives)
        if denom == 0:
            return 0.0
        return self.true_positives/denom
    
    def calculate_f1(self, precision, recall):
        denom = precision + recall
        if denom == 0:
            return 0.0
        
        return (2 * precision * recall) / denom

    def clear_vals(self):
        self.true_negatives = 0
        self.true_positives = 0
        self.false_negatives = 0
        self.false_positives = 0
    


        

class ValidationLossMonitor:
    def __init__(self, patience):
        self.patience = patience
        self.min_loss = float('inf')
        self.min_loss_epoch = -1
    

    def is_min_loss(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            return True

        return False

    def add_loss(self, loss, epoch):
        if loss < self.min_loss:
            self.min_loss = loss
            self.min_loss_epoch = epoch
            return True
        
        if loss >= self.min_loss:
            if epoch - self.min_loss_epoch > self.patience:
                return False
        
        return True

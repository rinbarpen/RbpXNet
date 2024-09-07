from metrics import scores

class LossCalculator:
    def __call__(self, *args):
        loss = 0.0
        for arg in args:
            loss += arg()
        return loss

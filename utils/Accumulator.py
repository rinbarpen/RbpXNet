
class Accumulator:
    score = 0.0
    count = 0

    def add(self, x):
        self.score += x
        self.count += 1

    def mean(self):
        return self.score / self.count

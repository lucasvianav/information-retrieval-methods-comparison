import math


class Evaluation:
    def __init__(self, returned_set: list, truth_set: list):
        self.returned_set = returned_set
        self.truth_set = truth_set

    def dcg(self):
        gain = [ 1. if doc in self.truth_set else 0. for doc in self.returned_set ]

        dcg = [ gain[0] ]
        for i in range(1, len(gain)):
            dcg.append(gain[i]/math.log2(i+1) + dcg[i-1])

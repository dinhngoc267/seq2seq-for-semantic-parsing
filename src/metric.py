import torchmetrics
import torch


class AccuracyMetric(torchmetrics.Metric):
    def __init__(self):
        super().__init__()

        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        assert len(targets) == len(preds)

        for idx in range(len(targets)):
            if targets[idx] == preds[idx]:
                self.correct += 1

        self.total += len(targets)

    def compute(self):
        return self.correct.float() / self.total.float()

class Precision(torchmetrics.Metric):
    def __init__(self):
        super().__init__()

        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_preds", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred, targets):
        assert len(targets) == len(pred)

        for idx in range(len(targets)):
            ground_truth_entities = targets[idx].split('@')
            pred_entities = [x for x in pred[idx].split('@') if x.strip() != ""]

            self.num_preds += len(set(pred_entities))
            self.total += len(ground_truth_entities)

            for entity in set(pred_entities):
                if entity in ground_truth_entities:
                    self.correct += 1

    def compute(self):
        return self.correct.float() / self.num_preds.float(), self.correct.float() / self.total.float()

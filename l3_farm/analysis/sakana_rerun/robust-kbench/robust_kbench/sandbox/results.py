from typing import Dict


class TestEvalResult(object):
    correct: bool = False
    num_correct: int = 0
    total_correct_trials: int = 0
    max_diff: float = 0.0
    atol: float = 1e-5
    rtol: float = 1e-5

    def __init__(
        self,
        correct: bool,
        num_correct: int,
        total_correct_trials: int,
        max_diff: float,
        atol: float,
        rtol: float,
    ):
        self.correct = correct
        self.num_correct = num_correct
        self.total_correct_trials = total_correct_trials
        self.max_diff = max_diff
        self.atol = atol
        self.rtol = rtol

    def to_dict(self) -> Dict[str, float]:
        return {
            "correct": self.correct,
            "num_correct": self.num_correct,
            "total_correct_trials": self.total_correct_trials,
            "max_diff": self.max_diff,
            "atol": self.atol,
            "rtol": self.rtol,
        }


class TimeEvalResult(object):
    mean_time: float = 0.0
    median_time: float = 0.0
    iqr_time: float = 0.0

    def __init__(
        self,
        mean_time: float,
        median_time: float,
        iqr_time: float,
    ):
        self.mean_time = mean_time
        self.median_time = median_time
        self.iqr_time = iqr_time

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_time": self.mean_time,
            "median_time": self.median_time,
            "iqr_time": self.iqr_time,
        }

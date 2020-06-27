import pandas as pd
import abc

from ExfilPlanner import BaseExfilPlanner

from typing import Optional


class BaseExfilPlannerFactory:
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    @abc.abstractmethod
    def create_exfil_planner(self, baseline_data: Optional[pd.DataFrame] = None) -> BaseExfilPlanner:
        pass

    def __call__(self, baseline_data: Optional[pd.DataFrame] = None) -> BaseExfilPlanner:
        return self.create_exfil_planner(baseline_data)


class ExfilPlannerFactory(BaseExfilPlannerFactory):
    def create_exfil_planner(self, baseline_data: Optional[pd.DataFrame] = None) -> BaseExfilPlanner:
        return self.cls(*self.args, **self.kwargs, baseline_data=baseline_data)

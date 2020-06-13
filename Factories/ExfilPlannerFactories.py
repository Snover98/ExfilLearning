import abc

from ExfilPlanner import BaseExfilPlanner


class BaseExfilPlannerFactory:
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    @abc.abstractmethod
    def create_exfil_planner(self) -> BaseExfilPlanner:
        pass

    def __call__(self, *args, **kwargs) -> BaseExfilPlanner:
        return self.create_exfil_planner()


class ExfilPlannerFactory(BaseExfilPlannerFactory):
    def create_exfil_planner(self) -> BaseExfilPlanner:
        return self.cls(*self.args, **self.kwargs)

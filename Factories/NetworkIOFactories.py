import abc

from NetworkIO import BaseNetworkIO, BaseEnsembleNetworkIO

from typing import List


class BaseNetworkIOFactory:
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    @abc.abstractmethod
    def create_network_io(self) -> BaseNetworkIO:
        pass

    def __call__(self, *args, **kwargs) -> BaseNetworkIO:
        return self.create_network_io()


class NetworkIOFactory(BaseNetworkIOFactory):
    def create_network_io(self) -> BaseNetworkIO:
        return self.cls(*self.args, **self.kwargs)


class EnsembleNetworkIOFactory(BaseNetworkIOFactory):
    def __init__(self, cls, sub_factories: List[BaseNetworkIOFactory], *args, **kwargs):
        super().__init__(cls, *args, **kwargs)
        self.sub_factories = sub_factories

    def create_network_io(self) -> BaseEnsembleNetworkIO:
        sub_network_ios = [sub_factory() for sub_factory in self.sub_factories]
        return self.cls(sub_network_ios, *self.args, **self.kwargs)

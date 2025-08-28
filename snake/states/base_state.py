import abc


class BaseState(abc.ABC):
    def __init__(self, game):
        self.game = game

    @abc.abstractmethod
    def handle_events(self, event): ...
    @abc.abstractmethod
    def update(self, dt): ...
    @abc.abstractmethod
    def draw(self, surface): ...

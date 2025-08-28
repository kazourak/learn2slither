import pygame

from snake.settings import FONT_DIR, IMG_DIR


class AssetLoader:
    def __init__(self):
        self._image_cache = {}
        self._font_cache = {}

    def load_font(self, name):
        if name not in self._font_cache:
            path = FONT_DIR + name
            font = pygame.font.Font(path, 16)
            self._font_cache[name] = font
        return self._font_cache[name]

    def load_image(self, name):
        if name not in self._image_cache:
            path = IMG_DIR + name
            image = pygame.image.load(path)
            self._image_cache[name] = image
        return self._image_cache[name]

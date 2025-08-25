import pygame
import importlib
import sys
from snake.states.base_state import BaseState
from snake.ui.animated_background import AnimatedGridBackground

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 200)
LIGHT_BLUE = (100, 150, 255)
GRAY = (128, 128, 128)
GREEN = (0, 200, 0)


class SettingsState(BaseState):
    def __init__(self, game, settings):
        super().__init__(game)
        self.font = self.game.loader.load_font("8bitoperator_jve.ttf")
        self.title_font = self.game.loader.load_font("8bitoperator_jve.ttf")

        self.settings = settings

        # Convertir le dictionnaire settings en liste d'options
        self.setting_options = [
            {"name": "Vitesse du serpent", "key": "snake_speed", "value": settings["snake_speed"], "min": 10, "max": 60, "step": 5},
            {"name": "Taille de la carte", "key": "map_size", "value": settings["map_size"], "min": 5, "max": 20, "step": 1},
            {"name": "Pommes vertes", "key": "green_apple_nbr", "value": settings["green_apple_nbr"], "min": 1, "max": 10, "step": 1},
            {"name": "Pommes rouges", "key": "red_apple_nbr", "value": settings["red_apple_nbr"], "min": 0, "max": 5, "step": 1},
            {"name": "Longueur initiale", "key": "snake_length", "value": settings["snake_length"], "min": 2, "max": 10, "step": 1}
        ]

        self.options = self.setting_options + [{"name": "Sauvegarder et quitter", "key": "SAVE"}]
        self.selected_index = 0
        self.button_height = 60
        self.button_width = 500
        self.button_spacing = 20
        self.start_y = 180

        self.animated_background = AnimatedGridBackground()
        self.modified = False

    def handle_events(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.selected_index = (self.selected_index - 1) % len(self.options)

            elif event.key == pygame.K_DOWN:
                self.selected_index = (self.selected_index + 1) % len(self.options)

            elif event.key == pygame.K_LEFT and self.selected_index < len(self.setting_options):
                setting = self.setting_options[self.selected_index]
                if setting["value"] > setting["min"]:
                    setting["value"] -= setting["step"]
                    # Mettre à jour aussi le dictionnaire settings original
                    self.settings[setting["key"]] = setting["value"]
                    self.modified = True

            elif event.key == pygame.K_RIGHT and self.selected_index < len(self.setting_options):
                setting = self.setting_options[self.selected_index]
                if setting["value"] < setting["max"]:
                    setting["value"] += setting["step"]
                    # Mettre à jour aussi le dictionnaire settings original
                    self.settings[setting["key"]] = setting["value"]
                    self.modified = True

    def handle_selection(self):
        if self.selected_index == len(self.setting_options):  # Changé de self.settings à self.setting_options
            # self.save_settings()
            self.game.change_state("TITLE")

    # def save_settings(self):
    #     if not self.modified:
    #         return
    #
    #     settings_path = "./snake/settings.py"
    #     try:
    #         with open(settings_path, 'r') as file:
    #             lines = file.readlines()
    #
    #         for i, line in enumerate(lines):
    #             for setting in self.settings:
    #                 if line.strip().startswith(setting["key"]):
    #                     lines[i] = f"{setting['key']} = {setting['value']}\n"
    #
    #         with open(settings_path, 'w') as file:
    #             file.writelines(lines)
    #
    #         # Recharger les paramètres
    #         importlib.reload(sys.modules["snake.settings"])
    #
    #     except Exception as e:
    #         print(f"Erreur lors de la sauvegarde des paramètres: {e}")

    def update(self, dt):
        self.animated_background.update(dt)

    def draw(self, surface):
        self.animated_background.draw(surface)

        # Titre
        title_surface = self.title_font.render("Paramètres", True, WHITE)
        title_rect = title_surface.get_rect(center=(self.settings["screen_width"] // 2, 100))
        surface.blit(title_surface, title_rect)

        # Paramètres
        for i, option in enumerate(self.options):
            x = (self.settings["screen_width"] - self.button_width) // 2
            y = self.start_y + i * (self.button_height + self.button_spacing)
            selected = (i == self.selected_index)

            if i < len(self.setting_options):  # Changé de self.settings à self.setting_options
                self.draw_setting(surface, option, x, y, selected)
            else:
                self.draw_button(surface, option["name"], x, y, selected)

    def draw_setting(self, surface, setting, x, y, selected=False):
        button_color = LIGHT_BLUE if selected else BLUE
        text_color = WHITE

        # Rect
        button_rect = pygame.Rect(x, y, self.button_width, self.button_height)
        pygame.draw.rect(surface, button_color, button_rect)
        pygame.draw.rect(surface, BLACK, button_rect, 3)

        # Selected effect
        if selected:
            pygame.draw.rect(surface, WHITE, button_rect, 5)

        # Text et valeur
        name_surface = self.font.render(f"{setting['name']}: {setting['value']}", True, text_color)
        name_rect = name_surface.get_rect(midleft=(x + 20, y + self.button_height // 2))
        surface.blit(name_surface, name_rect)

        # Flèches
        if selected:
            left_arrow = "◄" if setting["value"] > setting["min"] else " "
            right_arrow = "►" if setting["value"] < setting["max"] else " "

            arrow_font = self.font
            left_surf = arrow_font.render(left_arrow, True, WHITE)
            right_surf = arrow_font.render(right_arrow, True, WHITE)

            surface.blit(left_surf, (x + 10, y + self.button_height // 2 - left_surf.get_height() // 2))
            surface.blit(right_surf, (x + self.button_width - 30, y + self.button_height // 2 - right_surf.get_height() // 2))

    def draw_button(self, surface, text, x, y, selected=False):
        button_color = GREEN if selected else BLUE
        text_color = WHITE

        # Rect
        button_rect = pygame.Rect(x, y, self.button_width, self.button_height)
        pygame.draw.rect(surface, button_color, button_rect)
        pygame.draw.rect(surface, BLACK, button_rect, 3)

        # Selected effect
        if selected:
            pygame.draw.rect(surface, WHITE, button_rect, 5)

        # Text center
        text_surface = self.font.render(text, True, text_color)
        text_rect = text_surface.get_rect(center=button_rect.center)
        surface.blit(text_surface, text_rect)
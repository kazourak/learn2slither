import pygame
from snake.states.base_state import BaseState
from snake.settings import SCREEN_WIDTH
from snake.ui.animated_background import AnimatedGridBackground

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 200)
LIGHT_BLUE = (100, 150, 255)
GRAY = (128, 128, 128)


class TitleState(BaseState):
    def __init__(self, game):
        super().__init__(game)
        self.font = self.game.loader.load_font("8bitoperator_jve.ttf")
        self.menu_buttons = ["Play", "Play with AI", "Settings", "Quit"]
        self.selected_index = 0
        self.button_height = 60
        self.button_width = 300
        self.button_spacing = 20
        self.start_y = 250

        self.animated_background = AnimatedGridBackground()

    def handle_events(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.selected_index = (self.selected_index - 1) % len(self.menu_buttons)

            elif event.key == pygame.K_DOWN:
                self.selected_index = (self.selected_index + 1) % len(self.menu_buttons)

            elif event.key == pygame.K_RETURN:
                self.handle_selection()


    def handle_selection(self):
        selected_option = self.menu_buttons[self.selected_index]

        if selected_option == "Play":
            self.game.change_state("GAME")
        if selected_option == "Quit":
            self.game.running = False


    def draw_logo(self, surface, logo_path, x, y):
        logo_surface = self.game.loader.load_image(logo_path)
        logo_width = int(SCREEN_WIDTH * 0.50)
        logo_ratio = 1920 / 320
        logo_height = int(logo_width / logo_ratio)
        logo_surface = pygame.transform.scale(logo_surface, (logo_width, logo_height))
        logo_rect = logo_surface.get_rect(center=(x, y))
        surface.blit(logo_surface, logo_rect)


    def draw_button(self, surface, text, x, y, selected=False):
        button_color = LIGHT_BLUE if selected else BLUE
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

    def update(self, dt):
        self.animated_background.update(dt)

    def draw(self, surface):
        self.animated_background.draw(surface)
        for i, option in enumerate(self.menu_buttons):
            x = (SCREEN_WIDTH - self.button_width) // 2
            y = self.start_y + i * (self.button_height + self.button_spacing)
            selected = (i == self.selected_index)
            self.draw_button(surface, option, x, y, selected)
        self.draw_logo(surface, "logo.png", SCREEN_WIDTH // 2, 150)
import pygame

SCREEN_WIDTH = 1080
SCREEN_HEIGHT = 720

class AnimatedGridBackground:
    def __init__(self,
                 grid_size=90,
                 grid_speed_1=0.8,
                 grid_speed_2=0.3,
                 grid_color_1=(0, 80, 150),
                 grid_color_2=(0, 40, 100),
                 background_color=(0, 0, 0)):
        """
        Create an animated background with two grids moving in opposite directions.

        Args:
            grid_size: Size of the grid cells
            grid_speed_1: Speed of the first grid (to the right/down)
            grid_speed_2: Speed of the second grid (to the left/up)
            grid_color_1: Color of the first grid
            grid_color_2: Color of the second grid
            background_color: Background color
        """
        self.grid_size = grid_size
        self.grid_speed_1 = grid_speed_1
        self.grid_speed_2 = grid_speed_2
        self.grid_color_1 = grid_color_1
        self.grid_color_2 = grid_color_2
        self.background_color = background_color

        # Grid animation
        self.grid_offset_1 = 0  # For the first grid
        self.grid_offset_2 = 0  # For the second grid

        self.vertical_offset_2 = self.grid_size // 2

    def update(self, dt):
        """Update the grid animations"""
        # Animate the grids - opposite directions
        self.grid_offset_1 += self.grid_speed_1 * dt * 60
        self.grid_offset_2 += self.grid_speed_2 * dt * 60
        # Limit offsets to avoid overflow
        self.grid_offset_1 = self.grid_offset_1 % self.grid_size
        self.grid_offset_2 = self.grid_offset_2 % self.grid_size

    def draw_grid(self, surface, offset_x, offset_y, color, alpha=255):
        """Draw a grid with the given offset"""
        if alpha < 255:
            # Create a temporary surface with alpha for transparency
            temp_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            color_with_alpha = (*color, alpha)

            # Draw vertical lines
            x = offset_x % self.grid_size - self.grid_size
            while x < SCREEN_WIDTH + self.grid_size:
                pygame.draw.line(temp_surface, color_with_alpha, (x, 0), (x, SCREEN_HEIGHT), 1)
                x += self.grid_size

            # Draw horizontal lines
            y = offset_y % self.grid_size - self.grid_size
            while y < SCREEN_HEIGHT + self.grid_size:
                pygame.draw.line(temp_surface, color_with_alpha, (0, y), (SCREEN_WIDTH, y), 1)
                y += self.grid_size

            surface.blit(temp_surface, (0, 0))
        else:
            # Draw directly on the main surface
            # Draw vertical lines
            x = offset_x % self.grid_size - self.grid_size
            while x < SCREEN_WIDTH + self.grid_size:
                pygame.draw.line(surface, color, (x, 0), (x, SCREEN_HEIGHT), 1)
                x += self.grid_size

            # Draw horizontal lines
            y = offset_y % self.grid_size - self.grid_size
            while y < SCREEN_HEIGHT + self.grid_size:
                pygame.draw.line(surface, color, (0, y), (SCREEN_WIDTH, y), 1)
                y += self.grid_size

    def draw(self, surface):
        surface.fill(self.background_color)

        self.draw_grid(surface,
                       -self.grid_offset_2,
                       -self.grid_offset_2 + self.vertical_offset_2,
                       self.grid_color_2)

        self.draw_grid(surface,
                       self.grid_offset_1,
                       self.grid_offset_1,
                       self.grid_color_1)
import pygame
import math
import random

pygame.init()
WIDTH, HEIGHT = 1920, 1080 
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
CENTER = WIDTH // 2, HEIGHT // 2

# Schwarzschild radius mock, scale it
SCHWARZSCHILD_RADIUS = 5

class Particle:
    def __init__(self):
        self.r = SCHWARZSCHILD_RADIUS
        self.theta = random.uniform(0, 2 * math.pi)
        self.v = random.uniform(1.001, 1.01)  # >1 to simulate inverse gravity
        self.color = [random.randint(150, 255) for _ in range(3)]

    def move(self):
        self.r *= self.v  # radial expansion
        self.x = CENTER[0] + self.r * math.cos(self.theta)
        self.y = CENTER[1] + self.r * math.sin(self.theta)

    def draw(self):
        if 0 < self.x < WIDTH and 0 < self.y < HEIGHT:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), 2)

particles = [Particle() for _ in range(150000)]

running = True
while running:
    screen.fill((0, 0, 0))
    for p in particles:
        p.move()
        p.draw()
    pygame.display.flip()
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
pygame.quit()

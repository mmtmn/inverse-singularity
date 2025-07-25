import pygame
import math
import random
import sys

WIDTH, HEIGHT = 1000, 1000
CENTER = WIDTH // 2, HEIGHT // 2
NUM_PARTICLES = 1000
G = 1.0
M = 1.0
DT = 0.04

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

class Particle:
    def __init__(self):
        self.r = 5 + random.uniform(0, 3)
        self.theta = random.uniform(0, 2 * math.pi)
        self.pos = [0.0, 0.0]
        self.vel = [math.cos(self.theta), math.sin(self.theta)]
        self.color = [255, 255, 255]

    def update(self):
        x, y = self.pos
        r = math.sqrt(x**2 + y**2 + 1.0)
        f = 1.0 - 2 * G * M / r  # Normally pulls inward

        # INVERT signal: push outward using the same field
        fx = x / (r * r + 1) * f
        fy = y / (r * r + 1) * f

        self.vel[0] += fx * 0.05
        self.vel[1] += fy * 0.05

        self.pos[0] += self.vel[0] * DT
        self.pos[1] += self.vel[1] * DT

        # Fake blueshift: brighten with r
        glow = min(255, int(r * 6))
        self.color = [glow, glow, glow]

    def draw(self):
        x = int(CENTER[0] + self.pos[0])
        y = int(CENTER[1] + self.pos[1])
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            pygame.draw.circle(screen, self.color, (x, y), 1)

particles = [Particle() for _ in range(NUM_PARTICLES)]

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill((0, 0, 0))
    for p in particles:
        p.update()
        p.draw()

    pygame.display.flip()
    clock.tick(60)

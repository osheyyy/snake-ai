from __future__ import annotations

import random
from collections import namedtuple
from enum import Enum


Point = namedtuple("Point", "x y")


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


DIRECTION_VECTORS = {
    Direction.RIGHT: Point(1, 0),
    Direction.LEFT: Point(-1, 0),
    Direction.UP: Point(0, -1),
    Direction.DOWN: Point(0, 1),
}

FOOD_REWARD = 10.0
DEATH_REWARD = -10.0
MOVE_PENALTY = -0.01
CLOSER_REWARD = 0.05
FARTHER_REWARD = -0.05


class SnakeGame:
    """Grid-based Snake environment.

    The environment can run with or without rendering. Training uses
    render=False, so no Pygame window is opened. Pygame is only initialized
    when render=True, which is used by play.py.
    """

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        block_size: int = 24,
        render: bool = False,
        speed: int = 30,
    ) -> None:
        self.width = width
        self.height = height
        self.block_size = block_size
        self.render = render
        self.speed = speed

        self._pygame = None
        self.display = None
        self.clock = None
        self.font = None

        if self.render:
            self._init_pygame()

        self.reset()

    def _init_pygame(self) -> None:
        import pygame

        pygame.init()
        self._pygame = pygame
        self.display = pygame.display.set_mode(
            (self.width * self.block_size, self.height * self.block_size + 40)
        )
        pygame.display.set_caption("Snake AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 24)

    def reset(self) -> None:
        self.direction = Direction.RIGHT
        self.head = Point(self.width // 2, self.height // 2)
        self.snake = [
            self.head,
            Point(self.head.x - 1, self.head.y),
            Point(self.head.x - 2, self.head.y),
        ]
        self.score = 0
        self.frame_iteration = 0
        self.food = None
        self._place_food()

        if self.render:
            self._update_ui()

    def _place_food(self) -> None:
        while True:
            food = Point(random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if food not in self.snake:
                self.food = food
                return

    def play_step(self, action: int) -> tuple[float, bool, int]:
        """Advance the game by one move.

        Actions:
            0 = keep moving straight
            1 = turn right relative to current direction
            2 = turn left relative to current direction
        """

        if self.render:
            self._handle_events()

        self.frame_iteration += 1

        previous_distance = self._distance_to_food(self.head)
        self._move(action)
        self.snake.insert(0, self.head)

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            return DEATH_REWARD, True, self.score

        reward = MOVE_PENALTY
        if self.head == self.food:
            self.score += 1
            reward += FOOD_REWARD
            self._place_food()
        else:
            self.snake.pop()
            new_distance = self._distance_to_food(self.head)
            reward += CLOSER_REWARD if new_distance < previous_distance else FARTHER_REWARD

        if self.render:
            self._update_ui()
            self.clock.tick(self.speed)

        return reward, False, self.score

    def get_state(self) -> list[int]:
        """Return the required 11-value numeric state vector."""

        point_straight = self._next_point_for_action(0)
        point_right = self._next_point_for_action(1)
        point_left = self._next_point_for_action(2)

        moving_left = self.direction == Direction.LEFT
        moving_right = self.direction == Direction.RIGHT
        moving_up = self.direction == Direction.UP
        moving_down = self.direction == Direction.DOWN

        return [
            int(self.is_collision(point_straight)),
            int(self.is_collision(point_right)),
            int(self.is_collision(point_left)),
            int(moving_left),
            int(moving_right),
            int(moving_up),
            int(moving_down),
            int(self.food.x < self.head.x),
            int(self.food.x > self.head.x),
            int(self.food.y < self.head.y),
            int(self.food.y > self.head.y),
        ]

    def is_collision(self, point: Point | None = None) -> bool:
        if point is None:
            point = self.head

        hits_wall = point.x < 0 or point.x >= self.width or point.y < 0 or point.y >= self.height
        hits_self = point in self.snake[1:]
        return hits_wall or hits_self

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()

    def _distance_to_food(self, point: Point) -> int:
        return abs(point.x - self.food.x) + abs(point.y - self.food.y)

    def _next_point_for_action(self, action: int) -> Point:
        next_direction = self._direction_after_action(action)
        move = DIRECTION_VECTORS[next_direction]
        return Point(self.head.x + move.x, self.head.y + move.y)

    def _direction_after_action(self, action: int) -> Direction:
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        current_index = clockwise.index(self.direction)

        if action == 0:
            return self.direction
        if action == 1:
            return clockwise[(current_index + 1) % len(clockwise)]
        if action == 2:
            return clockwise[(current_index - 1) % len(clockwise)]

        raise ValueError(f"Invalid action {action}. Expected 0, 1, or 2.")

    def _move(self, action: int) -> None:
        self.direction = self._direction_after_action(action)
        move = DIRECTION_VECTORS[self.direction]
        self.head = Point(self.head.x + move.x, self.head.y + move.y)

    def _handle_events(self) -> None:
        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close()
                raise SystemExit

    def _update_ui(self) -> None:
        pygame = self._pygame
        self.display.fill((22, 22, 22))

        for index, point in enumerate(self.snake):
            color = (42, 168, 87) if index == 0 else (32, 128, 67)
            rect = pygame.Rect(
                point.x * self.block_size,
                point.y * self.block_size,
                self.block_size,
                self.block_size,
            )
            pygame.draw.rect(self.display, color, rect)
            pygame.draw.rect(self.display, (18, 86, 46), rect, 1)

        food_rect = pygame.Rect(
            self.food.x * self.block_size,
            self.food.y * self.block_size,
            self.block_size,
            self.block_size,
        )
        pygame.draw.rect(self.display, (220, 64, 64), food_rect)

        score_surface = self.font.render(f"Score: {self.score}", True, (230, 230, 230))
        self.display.blit(score_surface, (12, self.height * self.block_size + 8))
        pygame.display.flip()

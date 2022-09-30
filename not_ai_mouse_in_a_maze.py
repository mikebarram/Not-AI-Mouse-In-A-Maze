"""
*** HOW TO PROFILE ***
Install gprof2dot and dot
On Windows, to install dot, install Graphviz and make sure the path to the bin folder
that contains dot.exe is in the PATH environment variable (or include the full path to
dot when you call it)

Get the programme to start and end e.g. in main add
if statsInfoGlobal["Total frames"] > 10000:
            break
In the terminal run
python -m cProfile -o output.pstats maze_23_mouse_refactor.py
gprof2dot -f pstats output.pstats | "C:\\Program Files\\Graphviz\\bin\\dot.exe" -Tpng -o output23.png
Have a look at output.png
"""

# A program which uses recursive backtracking to generate a maze
# https://aryanab.medium.com/maze-generation-recursive-backtracking-5981bc5cc766
# https://github.com/AryanAb/MazeGenerator/blob/master/backtracking.py
# A good alternative might be https://github.com/AryanAb/MazeGenerator/blob/master/hunt_and_kill.py

# import the pygame module, so you can use it
import sys
import random
import math
import time
import os
import os.path
import json

# from collections import deque
from enum import Enum, auto
import numpy as np
import pygame
from numba import jit

sys.setrecursionlimit(8000)

USE_AI = False
MAZES_TO_ATTEMPT = 10000
FRAME_DISPLAY_RATE = 200
BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
PURE_WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
MAZE_SQUARE_SIZE = 70
MAZE_ROWS = 13
MAZE_COLS = 21

# ROWS AND COLS must be odd for a maze
if MAZE_ROWS % 2 == 0:
    MAZE_ROWS += 1
if MAZE_COLS % 2 == 0:
    MAZE_COLS += 1

# mouse will "crash" if it takes more than this distance to complete the maze
MAZE_MAX_DISTANCE_TO_COMPLETE = MAZE_ROWS * MAZE_COLS * MAZE_SQUARE_SIZE**1.5
# mazes that the mouse fails to solve are saved here.
# Any mazes in here are reloaded at the start of session.
MAZE_DIRECTORY = f"mazes\\{MAZE_COLS}x{MAZE_ROWS}\\"
MAZE_STATS_DIRECTORY = f"maze_stats\\{MAZE_COLS}x{MAZE_ROWS}\\"

WINDOW_WIDTH = MAZE_COLS * MAZE_SQUARE_SIZE
WINDOW_HEIGHT = MAZE_ROWS * MAZE_SQUARE_SIZE

# Mice can only look so far ahead. Needs to be larger than grid size for the maze
MOUSE_VISION_DISTANCE = round(2.0 * MAZE_SQUARE_SIZE)
MOUSE_ACCELERATION_MIN = -3  # change in speed in pixels per frame
MOUSE_ACCELERATION_MAX = 2  # change in speed in pixels per frame
MOUSE_STEERING_RADIANS_MAX = math.radians(45)
MOUSE_STEERING_RADIANS_DELTA_MAX = math.radians(45)
MOUSE_TRAIL_CIRCLE_ALPHA = None

# The following globals are to be set individually for each mouse
# to try to optimise their values
OPTIMISE_MOUSE_VISITED_PATH_RADIUS = 20
OPTIMISE_MOUSE_SPEED_MIN_INITIAL = 2  # pixels per frame
OPTIMISE_MOUSE_SPEED_MAX_INITIAL = 5  # pixels per frame
OPTIMISE_MOUSE_WEIGHTS_15 = 1.0 / 5.0
OPTIMISE_MOUSE_WEIGHTS_30 = 1.0 / 6.0
OPTIMISE_MOUSE_WEIGHTS_45 = 1.0 / 6.0
OPTIMISE_MOUSE_WEIGHTS_60 = 1.0 / 7.0
OPTIMISE_MOUSE_WEIGHTS_90 = 1.0 / 7.0
OPTIMISE_MOUSE_VISION_ANGLES_AND_WEIGHTS = (
    (math.radians(0.0), 1.0 / 2.0),
    (math.radians(-15.0), -OPTIMISE_MOUSE_WEIGHTS_15),
    (math.radians(15.0), OPTIMISE_MOUSE_WEIGHTS_15),
    (math.radians(-30.0), -OPTIMISE_MOUSE_WEIGHTS_30),
    (math.radians(30.0), OPTIMISE_MOUSE_WEIGHTS_30),
    (math.radians(-45.0), -OPTIMISE_MOUSE_WEIGHTS_45),
    (math.radians(45.0), OPTIMISE_MOUSE_WEIGHTS_45),
    (math.radians(-60.0), -OPTIMISE_MOUSE_WEIGHTS_60),
    (math.radians(60.0), OPTIMISE_MOUSE_WEIGHTS_60),
    (math.radians(-90.0), -OPTIMISE_MOUSE_WEIGHTS_90),
    (math.radians(90.0), OPTIMISE_MOUSE_WEIGHTS_90),
)  # 0 must be first, 90 degrees needed to get out of dead ends
OPTIMISE_MOUSE_STEERING_MULTIPLIER = 2.5
OPTIMISE_MOUSE_VISITED_PATH_AVOIDANCE_FACTOR = (
    1.25 * MAZE_SQUARE_SIZE / (2 * OPTIMISE_MOUSE_VISITED_PATH_RADIUS)
)
# the bigger the maze, the longer it could be before returning to a visited bit of maze
OPTIMISE_MOUSE_FRAMES_BETWEEN_BLURRING_VISITED = (MAZE_ROWS * MAZE_COLS) // 10


class Directions(Enum):
    """Directions a maze can go in"""

    DIRECTION_UP = 1
    DIRECTION_DOWN = 2
    DIRECTION_LEFT = 3
    DIRECTION_RIGHT = 4


class Backtracking:
    """backtracking algorithm for creating a maze"""

    def __init__(self, height, width):
        """heigth and width of maze should be odd, so add one if even"""
        if width % 2 == 0:
            width += 1
        if height % 2 == 0:
            height += 1

        self.width = width
        self.height = height

    def create_maze(self):
        """create a maze"""
        maze = np.ones((self.height, self.width), dtype=float)

        for i in range(self.height):
            for j in range(self.width):
                if i % 2 == 1 or j % 2 == 1:
                    maze[i, j] = 0
                if i == 0 or j == 0 or i == self.height - 1 or j == self.width - 1:
                    maze[i, j] = 0.5

        maze_sx = random.choice(range(2, self.width - 2, 2))
        maze_sy = random.choice(range(2, self.height - 2, 2))

        self.generator(maze_sx, maze_sy, maze)

        for i in range(self.height):
            for j in range(self.width):
                if maze[i, j] == 0.5:
                    maze[i, j] = 1

        return maze

    def generator(self, maze_cx, maze_cy, grid):
        """maze generator"""
        grid[maze_cy, maze_cx] = 0.5

        if (
            grid[maze_cy - 2, maze_cx] == 0.5
            and grid[maze_cy + 2, maze_cx] == 0.5
            and grid[maze_cy, maze_cx - 2] == 0.5
            and grid[maze_cy, maze_cx + 2] == 0.5
        ):
            pass
        else:
            maze_li = [1, 2, 3, 4]
            while len(maze_li) > 0:
                direction = random.choice(maze_li)
                maze_li.remove(direction)

                if direction == Directions.DIRECTION_UP.value:
                    maze_ny = maze_cy - 2
                    maze_my = maze_cy - 1
                elif direction == Directions.DIRECTION_DOWN.value:
                    maze_ny = maze_cy + 2
                    maze_my = maze_cy + 1
                else:
                    maze_ny = maze_cy
                    maze_my = maze_cy

                if direction == Directions.DIRECTION_LEFT.value:
                    maze_nx = maze_cx - 2
                    maze_mx = maze_cx - 1
                elif direction == Directions.DIRECTION_RIGHT.value:
                    maze_nx = maze_cx + 2
                    maze_mx = maze_cx + 1
                else:
                    maze_nx = maze_cx
                    maze_mx = maze_cx

                if grid[maze_ny, maze_nx] != 0.5:
                    grid[maze_my, maze_mx] = 0.5
                    self.generator(maze_nx, maze_ny, grid)


class Maze:
    """create and manage mazes
    Wiht the Backtracking algorithm, the start point for creating the maze
    is chosen randomly but every part of the generated maze can be visited
    from any other and the corners are always free,
    so the top left is the start and the bottom right is the end"""

    def __init__(self, rows, cols, square_size, maze_directory) -> None:
        self.rows = rows
        self.cols = cols
        self.square_size = square_size
        self.maze_directory = maze_directory
        self.maze_title = None
        self.maze_tiny = []
        self.maze_big = []
        self.from_saved = None
        self.path_distance = None

    def create(self):
        """create a new maze"""
        self.maze_title = "New maze"
        self.from_saved = False
        self.maze_tiny = self.get_new_maze(self.rows, self.cols)
        self.maze_big = self.get_big_bool_maze(self.maze_tiny, self.square_size)

    def save(self, save_reason):
        """save a maze. It will save to a folder based on the height and width of the
        maze. The file name will include the reason the file was saved, the date/time
        and the length of the path"""
        filename = "maze.txt"
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = self.maze_directory + "maze_" + save_reason + "_" + timestr
        if self.path_distance is not None:
            filename += "_path-" + str(self.path_distance)
        filename += ".txt"
        np.savetxt(filename, self.maze_tiny, fmt="%s")

    def load(self, filename):
        """load a previously saved maze"""
        self.maze_title = "Maze: " + filename
        self.from_saved = True
        self.file_name = filename
        self.maze_tiny = np.loadtxt(filename, dtype=int)
        self.maze_big = self.get_big_bool_maze(self.maze_tiny, self.square_size)

    @staticmethod
    def get_new_maze(rows, cols):
        """get a new maze"""
        backtracking = Backtracking(height=rows + 1, width=cols + 1)
        maze = backtracking.create_maze()
        # remove the outer elements of the array
        maze = np.delete(maze, [0, maze.shape[0] - 1], axis=0)
        maze = np.delete(maze, [0, maze.shape[1] - 1], axis=1)

        return maze

    @staticmethod
    def get_big_bool_maze(tiny_maze, scale_factor):
        """stretch maze"""
        maze_bool = tiny_maze.astype(dtype=bool)
        # from https://stackoverflow.com/a/4227280
        return np.repeat(
            np.repeat(maze_bool, scale_factor, axis=0), scale_factor, axis=1
        )


class MazeSolver:
    """solve the maze
    based on https://www.101computing.net/backtracking-maze-path-finder/
    """

    def __init__(self, maze_tiny) -> None:
        self.maze_tiny = maze_tiny

    def solve_maze(self):
        """
        maze_tiny has walls=0 and paths=1. change this to:
        walls = -1
        paths = 0
        end = -2
        shortest path will then count up from 1
        """
        maze = self.maze_tiny
        rows = len(maze)
        cols = len(maze[0])
        maze = maze - 1
        maze[rows - 2][cols - 2] = -2
        solved = self.explore_maze(maze, 1, 1, 1)

        min_path_distance = 0
        maze_path = None
        if solved:
            min_path_distance = int(np.max(maze))
            # change negative values to 0
            maze[maze < 0] = 0
            # scale the array values 100 to 255
            maze_path = np.trunc(maze * 155 / min_path_distance)
            maze_path[maze_path > 0] += 100

        return solved, min_path_distance, maze, maze_path

    # A backtracking/recursive function to check all possible paths until the exit is
    def explore_maze(self, maze, row, col, distance):
        if maze[row][col] == -2:
            # We found the exit
            return True
        elif maze[row][col] == 0:  # Empty path, not explored
            maze[row][col] = distance
            if row < len(maze) - 1:
                # Explore path below
                if self.explore_maze(maze, row + 1, col, distance + 1):
                    return True
            if row > 0:
                # Explore path above
                if self.explore_maze(maze, row - 1, col, distance + 1):
                    return True
            if col < len(maze[row]) - 1:
                # Explore path to the right
                if self.explore_maze(maze, row, col + 1, distance + 1):
                    return True
            if col > 0:
                # Explore path to the left
                if self.explore_maze(maze, row, col - 1, distance + 1):
                    return True
            # Backtrack
            maze[row][col] = -3


class MazeDrawer:
    """draw the maze with pygame and add graphical elements like the start and finish"""

    def __init__(self, maze, window_size, maze_surface, maze_path_surface) -> None:
        self.maze = maze
        self.window_size = window_size
        self.maze_surface = maze_surface
        self.maze_path_surface = maze_path_surface

    def draw_start(self, maze_surface, square_size, start_colour):
        rect_start = pygame.Rect(square_size, square_size, square_size, square_size)
        pygame.draw.rect(maze_surface, start_colour, rect_start)

    def draw_finish(self, window_size, maze_surface, square_size, finish_colour):
        # might be better to define the finish in terms of the maze size,
        # rather than window size, in case the maze doesn't fill the window
        rect_finish = pygame.Rect(
            window_size[0] - 2 * square_size,
            window_size[1] - 2 * square_size,
            square_size,
            square_size,
        )
        pygame.draw.rect(maze_surface, finish_colour, rect_finish)

    def draw_maze(self, maze_tiny, wall_colour, passage_colour):
        """create a tiny maze with just one pixel per segment of the maze, then scale it up"""
        surf = pygame.Surface((self.maze.cols, self.maze.rows))
        surf.fill(wall_colour)  # BLACK
        for i in range(0, maze_tiny.shape[1]):
            for j in range(0, maze_tiny.shape[0]):
                if maze_tiny[j][i] == 0:
                    surf.set_at((i, j), passage_colour)

        pygame.transform.scale(surf, self.window_size, self.maze_surface)

    def draw_shortest_path(self, maze_path):
        surf = pygame.Surface((self.maze.cols, self.maze.rows))
        for i in range(0, maze_path.shape[1]):
            for j in range(0, maze_path.shape[0]):
                path_colour_r = maze_path[j][i]
                if path_colour_r > 0:
                    path_colour = (path_colour_r, 0, 0)
                    surf.set_at((i, j), path_colour)
        pygame.transform.scale(surf, self.window_size, self.maze_path_surface)


class MouseIcon(pygame.sprite.Sprite):
    """Mouse icon"""

    def __init__(self, pos_x, pos_y):
        super().__init__()
        # mouse.png 40x65
        # https://flyclipart.com/lab-mouse-template-clip-art-free-mouse-clipart-791054#
        self.image = pygame.image.load("images/mouse.png")
        self.rect = self.image.get_rect()
        self.rect.center = [pos_x, pos_y]
        self.image_original = self.image
        self.rect_original = self.rect

    def update(self, pos_x, pos_y, angle_radians):
        """update the mouse icon to rotate it"""
        self.image, self.rect = self.rot_center(
            self.image_original, self.rect_original, angle_radians
        )
        self.rect.center = [pos_x, pos_y]

    def rot_center(self, image, rect, angle_radians):
        """rotate an image while keeping its center"""
        rot_image = pygame.transform.rotate(image, 270 - math.degrees(angle_radians))
        rot_rect = rot_image.get_rect(center=rect.center)
        return rot_image, rot_rect


class MouseStatus(Enum):
    """statuses that a mouse can have when hunting to find the end of a maze"""

    HUNTING = auto()
    SUCCESSFUL = auto()
    CRASHED = auto()
    TIMEDOUT = auto()


class Mouse:
    """a mouse"""

    def __init__(
        self,
        window_size,
        maze_big,
        maze_min_path_distance,
        maze_distance_score,
        visited_path_radius,
        speed_min_initial,
        speed_max_initial,
        vision_angles_and_weights,
        steering_multiplier,
        visited_path_avoidance_factor,
        frames_between_blurring_visited,
    ):
        self.window_size = window_size
        self.maze_big = (
            maze_big  # when accessing an element of maze_big, it's maze_big[y][x]
        )
        self.maze_min_path_distance = maze_min_path_distance
        self.maze_distance_score = maze_distance_score
        self.visited_path_radius = visited_path_radius
        self.speed_min_initial = speed_min_initial
        self.speed_max_initial = speed_max_initial
        self.vision_angles_and_weights = vision_angles_and_weights
        self.steering_multiplier = steering_multiplier
        self.visited_path_avoidance_factor = visited_path_avoidance_factor
        self.frames_between_blurring_visited = frames_between_blurring_visited

        """
        *** fixed globals used in the Mouse class ***
        MAZE_SQUARE_SIZE
        MAZE_COLS
        MAZE_ROWS
        MOUSE_STEERING_RADIANS_DELTA_MAX
        MOUSE_STEERING_RADIANS_MAX
        MOUSE_VISION_DISTANCE
        MOUSE_ACCELERATION_MIN
        MOUSE_ACCELERATION_MAX

        *** globals used in the Mouse class that will become variables, set for each instance by 'AI' ***
        OPTIMISE_MOUSE_VISITED_PATH_RADIUS
        OPTIMISE_MOUSE_SPEED_MIN_INITIAL
        OPTIMISE_MOUSE_SPEED_MAX_INITIAL
        OPTIMISE_MOUSE_WEIGHTS_15
        OPTIMISE_MOUSE_WEIGHTS_30
        OPTIMISE_MOUSE_WEIGHTS_45
        OPTIMISE_MOUSE_WEIGHTS_60
        OPTIMISE_MOUSE_WEIGHTS_90
        OPTIMISE_MOUSE_VISION_ANGLES_AND_WEIGHTS
        OPTIMISE_MOUSE_STEERING_MULTIPLIER
        OPTIMISE_MOUSE_VISITED_PATH_AVOIDANCE_FACTOR
        OPTIMISE_MOUSE_FRAMES_BETWEEN_BLURRING_VISITED

        """

        # actual position is recorded as a tuple of floats
        # position is rounded just for display
        # and to see if the mouse is still on the maze
        # starting position is in the middle of top left square inside the border
        self.position = (MAZE_SQUARE_SIZE * 1.5, MAZE_SQUARE_SIZE * 1.5)
        self.position_rounded = (round(self.position[0]), round(self.position[1]))
        self.speed = MAZE_SQUARE_SIZE / 50  # pixels per frame
        self.speed_min = speed_min_initial
        self.speed_max = speed_max_initial
        self.status = MouseStatus.HUNTING
        # could try to set initial direction based on the shape of the
        # maze that's generated but this is fine
        self.direction_radians = math.pi / 3
        self.steering_radians = 0
        self.position_previous_rounded = self.position_rounded
        self.visited_alpha = np.zeros(window_size, dtype=np.int16)
        self.whiskers = None

        self.trail_circle_alpha = self.create_trail_circle_alpha(
            self.visited_path_radius, 10
        )

        self.stats_info_mouse = {
            "distance": 0.0,
            "frames": 0,
            "average speed": 0.0,
            "speed min": self.speed_min,
            "speed max": self.speed_max,
            "maze progress": 0,
        }

        self.score = 0
        self.max_happy_path_reached = 0

        """
        self.instructions = {
            "speed": 0.0,
            "speed_delta": 0.0,
            "direction_radians": 0.0,
            "steering_radians": 0.0,
        }
        # keep a list of the 20 last instructions, so we can see where it went wrong
        self.latest_instructions = deque(maxlen=20)
        """

    @staticmethod
    def create_trail_circle_alpha(circle_radius, trail_alpha):
        """create a 2D array of integers that have a given value if they are in a circle,
        otherwise they are zero, except for those around the edge of the circle"""
        x_list = np.arange(-circle_radius, circle_radius)
        y_list = np.arange(-circle_radius, circle_radius)
        array_alpha = np.zeros((y_list.size, x_list.size), dtype=np.int16)

        # the circle will mostly have opacity trail_alpha
        # but be 1/3 and 2/3 that around the outside
        mask_outer = (x_list[np.newaxis, :]) ** 2 + (
            y_list[:, np.newaxis]
        ) ** 2 < circle_radius**2
        mask_middle = (x_list[np.newaxis, :]) ** 2 + (y_list[:, np.newaxis]) ** 2 < (
            circle_radius - 4
        ) ** 2
        mask_inner = (x_list[np.newaxis, :]) ** 2 + (y_list[:, np.newaxis]) ** 2 < (
            circle_radius - 8
        ) ** 2

        array_alpha[mask_outer] = trail_alpha // 3
        array_alpha[mask_middle] = (2 * trail_alpha) // 3
        array_alpha[mask_inner] = trail_alpha

        return array_alpha

    def speed_increase(self):
        """increase speed by 1 pixel per frame"""
        self.speed_min += 1
        self.speed_max += 1

    def speed_decrease(self):
        """decrease speed by 1 pixel per frame to a minimum of 1"""
        self.speed_min = min(1, self.speed_min - 1)
        self.speed_max = min(1, self.speed_max - 1)

    def update_happy_path_progress(self):
        # get the position of the mouse on a scale where each sqaure of the maze is 1 pixel (as in maze_path and maze_tiny)
        position_tiny_x = round(self.position[0] / MAZE_SQUARE_SIZE)
        position_tiny_y = round(self.position[1] / MAZE_SQUARE_SIZE)
        # get the value of maze_path at that position
        progress_for_position = self.maze_distance_score[position_tiny_y][
            position_tiny_x
        ]
        # update max_happy_path_reached if further progress has been reached than ever before
        if progress_for_position > self.max_happy_path_reached:
            self.max_happy_path_reached = round(progress_for_position)

    def update_score(self):
        """
        the  mouse's score is updated when it stops hunting - it's successful, crashes or times out.
        """
        if self.status is MouseStatus.SUCCESSFUL:
            # nice high score. Multiply by min-path distance. Divide by number of steps (frames) taken.
            self.score = (
                1000
                * self.maze_min_path_distance
                * MAZE_SQUARE_SIZE
                / self.stats_info_mouse["frames"]
            )
        elif self.status is MouseStatus.CRASHED or self.status is MouseStatus.TIMEDOUT:
            # score based on how far the mouse got through the maze along the path to the end
            # mouse could have reached much further along the path to the end than where it crashed, so need to calculate how far it got with each step.
            self.score = 100 * self.max_happy_path_reached / self.maze_min_path_distance

    def get_driving_changes_with_ai(maze_wall_distances, speed_previous):
        new_steering_radians = 0
        speed_delta = 0
        return new_steering_radians, speed_delta

    def drive(self, draw_frame, use_ai):
        """to be called every frame to move the mouse along"""
        maze_wall_distances, self.whiskers = self.get_maze_wall_distances()

        if self.status is not MouseStatus.HUNTING:
            return

        if self.check_if_position_wins(self.position[0], self.position[1]):
            self.status = MouseStatus.SUCCESSFUL
            self.update_score()
            return

        self.position_previous_rounded = self.position_rounded

        steering_radians_previous = self.steering_radians
        new_steering_radians = 0
        speed_delta = 0

        if use_ai:
            new_steering_radians, speed_delta = self.get_driving_changes_with_ai(maze_wall_distances, self.speed)
        else:
            is_dead_end = self.check_if_dead_end(maze_wall_distances)

            if is_dead_end:
                new_steering_radians = self.get_steering_radians_for_dead_end(
                    maze_wall_distances
                )
            else:
                new_steering_radians = self.get_steering_radians(
                    maze_wall_distances,
                    self.steering_multiplier,
                    self.visited_path_avoidance_factor,
                )

            # restrict how much the steering can be changed per frame
            new_steering_radians = self.restrict_steering_changes(
                new_steering_radians, steering_radians_previous
            )

            # restrict how much the steering can be per frame
            new_steering_radians = self.restrict_steering(new_steering_radians)

            new_direction_radians = self.direction_radians + new_steering_radians
            (maze_wall_distance, _, _, _, _,) = self.get_maze_wall_distance(
                self.maze_big,
                self.visited_alpha,
                self.direction_radians,
                self.position_rounded[0],
                self.position_rounded[1],
                new_steering_radians,
            )

            speed_delta = 4 * (maze_wall_distance / MOUSE_VISION_DISTANCE) - 2

        if speed_delta < MOUSE_ACCELERATION_MIN:
            speed_delta = MOUSE_ACCELERATION_MIN

        if speed_delta > MOUSE_ACCELERATION_MAX:
            speed_delta = MOUSE_ACCELERATION_MAX

        new_speed = self.speed + speed_delta

        if new_speed < self.speed_min:
            new_speed = self.speed_min

        if new_speed > self.speed_max:
            new_speed = self.speed_max

        new_position = (
            self.position[0] + new_speed * math.cos(new_direction_radians),
            self.position[1] + new_speed * math.sin(new_direction_radians),
        )

        self.speed = new_speed
        self.steering_radians = new_steering_radians
        self.direction_radians = new_direction_radians

        self.position = new_position
        self.position_rounded = (round(self.position[0]), round(self.position[1]))

        if draw_frame:
            pygame.display.update(
                pygame.Rect(self.position_rounded[0], self.position_rounded[1], 1, 1)
            )
            pygame.display.update()

        self.update_visited(
            self.position,
            self.direction_radians,
            self.visited_alpha,
            self.trail_circle_alpha,
            self.visited_path_radius,
        )

        # decided that the mouse has "crashed" if it has taken more than this
        # distance to complete the maze - better to create a "crashed" status
        if self.stats_info_mouse["distance"] > MAZE_MAX_DISTANCE_TO_COMPLETE:
            self.status = MouseStatus.TIMEDOUT
            self.update_score()
            return

        if self.status is MouseStatus.HUNTING:
            self.update_happy_path_progress()

        self.stats_info_mouse["frames"] += 1
        self.stats_info_mouse["distance"] += new_speed
        self.stats_info_mouse["average speed"] = (
            self.stats_info_mouse["distance"] // self.stats_info_mouse["frames"]
        )
        self.stats_info_mouse["speed min"] = self.speed_min
        self.stats_info_mouse["speed max"] = self.speed_max
        self.stats_info_mouse["maze progress"] = self.max_happy_path_reached

        """
        self.instructions = {
            "speed": self.speed,
            "speed_delta": speed_delta,
            "direction_radians": self.direction_radians,
            "steering_radians": self.steering_radians,
            "maze_wall_distances": maze_wall_distances,
        }
        self.latest_instructions.appendleft(self.instructions)
        """

        if self.stats_info_mouse["frames"] % self.frames_between_blurring_visited == 0:
            self.fade_visited(self.visited_alpha)

    @staticmethod
    def get_steering_radians_for_dead_end(maze_wall_distances):
        # old code from maze v3 that doesn't take into account
        # whether pixels have been visited
        new_steering_angle = 0
        max_maze_distance = 1
        for ted in maze_wall_distances:
            if ted[0][0] == 0:
                continue

            new_steering_angle += ted[1] * ted[0][1]
            if ted[1] > max_maze_distance:
                max_maze_distance = ted[1]

        return 5 * new_steering_angle / max_maze_distance

    @staticmethod
    def get_steering_radians(
        maze_wall_distances, steering_multiplier, visited_path_avoidance_factor
    ):
        new_steering_angle = 0
        max_maze_distance = 1
        min_maze_distance = 1000
        for ted in maze_wall_distances:
            if ted[0][0] == 0:
                continue

            # this takes into account the sum of the alpha channel of the pixels
            # that have been visited, rather than just the number of visited pixels
            distance_for_angle = max(
                0,
                ted[1]
                * (1 - (ted[3] * visited_path_avoidance_factor) / (255 * ted[1])),
            )

            new_steering_angle += distance_for_angle * ted[0][1]
            if distance_for_angle > max_maze_distance:
                max_maze_distance = distance_for_angle
            if distance_for_angle < min_maze_distance:
                min_maze_distance = distance_for_angle

        return new_steering_angle * steering_multiplier / max_maze_distance

    @staticmethod
    def restrict_steering_changes(new_steering_radians, steering_radians_previous):
        # restrict how much the steering can be changed per frame
        if (
            new_steering_radians
            < steering_radians_previous - MOUSE_STEERING_RADIANS_DELTA_MAX
        ):
            new_steering_radians = (
                steering_radians_previous - MOUSE_STEERING_RADIANS_DELTA_MAX
            )
        elif (
            new_steering_radians
            > steering_radians_previous + MOUSE_STEERING_RADIANS_DELTA_MAX
        ):
            new_steering_radians = (
                steering_radians_previous + MOUSE_STEERING_RADIANS_DELTA_MAX
            )
        return new_steering_radians

    @staticmethod
    def restrict_steering(new_steering_radians):
        # restrict how much the steering can be per frame
        if new_steering_radians > MOUSE_STEERING_RADIANS_MAX:
            new_steering_radians = MOUSE_STEERING_RADIANS_MAX
        elif new_steering_radians < -MOUSE_STEERING_RADIANS_MAX:
            new_steering_radians = -MOUSE_STEERING_RADIANS_MAX
        return new_steering_radians

    @staticmethod
    @jit(nopython=True)
    def update_visited(
        position,
        direction_radians,
        visited_alpha,
        trail_circle_alpha,
        visited_path_radius,
    ):
        """Updates a mouses 2D array to record that the mouse has been
        in a particular area"""
        # update a 2d numpy array that represents visited pixels.
        # it should add a circular array behind itself.
        # needs to know the angle of travel
        # needs to add but only to a level of saturation (max 255)
        circle_top_left_x = round(
            position[0]
            - visited_path_radius * math.cos(direction_radians)
            - visited_path_radius
        )
        circle_top_left_y = round(
            position[1]
            - visited_path_radius * math.sin(direction_radians)
            - visited_path_radius
        )
        # https://stackoverflow.com/questions/9886303/adding-different-sized-shaped-displaced-numpy-matrices
        section_to_update = visited_alpha[
            circle_top_left_x : circle_top_left_x + 2 * visited_path_radius,
            circle_top_left_y : circle_top_left_y + 2 * visited_path_radius,
        ]
        section_to_update += trail_circle_alpha  # use array slicing
        # section_to_update[section_to_update > 255] = 255 -- this doesn't work. No idea why
        np.clip(section_to_update, 0, 255, out=section_to_update)

    @staticmethod
    # @jit(nopython=True)
    def fade_visited(visited_alpha):
        """fades the mouses 2D array of records of where it has been - like
        a scent fading away"""
        # subtract from the alpha channel but stop it going below zero
        visited_alpha[visited_alpha >= 1] -= 1
        return

    @staticmethod
    # @jit(nopython=True)
    def check_if_position_wins(position_x, position_y):
        """check if the mouse has reached the end of the maze"""
        if (
            -2 < position_x / MAZE_SQUARE_SIZE - MAZE_COLS < -1
            and -2 < position_y / MAZE_SQUARE_SIZE - MAZE_ROWS < -1
        ):
            return True
        else:
            return False

    @staticmethod
    # @jit(nopython=True)
    def check_if_dead_end(maze_wall_distances):
        """check if the mouse is in a dead end"""
        # if any of the distances is more than the size of a square,
        # then it's not a dead end
        for ted in maze_wall_distances:
            if (
                math.pi / -2.0 <= ted[0][0] <= math.pi / 2.0
                and ted[1] > MAZE_SQUARE_SIZE
            ):
                return False

        return True

    def get_maze_wall_distances(self):
        """get the distance of the mouse from the edges of the maze along
        a defined list of angles from its direction of travel"""
        mouse_in_maze_passage = self.maze_big[self.position_rounded[1]][
            self.position_rounded[0]
        ]
        if not mouse_in_maze_passage:
            self.status = MouseStatus.CRASHED
            self.update_score()
            return [], []

        maze_wall_distances = []
        whiskers = []

        for vision_angle_and_weight in self.vision_angles_and_weights:
            (
                maze_wall_distance,
                visited_count,
                visited_alpha_total,
                edge_x,
                edge_y,
            ) = self.get_maze_wall_distance(
                self.maze_big,
                self.visited_alpha,
                self.direction_radians,
                self.position_rounded[0],
                self.position_rounded[1],
                vision_angle_and_weight[0],
            )
            whiskers += [self.position_rounded, (edge_x, edge_y)]

            maze_wall_distances.append(
                (
                    vision_angle_and_weight,
                    maze_wall_distance,
                    visited_count,
                    visited_alpha_total,
                )
            )

        return maze_wall_distances, whiskers

    @staticmethod
    @jit(nopython=True)
    def get_maze_wall_distance(
        maze_big,
        visited_alpha,
        direction_radians,
        position_rounded_x,
        position_rounded_y,
        vision_angle,
    ):
        """get the distance of the mouse from the edges of the maze along a
        single angle from its direction of travel"""
        # from x,y follow a line at vision_angle until no longer on the maze passage
        # or until MOUSE_VISION_DISTANCE has been reached
        search_angle_radians = direction_radians + vision_angle
        delta_x = math.cos(search_angle_radians)
        delta_y = math.sin(search_angle_radians)

        edge_distance = np.int64(0)
        visited_count = np.int64(0)
        visited_alpha_total = np.int64(0)

        for i in range(1, MOUSE_VISION_DISTANCE):
            edge_distance = i
            # incrementing test_x e.g. test_x += delta_x
            # makes the function slower by ~10%
            test_x = position_rounded_x + i * delta_x
            test_y = position_rounded_y + i * delta_y
            # saving the rounded values, rather than rounding twice
            # improves performance of this function by ~5% and by ~3% overall
            test_x_round = round(test_x)
            test_y_round = round(test_y)
            if maze_big[test_y_round][test_x_round] is False:
                break

            visited_alpha_pixel = visited_alpha[test_x_round, test_y_round]
            visited_alpha_total += visited_alpha_pixel
            if visited_alpha_pixel > 0:
                visited_count += 1

        return (
            edge_distance,
            visited_count,
            visited_alpha_total,
            test_x_round,
            test_y_round,
        )


class MouseDrawer:
    """draws a mouse"""

    """
        *** fixed globals used in the MouseDrawer class ***
        MAZE_SQUARE_SIZE
        GREEN
        WHITE
        BLACK
    """

    def __init__(
        self,
        screen,
        visited_by_mouse_screen,
        maze_wall_distances_screen,
        position_rounded,
    ):
        self.screen = screen
        self.visited_by_mouse_screen = visited_by_mouse_screen
        self.maze_wall_distances_screen = maze_wall_distances_screen
        self.position_rounded = position_rounded

        self.mouse_icon = MouseIcon(self.position_rounded[0], self.position_rounded[1])
        self.mouse_icon_group = pygame.sprite.Group()
        self.mouse_icon_group.add(self.mouse_icon)

    def draw_mouse_trail(self, position_rounded, speed, speed_max):
        mouse_speed_colour = round(255 * speed / speed_max)
        mouse_colour = (255 - mouse_speed_colour, mouse_speed_colour, 0)
        self.screen.set_at(position_rounded, mouse_colour)

    def draw_mouse(
        self, status, position_rounded, direction_radians, visited_alpha, whiskers
    ):
        self.mouse_icon_group.update(
            position_rounded[0],
            position_rounded[1],
            direction_radians,
        )

        if status is not MouseStatus.CRASHED:
            # if crashed into a corner, all whiskers might be zero length
            # which would give an error of "points argument must contain 2 or more points"
            self.draw_lines_to_maze_edge(whiskers)

        if status is MouseStatus.SUCCESSFUL:
            self.draw_mouse_finish_location(GREEN)
        elif status is MouseStatus.TIMEDOUT:
            self.draw_mouse_finish_location(RED)
        elif status is MouseStatus.CRASHED:
            self.draw_mouse_finish_location(RED)

        # from https://github.com/pygame/pygame/issues/1244
        surface_alpha = np.array(self.visited_by_mouse_screen.get_view("A"), copy=False)
        surface_alpha[:, :] = visited_alpha

    def draw_lines_to_maze_edge(self, whiskers):
        """draw lines from the mouse to the edge of the maze"""
        self.maze_wall_distances_screen.fill((0, 0, 0, 0))
        pygame.draw.lines(self.maze_wall_distances_screen, WHITE, False, whiskers)

    def draw_mouse_finish_location(self, highlight_colour):
        """draw where the mouse finishes"""
        finish_radius = MAZE_SQUARE_SIZE // 2
        pygame.draw.circle(
            self.screen, highlight_colour, self.position_rounded, finish_radius, width=2
        )
        finish_zone = pygame.Rect(
            self.position_rounded[0] - finish_radius,
            self.position_rounded[1] - finish_radius,
            2 * finish_radius,
            2 * finish_radius,
        )
        pygame.display.update(finish_zone)


def stats_info_global_update(stats_info_global, start_time, mouse):
    elapsed_time = time.monotonic() - start_time
    stats_info_global["FPS"] = stats_info_global["Total frames"] // elapsed_time
    stats_info_global["Total mazes"] += 1
    if mouse.status is MouseStatus.SUCCESSFUL:
        stats_info_global["Success count"] += 1
        stats_info_global["Total successes"] += 1
        stats_info_global["Max successes in a row"] = max(
            stats_info_global["Max successes in a row"],
            stats_info_global["Success count"],
        )
    elif (
        mouse.status is MouseStatus.CRASHED
        or mouse.status is MouseStatus.TIMEDOUT
    ):
        stats_info_global["Max successes in a row"] = max(
            stats_info_global["Max successes in a row"],
            stats_info_global["Success count"],
        )
        stats_info_global["Success count"] = 0
    stats_info_global["Success ratio"] = (
        stats_info_global["Total successes"]
        / stats_info_global["Total mazes"]
    )
    stats_info_global["Average frames per maze"] = (
        stats_info_global["Total frames"] / stats_info_global["Total mazes"]
    )


def stats_update(stats_surface, stats_info_mouse, stats_info_global):
    """updates the stats surface with global and local stats"""
    stats_surface.fill((0, 0, 0, 0))
    font = pygame.font.SysFont("Arial", 12, bold=False)
    text_top = 0

    for stats in (stats_info_mouse, stats_info_global):
        for stats_key, stats_value in stats.items():
            if stats_value > 10:
                stats_value = round(stats_value)
            img = font.render(
                stats_key + ": " + str(stats_value),
                True,
                pygame.Color(BLACK),
                pygame.Color(WHITE),
            )
            img_size = img.get_size()
            text_top += img_size[1] + 10
            stats_surface.blit(img, (10, text_top))


class MazeStats:
    def __init__(self):
        self.maze_stats = []

    def add_maze(self, mouse):
        self.maze_stats.append(
            {
                "frames": mouse.stats_info_mouse["frames"],
                "distance": mouse.stats_info_mouse["distance"],
                "maze_min_path_distance": mouse.maze_min_path_distance,
                "max_happy_path_reached": mouse.max_happy_path_reached,
                "status": mouse.status.name,
                "score": mouse.score,
            }
        )

    def save_to_file(self):
        if not os.path.exists(MAZE_STATS_DIRECTORY):
            os.makedirs(MAZE_STATS_DIRECTORY)
        with open(MAZE_STATS_DIRECTORY + "maze_stats.json", "w") as fout:
            fout.write("{\n")
            for statsDict in self.maze_stats:
                json.dump(statsDict, fout)
                fout.write("\n")
            fout.write("}")


# define a main function
def main():
    """main function"""

    # initialize the pygame module
    pygame.init()
    # clock = pygame.time.Clock()
    start_time = time.monotonic()
    # load and set the logo
    logo = pygame.image.load("logo32x32.png")
    pygame.display.set_icon(logo)
    pygame.display.set_caption("Not AI mouse")

    window_size = (WINDOW_WIDTH, WINDOW_HEIGHT)

    # create a surface on screen that has the size defined globally
    screen = pygame.display.set_mode(window_size)
    background = pygame.Surface(window_size)

    maze_path_surface = pygame.Surface(window_size)  # , pygame.SRCALPHA, 32)
    maze_path_surface.set_alpha(60)
    maze_path_surface.fill((255, 255, 255))

    visited_by_mouse_screen = pygame.Surface(window_size, pygame.SRCALPHA, 32)
    visited_by_mouse_screen = visited_by_mouse_screen.convert_alpha()
    visited_by_mouse_screen.fill((0, 0, 0, 0))

    maze_wall_distances_screen = pygame.Surface(window_size, pygame.SRCALPHA, 32)
    maze_wall_distances_screen = maze_wall_distances_screen.convert_alpha()

    stats_surface = pygame.Surface(window_size)
    stats_surface = stats_surface.convert_alpha()
    stats_surface.fill((0, 0, 0, 0))

    # define a variable to control the main loop
    running = True
    paused = False
    draw_frame = False
    new_maze_and_mouse_needed = True
    if not os.path.exists(MAZE_DIRECTORY):
        os.mkdir(MAZE_DIRECTORY)
    saved_maze_files = [
        name
        for name in os.listdir(MAZE_DIRECTORY)
        if os.path.isfile(os.path.join(MAZE_DIRECTORY, name))
    ]
    saved_maze_count = len(saved_maze_files)
    saved_maze_counter = 0
    stats_info_global = {
        "Total frames": 0,
        "Success count": 0,
        "Max successes in a row": 0,
        "Total successes": 0,
        "Total mazes": 0,
        "Success ratio": 0,
        "Average frames per maze": 0,
        "FPS": 0,
    }
    mouse = None
    maze_stats = MazeStats()

    # main loop
    while running:
        stats_info_global["Total frames"] += 1
        if (
            MAZES_TO_ATTEMPT != 0
            and stats_info_global["Total mazes"] >= MAZES_TO_ATTEMPT
        ):
            maze_stats.save_to_file()
            pygame.time.wait(10000)
            break
        if new_maze_and_mouse_needed:
            # create the maze and draw it on the background
            if mouse is not None:
                stats_info_global_update(stats_info_global, start_time, mouse)
            maze = Maze(MAZE_ROWS, MAZE_COLS, MAZE_SQUARE_SIZE, MAZE_DIRECTORY)
            if saved_maze_counter < saved_maze_count:
                maze.load(MAZE_DIRECTORY + saved_maze_files[saved_maze_counter])
                saved_maze_counter += 1
            else:
                maze.create()

            maze_solver = MazeSolver(maze.maze_tiny)
            (
                maze_is_solved,
                maze_min_path_distance,
                maze_distance_score,
                maze_path,
            ) = maze_solver.solve_maze()
            maze.path_distance = maze_min_path_distance
            stats_info_global["Path distance"] = maze_min_path_distance

            maze_drawer = MazeDrawer(maze, window_size, background, maze_path_surface)
            maze_drawer.draw_maze(maze.maze_tiny, PURE_WHITE, BLACK)
            maze_drawer.draw_shortest_path(maze_path)
            maze_drawer.draw_start(background, MAZE_SQUARE_SIZE, (100, 100, 100))
            maze_drawer.draw_finish(
                window_size, background, MAZE_SQUARE_SIZE, (100, 100, 100)
            )
            mouse = Mouse(
                window_size,
                maze.maze_big,
                maze_min_path_distance,
                maze_distance_score,
                OPTIMISE_MOUSE_VISITED_PATH_RADIUS,
                OPTIMISE_MOUSE_SPEED_MIN_INITIAL,
                OPTIMISE_MOUSE_SPEED_MAX_INITIAL,
                OPTIMISE_MOUSE_VISION_ANGLES_AND_WEIGHTS,
                OPTIMISE_MOUSE_STEERING_MULTIPLIER,
                OPTIMISE_MOUSE_VISITED_PATH_AVOIDANCE_FACTOR,
                OPTIMISE_MOUSE_FRAMES_BETWEEN_BLURRING_VISITED,
            )
            mouse_drawer = MouseDrawer(
                background,
                visited_by_mouse_screen,
                maze_wall_distances_screen,
                mouse.position_rounded,
            )
            new_maze_and_mouse_needed = False

        # event handling, gets all event from the event queue
        for event in pygame.event.get():
            # only do something if the event is of type QUIT
            if event.type == pygame.QUIT:
                # change the value to False, to exit the main loop
                running = False
                sys.exit()
                break

            # for the next bit, on windows, you need to:
            # pip install windows-curses
            # https://stackoverflow.com/questions/35850362/importerror-no-module-named-curses-when-trying-to-import-blessings
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n:
                    new_maze_and_mouse_needed = True
                    continue
                elif event.key == pygame.K_ESCAPE:
                    running = False
                    maze_stats.save_to_file()
                    sys.exit()
                    break
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    mouse.speed_increase()
                elif event.key == pygame.K_LEFT:
                    mouse.speed_decrease()
                elif event.key == pygame.K_s:
                    maze.save(save_reason="manual")
                elif event.key == pygame.K_f:
                    maze_stats.save_to_file()

        if paused:
            continue

        if (
            stats_info_global["Total frames"] % FRAME_DISPLAY_RATE == 0
            or mouse.status is not MouseStatus.HUNTING
        ):
            draw_frame = True
        else:
            draw_frame = False

        if mouse.status is MouseStatus.HUNTING:
            mouse.drive(draw_frame, USE_AI)
            # the trail is drawn to its surface (but not necessarily rendered) on every frame
            mouse_drawer.draw_mouse_trail(
                mouse.position_rounded, mouse.speed, mouse.speed_max
            )

        if mouse.status is MouseStatus.SUCCESSFUL:
            new_maze_and_mouse_needed = True
            # pygame.time.wait(1000)

        if mouse.status is MouseStatus.CRASHED or mouse.status is MouseStatus.TIMEDOUT:
            # don't re-save a maze that has been loaded from a saved maze
            # however a log of those saved ones that have failed again would be n
            if not maze.from_saved:
                maze.save(save_reason=mouse.status.name)
            new_maze_and_mouse_needed = True
            # pygame.time.wait(1000)

        if mouse.status is not MouseStatus.HUNTING:
            maze_stats.add_maze(mouse)

        if draw_frame:
            mouse_drawer.draw_mouse(
                mouse.status,
                mouse.position_rounded,
                mouse.direction_radians,
                mouse.visited_alpha,
                mouse.whiskers,
            )
            stats_update(stats_surface, mouse.stats_info_mouse, stats_info_global)
            pygame.display.flip()
            screen.blit(background, (0, 0))
            screen.blit(maze_path_surface, (0, 0))
            screen.blit(mouse_drawer.visited_by_mouse_screen, (0, 0))
            screen.blit(mouse_drawer.maze_wall_distances_screen, (0, 0))
            mouse_drawer.mouse_icon_group.draw(screen)
            screen.blit(stats_surface, (0, 0))
            pygame.display.update()

        # clock.tick(400)


# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__ == "__main__":
    # call the main function
    main()

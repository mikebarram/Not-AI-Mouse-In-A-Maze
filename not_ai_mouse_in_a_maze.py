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
python -m cProfile -o output.pstats maze-18-JIT-4650-FPS.py
gprof2dot -f pstats output.pstats | "C:\\Program Files\\Graphviz\\bin\\dot.exe" -Tpng -o outputJIT.png
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
from collections import deque
from enum import Enum
import numpy as np
import pygame
from numba import jit

sys.setrecursionlimit(8000)

BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
PURE_WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
SQUARE_SIZE = 70
ROWS = 13
COLS = 19
# mouse will "crash" if it takes more than this distance to complete the maze
MAZE_MAX_DISTANCE_TO_COMPLETE = ROWS * COLS * SQUARE_SIZE**1.5
# mazes that the mouse fails to solve are saved here.
# Any mazes in here are reloaded at the start of session.
MAZE_DIRECTORY = f"mazes\\{COLS}x{ROWS}\\"

WINDOW_WIDTH = COLS * SQUARE_SIZE
WINDOW_HEIGHT = ROWS * SQUARE_SIZE

# Mice can only look so far ahead. Needs to be larger than grid size for the maze
CAR_VISION_DISTANCE = round(2.0 * SQUARE_SIZE)
CAR_VISION_ANGLES_AND_WEIGHTS = (
    (math.radians(0.0), 1.0 / 2.0),
    (math.radians(-15.0), -1.0 / 5.0),
    (math.radians(15.0), 1.0 / 5.0),
    (math.radians(-30.0), -1.0 / 6.0),
    (math.radians(30.0), 1.0 / 6.0),
    (math.radians(-45.0), -1.0 / 6.0),
    (math.radians(45.0), 1.0 / 6.0),
    (math.radians(-60.0), -1.0 / 7.0),
    (math.radians(60.0), 1.0 / 7.0),
    (math.radians(-90.0), -1.0 / 7.0),
    (math.radians(90.0), 1.0 / 7.0),
)  # 0 must be first, 90 degrees needed to get out of dead ends
CAR_SPEED_MIN_INITIAL = 2  # pixels per frame
CAR_SPEED_MAX_INITIAL = 5  # pixels per frame
CAR_ACCELERATION_MIN = -3  # change in speed in pixels per frame
CAR_ACCELERATION_MAX = 2  # change in speed in pixels per frame
CAR_STEERING_RADIANS_MAX = math.radians(45)
CAR_STEERING_RADIANS_DELTA_MAX = math.radians(45)
CAR_STEERING_MULTIPLIER = 2.5
CAR_VISITED_PATH_RADIUS = 20
CAR_TRAIL_CIRCLE_ALPHA = None
# how much wider the maze is than the path
CAR_VISITED_PATH_AVOIDANCE_FACTOR = 1.25 * SQUARE_SIZE / (2 * CAR_VISITED_PATH_RADIUS)
# the bigger the maze, the longer it could be before returning to a visited bit of maze
FRAMES_BETWEEN_BLURRING_VISITED = (ROWS * COLS) // 10
FRAME_DISPLAY_RATE = 500


@staticmethod
def scale_2d_np_array(array_in, scale_factor):
    """scale a 2d numpy array by the same factor in both directions"""
    # from https://stackoverflow.com/a/4227280
    return np.repeat(np.repeat(array_in, scale_factor, axis=0), scale_factor, axis=1)


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
    """create and manage mazes"""

    def __init__(self, rows, cols) -> None:
        self.rows = rows
        self.cols = cols
        self.maze_title = None
        self.maze_tiny = []
        self.maze_big = []
        self.from_saved = None

    def create(self):
        """create a new maze"""
        self.maze_title = "New maze"
        self.from_saved = False
        self.maze_tiny = self.get_new_maze()
        maze_bool = self.maze_tiny.astype(dtype=bool)
        self.maze_big = self.stretch_maze(maze_bool)

    def save(self, timestamp):
        """save a maze. It will save to a folder based on the height and width of the
        maze and a file that includes the current time"""
        filename = "maze.txt"
        if timestamp:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            filename = MAZE_DIRECTORY + "maze_" + timestr + ".txt"
        np.savetxt(filename, self.maze_tiny, fmt="%s")

    def load(self, filename):
        """load a previously saved maze"""
        self.maze_title = "Maze: " + filename
        self.from_saved = True
        self.file_name = filename
        self.maze_tiny = np.loadtxt(filename, dtype=np.int)
        maze_bool = self.maze_tiny.astype(dtype=bool)
        self.maze_big = self.stretch_maze(maze_bool)

    def get_new_maze(self):
        """get a new maze"""
        maze_height = self.rows + 1
        maze_width = self.cols + 1

        backtracking = Backtracking(maze_height, maze_width)
        maze = backtracking.create_maze()
        # remove the outer elements of the array
        maze = np.delete(maze, [0, maze.shape[0] - 1], axis=0)
        maze = np.delete(maze, [0, maze.shape[1] - 1], axis=1)

        return maze

    def stretch_maze(self, tiny_maze):
        """stretch maze"""
        tiny_maze = np.swapaxes(tiny_maze, 1, 0)
        return scale_2d_np_array(tiny_maze, SQUARE_SIZE)


class MazeDrawer:
    """draw the maze with pygame and add graphical elements like the start and finish"""

    def __init__(self, maze, window_size, maze_surface) -> None:
        self.maze = maze
        self.window_size = window_size
        self.maze_surface = maze_surface

    def draw_start(self, maze_surface, square_size, start_colour):
        rect_start = pygame.Rect(
            square_size, square_size, square_size, square_size
        )
        pygame.draw.rect(maze_surface, start_colour, rect_start)

    def draw_finish(self, window_size, maze_surface, square_size, finish_colour):
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


class Mouse:
    """a mouse"""

    def __init__(self, screen, visited_by_mouse_screen, maze_wall_distances_screen, maze_big):
        self.screen = screen
        self.visited_by_mouse_screen = visited_by_mouse_screen
        self.maze_wall_distances_screen = maze_wall_distances_screen
        self.maze_big = maze_big

        # actual position is recorded as a tuple of floats
        # position is rounded just for display
        # and to see if the mouse is still on the maze
        # starting position is in the middle of top left square inside the border
        self.position = (SQUARE_SIZE * 1.5, SQUARE_SIZE * 1.5)
        self.position_rounded = (round(self.position[0]), round(self.position[1]))
        self.speed = SQUARE_SIZE / 50  # pixels per frame
        self.speed_min = CAR_SPEED_MIN_INITIAL
        self.speed_max = CAR_SPEED_MAX_INITIAL
        # could try to set initial direction based on the shape of the
        # maze that's generated but this is fine
        self.direction_radians = math.pi / 3
        self.steering_radians = 0
        self.crashed = False
        self.won = False
        self.position_previous_rounded = self.position_rounded
        self.visited_alpha = np.zeros(screen.get_size(), dtype=np.int16)

        self.stats_info_mouse = {
            "distance": 0.0,
            "frames": 0,
            "average speed": 0.0,
            "speed min": self.speed_min,
            "speed max": self.speed_max,
        }

        self.instructions = {
            "speed": 0.0,
            "speed_delta": 0.0,
            "direction_radians": 0.0,
            "steering_radians": 0.0,
        }
        # keep a list of the 20 last instructions, so we can see where it went wrong
        self.latest_instructions = deque(maxlen=20)

        self.mouse_icon = MouseIcon(self.position_rounded[0], self.position_rounded[1])
        self.mouse_icon_group = pygame.sprite.Group()
        self.mouse_icon_group.add(self.mouse_icon)

        global CAR_TRAIL_CIRCLE_ALPHA
        if CAR_TRAIL_CIRCLE_ALPHA is None:
            CAR_TRAIL_CIRCLE_ALPHA = self.create_trail_circle_alpha(
                CAR_VISITED_PATH_RADIUS, 10
            )

    @staticmethod
    def create_trail_circle_alpha(circle_radius, trail_alpha):
        """create a 2D array of integers that have a given value if they are in a circle,
        otherwise they are zero, except for those around the edge of the circle"""
        x_list = np.arange(-CAR_VISITED_PATH_RADIUS, CAR_VISITED_PATH_RADIUS)
        y_list = np.arange(-CAR_VISITED_PATH_RADIUS, CAR_VISITED_PATH_RADIUS)
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

    def drive(self, draw_frame):
        """to be called every frame to move the mouse along"""
        maze_wall_distances, whiskers = self.get_maze_wall_distances()

        if self.crashed:
            return

        if draw_frame:
            self.draw_lines_to_maze_edge(whiskers)

        self.won = self.check_if_position_wins(self.position[0], self.position[1])
        if self.won:
            self.draw_mouse_finish_location(GREEN)
            return

        self.position_previous_rounded = self.position_rounded

        is_dead_end = self.check_if_dead_end(maze_wall_distances)

        steering_radians_previous = self.steering_radians
        new_steering_angle = 0
        new_steering_radians = 0
        max_maze_distance = 1
        min_maze_distance = 1000

        if is_dead_end:
            # old code from maze v3 that doesn't take into account
            # whether pixels have been visited
            for ted in maze_wall_distances:
                if ted[0][0] == 0:
                    continue

                new_steering_angle += ted[1] * ted[0][1]
                if ted[1] > max_maze_distance:
                    max_maze_distance = ted[1]

            new_steering_radians = 5 * new_steering_angle / max_maze_distance
        else:
            for ted in maze_wall_distances:
                if ted[0][0] == 0:
                    continue

                # this takes into account the sum of the alpha channel of the pixels
                # that have been visited, rather than just the number of visited pixels
                distance_for_angle = max(
                    0,
                    ted[1]
                    * (
                        1
                        - (ted[3] * CAR_VISITED_PATH_AVOIDANCE_FACTOR) / (255 * ted[1])
                    ),
                )

                new_steering_angle += distance_for_angle * ted[0][1]
                if distance_for_angle > max_maze_distance:
                    max_maze_distance = distance_for_angle
                if distance_for_angle < min_maze_distance:
                    min_maze_distance = distance_for_angle

            new_steering_radians = (
                new_steering_angle * CAR_STEERING_MULTIPLIER / max_maze_distance
            )

        # restrict how much the steering can be changed per frame
        if (
            new_steering_radians
            < steering_radians_previous - CAR_STEERING_RADIANS_DELTA_MAX
        ):
            new_steering_radians = (
                steering_radians_previous - CAR_STEERING_RADIANS_DELTA_MAX
            )
        elif (
            new_steering_radians
            > steering_radians_previous + CAR_STEERING_RADIANS_DELTA_MAX
        ):
            new_steering_radians = (
                steering_radians_previous + CAR_STEERING_RADIANS_DELTA_MAX
            )

        # restrict how much the steering can be per frame
        if new_steering_radians > CAR_STEERING_RADIANS_MAX:
            new_steering_radians = CAR_STEERING_RADIANS_MAX
        elif new_steering_radians < -CAR_STEERING_RADIANS_MAX:
            new_steering_radians = -CAR_STEERING_RADIANS_MAX

        new_direction_radians = self.direction_radians + new_steering_radians
        (maze_wall_distance, _, _, _, _,) = self.get_maze_wall_distance(
            self.maze_big,
            self.visited_alpha,
            self.direction_radians,
            self.position_rounded[0],
            self.position_rounded[1],
            new_steering_radians,
        )

        speed_delta = 4 * (maze_wall_distance / CAR_VISION_DISTANCE) - 2

        if speed_delta < CAR_ACCELERATION_MIN:
            speed_delta = CAR_ACCELERATION_MIN

        if speed_delta > CAR_ACCELERATION_MAX:
            speed_delta = CAR_ACCELERATION_MAX

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

        mouse_speed_colour = round(255 * self.speed / self.speed_max)
        mouse_colour = (255 - mouse_speed_colour, mouse_speed_colour, 0)
        self.screen.set_at(self.position_rounded, mouse_colour)

        if draw_frame:
            pygame.display.update(
                pygame.Rect(self.position_rounded[0], self.position_rounded[1], 1, 1)
            )
            pygame.display.update()

        self.update_visited(self.position, self.direction_radians, self.visited_alpha)

        if draw_frame:
            # from https://github.com/pygame/pygame/issues/1244
            surface_alpha = np.array(
                self.visited_by_mouse_screen.get_view("A"), copy=False
            )
            surface_alpha[:, :] = self.visited_alpha

        # decided that the mouse has "crashed" if it has taken more than this
        # distance to complete the maze - better to create a "crashed" status
        if self.stats_info_mouse["distance"] > MAZE_MAX_DISTANCE_TO_COMPLETE:
            self.crashed = True
            self.draw_mouse_finish_location(RED)
            return

        self.stats_info_mouse["frames"] += 1
        self.stats_info_mouse["distance"] += new_speed
        self.stats_info_mouse["average speed"] = (
            self.stats_info_mouse["distance"] // self.stats_info_mouse["frames"]
        )
        self.stats_info_mouse["speed min"] = self.speed_min
        self.stats_info_mouse["speed max"] = self.speed_max

        self.instructions = {
            "speed": self.speed,
            "speed_delta": speed_delta,
            "direction_radians": self.direction_radians,
            "steering_radians": self.steering_radians,
            "maze_wall_distances": maze_wall_distances,
        }
        self.latest_instructions.appendleft(self.instructions)

        if draw_frame:
            self.mouse_icon_group.update(
                self.position_rounded[0],
                self.position_rounded[1],
                self.direction_radians,
            )

        if self.stats_info_mouse["frames"] % FRAMES_BETWEEN_BLURRING_VISITED == 0:
            self.fade_visited(self.visited_alpha)

    @staticmethod
    @jit(nopython=True)
    def update_visited(position, direction_radians, visited_alpha):
        """Updates a mouses 2D array to record that the mouse has been
        in a particular area"""
        # update a 2d numpy array that represents visited pixels.
        # it should add a circular array behind itself.
        # needs to know the angle of travel
        # needs to add but only to a level of saturation (max 255)
        circle_top_left_x = round(
            position[0]
            - CAR_VISITED_PATH_RADIUS * math.cos(direction_radians)
            - CAR_VISITED_PATH_RADIUS
        )
        circle_top_left_y = round(
            position[1]
            - CAR_VISITED_PATH_RADIUS * math.sin(direction_radians)
            - CAR_VISITED_PATH_RADIUS
        )
        # https://stackoverflow.com/questions/9886303/adding-different-sized-shaped-displaced-numpy-matrices
        section_to_update = visited_alpha[
            circle_top_left_x : circle_top_left_x + 2 * CAR_VISITED_PATH_RADIUS,
            circle_top_left_y : circle_top_left_y + 2 * CAR_VISITED_PATH_RADIUS,
        ]
        section_to_update += CAR_TRAIL_CIRCLE_ALPHA  # use array slicing
        np.clip(section_to_update, 0, 255, out=section_to_update)

    @staticmethod
    # @jit(nopython=True)
    def fade_visited(visited_alpha):
        """fades the mouses 2D array of records of where it has been - like
        a scent fading away"""
        # subtract from the alpha channel but stop it going below zero
        visited_alpha -= 1
        np.clip(visited_alpha, 0, 255, out=visited_alpha)
        return

    @staticmethod
    # @jit(nopython=True)
    def check_if_position_wins(position_x, position_y):
        """check if the mouse has reached the end of the maze"""
        if (
            -2 < position_x / SQUARE_SIZE - COLS < -1
            and -2 < position_y / SQUARE_SIZE - ROWS < -1
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
            if math.pi / -2.0 <= ted[0][0] <= math.pi / 2.0 and ted[1] > SQUARE_SIZE:
                return False

        return True

    def get_maze_wall_distances(self):
        """get the distance of the mouse from the edges of the maze along
        a defined list of angles from its direction of travel"""
        mouse_in_maze_passage = self.maze_big[self.position_rounded]
        if not mouse_in_maze_passage:
            self.crashed = True
            self.draw_mouse_finish_location(RED)
            return [], []

        maze_wall_distances = []
        whiskers = []

        for vision_angle_and_weight in CAR_VISION_ANGLES_AND_WEIGHTS:
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
        # or until CAR_VISION_DISTANCE has been reached
        search_angle_radians = direction_radians + vision_angle
        delta_x = math.cos(search_angle_radians)
        delta_y = math.sin(search_angle_radians)

        edge_distance = np.int64(0)
        visited_count = np.int64(0)
        visited_alpha_total = np.int64(0)

        for i in range(CAR_VISITED_PATH_RADIUS, CAR_VISION_DISTANCE):
            edge_distance = i
            # incrementing test_x e.g. test_x += delta_x
            # makes the function slower by ~10%
            test_x = position_rounded_x + i * delta_x
            test_y = position_rounded_y + i * delta_y
            # saving the rounded values, rather than rounding twice
            # improves performance of this function by ~5% and by ~3% overall
            test_x_round = round(test_x)
            test_y_round = round(test_y)
            if maze_big[test_x_round][test_y_round] is False:
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

    def draw_lines_to_maze_edge(self, whiskers):
        """draw lines from the mouse to the edge of the maze"""
        self.maze_wall_distances_screen.fill((0, 0, 0, 0))
        pygame.draw.lines(self.maze_wall_distances_screen, WHITE, False, whiskers)

    def draw_mouse_finish_location(self, highlight_colour):
        """draw where the mouse finishes"""
        finish_radius = SQUARE_SIZE // 2
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

    # main loop
    while running:
        stats_info_global["Total frames"] += 1
        if stats_info_global["Total frames"] > 10000000:
            pygame.time.wait(1000000)
            break
        elapsed_time = time.monotonic() - start_time
        stats_info_global["FPS"] = stats_info_global["Total frames"] // elapsed_time
        if new_maze_and_mouse_needed:
            # create the maze and draw it on the background
            if mouse is not None:
                stats_info_global["Total mazes"] += 1
                if mouse.won:
                    stats_info_global["Success count"] += 1
                    stats_info_global["Total successes"] += 1
                    stats_info_global["Max successes in a row"] = max(
                        stats_info_global["Max successes in a row"],
                        stats_info_global["Success count"],
                    )
                elif mouse.crashed:
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
            maze = Maze(ROWS, COLS)
            if saved_maze_counter < saved_maze_count:
                maze.load(MAZE_DIRECTORY + saved_maze_files[saved_maze_counter])
                saved_maze_counter += 1
            else:
                maze.create()
            maze_drawer = MazeDrawer(maze, window_size, background)
            maze_drawer.draw_maze(maze.maze_tiny, PURE_WHITE, BLACK)
            maze_drawer.draw_start(background, SQUARE_SIZE, (100, 100, 100))
            maze_drawer.draw_finish(
                window_size, background, SQUARE_SIZE, (100, 100, 100)
            )
            mouse = Mouse(
                background,
                visited_by_mouse_screen,
                maze_wall_distances_screen,
                maze.maze_big,
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
                    sys.exit()
                    break
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    mouse.speed_increase()
                elif event.key == pygame.K_LEFT:
                    mouse.speed_decrease()
                elif event.key == pygame.K_s:
                    maze.save(timestamp=False)

        if paused:
            continue

        if (
            stats_info_global["Total frames"] % FRAME_DISPLAY_RATE == 0
            or mouse.won is True
            or mouse.crashed is True
        ):
            draw_frame = True
        else:
            draw_frame = False

        if not mouse.crashed and not mouse.won:
            mouse.drive(draw_frame)

        if draw_frame:
            stats_update(stats_surface, mouse.stats_info_mouse, stats_info_global)
            pygame.display.flip()
            screen.blit(background, (0, 0))
            screen.blit(mouse.visited_by_mouse_screen, (0, 0))
            screen.blit(mouse.maze_wall_distances_screen, (0, 0))
            mouse.mouse_icon_group.draw(screen)
            screen.blit(stats_surface, (0, 0))
            pygame.display.update()

        # clock.tick(400)

        if mouse.won:
            new_maze_and_mouse_needed = True
            # pygame.time.wait(1000)

        if mouse.crashed:
            # don't re-save a maze that has been loaded from a saved maze
            # however a log of those saved ones that have failed again would be n
            if not maze.from_saved:
                maze.save(timestamp=True)
            new_maze_and_mouse_needed = True
            # pygame.time.wait(1000)


# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__ == "__main__":
    # call the main function
    main()

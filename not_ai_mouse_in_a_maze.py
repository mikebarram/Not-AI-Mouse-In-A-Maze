'''
*** HOW TO PROFILE ***
Install gprof2dot and dot
On Windows, to install dot, install Graphviz and make sure the path to the bin folder that contains dot.exe is in the PATH environment variable (or include the full path to dot when you call it)

Get the programme to start and end e.g. in main add
if statsInfoGlobal["Total frames"] > 10000:
            break
In the terminal run
python -m cProfile -o output.pstats maze-18-JIT-4650-FPS.py
gprof2dot -f pstats output.pstats | "C:\Program Files\Graphviz\bin\dot.exe" -Tpng -o outputJIT.png
Have a look at output.png
'''

# A program which uses recursive backtracking to generate a maze
# https://aryanab.medium.com/maze-generation-recursive-backtracking-5981bc5cc766
# https://github.com/AryanAb/MazeGenerator/blob/master/backtracking.py
# A good alternative might be https://github.com/AryanAb/MazeGenerator/blob/master/hunt_and_kill.py

# import the pygame module, so you can use it
from asyncio.windows_events import NULL
from curses import KEY_LEFT
from queue import Empty
import pygame, sys
import numpy as np
import random
import math
import time
import os, os.path
from scipy.interpolate import splprep, splev
from scipy.ndimage import uniform_filter1d
from collections import deque
from numba import jit, njit
from enum import Enum

sys.setrecursionlimit(8000)

BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
START_COLOUR = (200, 200, 0)
SQUARE_SIZE = 70
ROWS = 13
COLS = 19
TRACK_MAX_DISTANCE_TO_COMPLETE = ROWS * COLS * SQUARE_SIZE ** 1.5 # car will "crash" if it takes more than this distance to complete the maze
MAZE_DIRECTORY = f'mazes\\{COLS}x{ROWS}\\' # mazes that the mouse fails to solve are saved here. Any mazes in here are reloaded at the start of session.

WINDOW_WIDTH = COLS * SQUARE_SIZE
WINDOW_HEIGHT = ROWS * SQUARE_SIZE

# Mice can only look so far ahead. Needs to be somewhat larger than drid size for the maze
CAR_VISION_DISTANCE = round(2.0 * SQUARE_SIZE)
CAR_VISION_ANGLES_AND_WEIGHTS = (
    (math.radians(0.0), 1.0/2.0), 
    (math.radians(-15.0), -1.0/5.0), 
    (math.radians(15.0), 1.0/5.0), 
    (math.radians(-30.0), -1.0/6.0), 
    (math.radians(30.0), 1.0/6.0), 
    (math.radians(-45.0), -1.0/6.0), 
    (math.radians(45.0), 1.0/6.0), 
    (math.radians(-60.0), -1.0/7.0), 
    (math.radians(60.0), 1.0/7.0), 
    (math.radians(-90.0), -1.0/7.0), 
    (math.radians(90.0), 1.0/7.0)
    ) # 0 must be first, 90 degrees needed to get out of dead ends
CAR_SPEED_MIN_INITIAL = 2 # pixels per frame
CAR_SPEED_MAX_INITIAL = 5 # pixels per frame
CAR_SPEED_MIN = CAR_SPEED_MIN_INITIAL # pixels per frame
CAR_SPEED_MAX = CAR_SPEED_MAX_INITIAL # pixels per frame
CAR_ACCELERATION_MIN = -3 # change in speed in pixels per frame
CAR_ACCELERATION_MAX = 2 # change in speed in pixels per frame
CAR_STEERING_RADIANS_MAX = math.radians(45)
CAR_STEERING_RADIANS_DELTA_MAX = math.radians(45)
CAR_STEERING_MULTIPLIER = 2.5
CAR_PATH_COLOUR = RED
CAR_COLOUR = GREEN
CAR_VISITED_PATH_RADIUS = 20
CAR_VISITED_PATH_AVOIDANCE_FACTOR = 1.25 * SQUARE_SIZE/ (2 * CAR_VISITED_PATH_RADIUS) # how much wider the maze is than the path
CAR_VISITED_PATH_AVOIDANCE_FACTOR_FOR_DEAD_END = 0.8
CAR_WHEN_TO_STEER_FACTOR = 2.0
CAR_VISITED_COLOUR = (100,100,100,50)
CAR_VISITED_FADE_LEVEL = 1.17
FRAMES_BETWEEN_BLURRING_VISITED = (ROWS * COLS)//12 # the bigger the maze, the longer it could be before returning to a visited bit of the maze
FRAME_DISPLAY_RATE = 500

def CreateTrailCircleAlpha(CAR_VISITED_PATH_RADIUS, trail_alpha):
    x = np.arange(-CAR_VISITED_PATH_RADIUS, CAR_VISITED_PATH_RADIUS)
    y = np.arange(-CAR_VISITED_PATH_RADIUS, CAR_VISITED_PATH_RADIUS)
    arrAlpha = np.zeros((y.size, x.size), dtype=np.int16)

    cx = 0.
    cy = 0.
    r = CAR_VISITED_PATH_RADIUS

    # The two lines below could be merged, but I stored the mask
    # for code clarity.
    maskOuter = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < r**2
    maskMiddle = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < (r-4)**2
    maskInner = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < (r-8)**2

    arrAlpha[maskOuter] = trail_alpha//3
    arrAlpha[maskMiddle] = (2*trail_alpha)//3
    arrAlpha[maskInner] = trail_alpha
    # would be nice for the values to be lower around the edge of the circle
    return arrAlpha

CAR_TRAIL_CIRCLE_ALPHA = CreateTrailCircleAlpha(CAR_VISITED_PATH_RADIUS, 10)

class Directions(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    
class Backtracking:

    def __init__(self, height, width):

        if width % 2 == 0:
            width += 1
        if height % 2 == 0:
            height += 1

        self.width = width
        self.height = height

    def createMaze(self):
        maze = np.ones((self.height, self.width), dtype=float)

        for i in range(self.height):
            for j in range(self.width):
                if i % 2 == 1 or j % 2 == 1:
                    maze[i, j] = 0
                if i == 0 or j == 0 or i == self.height - 1 or j == self.width - 1:
                    maze[i, j] = 0.5

        sx = random.choice(range(2, self.width - 2, 2))
        sy = random.choice(range(2, self.height - 2, 2))

        self.generator(sx, sy, maze)

        for i in range(self.height):
            for j in range(self.width):
                if maze[i, j] == 0.5:
                    maze[i, j] = 1

        maze = maze * 255.0

        return maze

    def generator(self, cx, cy, grid):
        grid[cy, cx] = 0.5

        if grid[cy - 2, cx] == 0.5 and grid[cy + 2, cx] == 0.5 and grid[cy, cx - 2] == 0.5 and grid[cy, cx + 2] == 0.5:
            pass
        else:
            li = [1, 2, 3, 4]
            while len(li) > 0:
                dir = random.choice(li)
                li.remove(dir)

                if dir == Directions.UP.value:
                    ny = cy - 2
                    my = cy - 1
                elif dir == Directions.DOWN.value:
                    ny = cy + 2
                    my = cy + 1
                else:
                    ny = cy
                    my = cy

                if dir == Directions.LEFT.value:
                    nx = cx - 2
                    mx = cx - 1
                elif dir == Directions.RIGHT.value:
                    nx = cx + 2
                    mx = cx + 1
                else:
                    nx = cx
                    mx = cx

                if grid[ny, nx] != 0.5:
                    grid[my, mx] = 0.5
                    self.generator(nx, ny, grid)

class CarIcon(pygame.sprite.Sprite):
    def __init__(self, pos_x, pos_y):
        super().__init__()
        self.image = pygame.image.load("images/mouse.png") #40x65 https://flyclipart.com/lab-mouse-template-clip-art-free-mouse-clipart-791054#
        self.rect = self.image.get_rect()
        self.rect.center = [pos_x,pos_y]
        self.image_original = self.image
        self.rect_original = self.rect

    def update(self, pos_x, pos_y, angle_radians):
        self.image, self.rect = self.rot_center(self.image_original, self.rect_original, angle_radians)
        self.rect.center = [pos_x,pos_y]

    def rot_center(self, image, rect, angle_radians):
        # rotate an image while keeping its center"""
        rot_image = pygame.transform.rotate(image, 270 - math.degrees(angle_radians))
        rot_rect = rot_image.get_rect(center=rect.center)
        return rot_image,rot_rect
    
class Car():
    def __init__(self, screen, visitedByCarScreen, trackDistancesScreen, track):
        self.screen = screen
        self.visitedByCarScreen = visitedByCarScreen
        self.trackDistancesScreen = trackDistancesScreen
        self.track = track

        #if isinstance(position, tuple):
        #    if isinstance(position[0], int) and isinstance(position[1], int):
        #        self.position = (int(position[0]),int(position[1]))
        #    else:            
        #        raise (TypeError, "expected a tuple of integers")
        #else:
        #    raise (TypeError, "expected a tuple")
        
        # actual position is recorded as a tuple of floats
        # position is rounded just for display and to see if the car is still on the track
        self.position = (SQUARE_SIZE*1.5, SQUARE_SIZE*1.5) # middle of top left square inside the border
        self.position_rounded = (round(self.position[0]),round(self.position[1]))
        self.speed = SQUARE_SIZE/50  # pixels per frame
        self.direction_radians = math.pi/3 # could try to set initial direction based on the shape of the maze that's generated but this is fine
        self.steering_radians = 0
        self.crashed = False
        self.won = False
        self.position_previous_rounded = self.position_rounded
        self.visitedAlpha = np.zeros(screen.get_size(), dtype=np.int16)
        
        self.statsInfoCar = {
            "distance":0.0,
            "frames":0,
            "average speed":0.0,
            "CAR_SPEED_MIN":CAR_SPEED_MIN,
            "CAR_SPEED_MAX":CAR_SPEED_MAX
            }

        self.instructions = {
            "speed":0.0,
            "speed_delta":0.0,
            "direction_radians":0.0,
            "steering_radians":0.0,
            }
        self.latestInstructions = deque(maxlen=20) # keep a list of the 20 last instructions, so we can see where it went wrong
        
        self.carIcon = CarIcon(self.position_rounded[0],self.position_rounded[1])
        self.carIconGroup = pygame.sprite.Group()
        self.carIconGroup.add(self.carIcon)

    def Drive(self, drawFrame):
        track_edge_distances, whiskers = self.GetTrackEdgeDistances()

        if self.crashed:
            return

        if drawFrame:
            self.DrawLinesToTrackEdge(whiskers)
        
        self.won = self.CheckIfPositionWins(self.position[0], self.position[1])
        if self.won:            
            self.DrawCarFinishLocation(GREEN)
            return
        
        self.position_previous_rounded = self.position_rounded

        IsDeadEnd = self.CheckIfDeadEnd(track_edge_distances)

        steering_radians_previous = self.steering_radians
        new_steering_angle = 0
        new_steering_radians = 0
        max_track_distance = 1
        min_track_distance = 1000
        max_ted = []
        
        if IsDeadEnd:
            # some old code from maze v3 that doesn't take into account whether pixels have been visited       
            for ted in track_edge_distances:
                if ted[0][0] == 0:
                    continue

                new_steering_angle += ted[1] * ted[0][1]
                if ted[1] > max_track_distance: max_track_distance = ted[1]

            new_steering_radians = 5 * new_steering_angle / max_track_distance
        else:
            for ted in track_edge_distances:
                if ted[0][0] == 0:
                    continue

                distance_for_angle = max(0, ted[1] * (1 - (ted[3] * CAR_VISITED_PATH_AVOIDANCE_FACTOR) / (255 * ted[1])))  # this takes into account how many of the pixels have been visited but not alpha for each pixel (summed in ted[3])
                    
                new_steering_angle += distance_for_angle * ted[0][1]
                if distance_for_angle > max_track_distance: 
                    max_track_distance = distance_for_angle
                    max_ted = ted
                if distance_for_angle < min_track_distance: min_track_distance = distance_for_angle

            new_steering_radians =  new_steering_angle * CAR_STEERING_MULTIPLIER / max_track_distance

        # restrict how much the steering can be changed per frame
        if new_steering_radians < steering_radians_previous - CAR_STEERING_RADIANS_DELTA_MAX:
            new_steering_radians = steering_radians_previous - CAR_STEERING_RADIANS_DELTA_MAX
        elif new_steering_radians > steering_radians_previous + CAR_STEERING_RADIANS_DELTA_MAX:
            new_steering_radians = steering_radians_previous + CAR_STEERING_RADIANS_DELTA_MAX

        # restrict how much the steering can be per frame
        if new_steering_radians > CAR_STEERING_RADIANS_MAX:
            new_steering_radians = CAR_STEERING_RADIANS_MAX
        elif new_steering_radians < -CAR_STEERING_RADIANS_MAX:
            new_steering_radians = -CAR_STEERING_RADIANS_MAX

        new_direction_radians = self.direction_radians + new_steering_radians #* speed_new # direction changes more per frame if you're goig faster
        track_edge_distance, visited_count, visited_alpha_total, edge_x, edge_y = self.GetTrackEdgeDistance(self.track.track_pixels, self.visitedAlpha, self.direction_radians, self.position_rounded[0], self.position_rounded[1], new_steering_radians)
        
        speed_delta = 4 * (track_edge_distance/CAR_VISION_DISTANCE) - 2
        
        if speed_delta < CAR_ACCELERATION_MIN:
            speed_delta = CAR_ACCELERATION_MIN
        
        if speed_delta > CAR_ACCELERATION_MAX:
            speed_delta = CAR_ACCELERATION_MAX

        new_speed = self.speed + speed_delta
        
        if new_speed < CAR_SPEED_MIN:
            new_speed = CAR_SPEED_MIN
        
        if new_speed > CAR_SPEED_MAX:
            new_speed = CAR_SPEED_MAX

        new_position = (self.position[0] + new_speed * math.cos(new_direction_radians), self.position[1] + new_speed * math.sin(new_direction_radians))

        self.speed = new_speed
        self.steering_radians =  new_steering_radians
        self.direction_radians = new_direction_radians

        self.position = new_position
        self.position_rounded = (round(self.position[0]),round(self.position[1]))    

        car_speed_colour = round(255 * self.speed / CAR_SPEED_MAX)
        car_colour = (255 - car_speed_colour, car_speed_colour, 0)
        self.screen.set_at(self.position_rounded, car_colour)

        if drawFrame:
            pygame.display.update(pygame.Rect(self.position_rounded[0],self.position_rounded[1],1,1))
            pygame.display.update()

        self.UpdateVisited(self.position, self.direction_radians, self.visitedAlpha)
            
        if drawFrame:
            # from https://github.com/pygame/pygame/issues/1244
            surface_alpha = np.array(self.visitedByCarScreen.get_view('A'), copy=False)
            surface_alpha[:,:] = self.visitedAlpha

        # decided that the car has "crashed" if it has taken more than this distance to complete the maze
        if self.statsInfoCar["distance"] > TRACK_MAX_DISTANCE_TO_COMPLETE:
            self.crashed = True
            self.DrawCarFinishLocation(RED)
            return

        self.statsInfoCar["frames"] += 1
        self.statsInfoCar["distance"] += new_speed
        self.statsInfoCar["average speed"] = self.statsInfoCar["distance"] // self.statsInfoCar["frames"]
        self.statsInfoCar["CAR_SPEED_MIN"] = CAR_SPEED_MIN
        self.statsInfoCar["CAR_SPEED_MAX"] = CAR_SPEED_MAX
    
        self.instructions = {
            "speed":self.speed,
            "speed_delta":speed_delta,
            "direction_radians":self.direction_radians,
            "steering_radians":self.steering_radians,
            "track_edge_distances":track_edge_distances
            }
        self.latestInstructions.appendleft(self.instructions)
            
        if drawFrame:
            self.carIconGroup.update(self.position_rounded[0], self.position_rounded[1], self.direction_radians)

        if self.statsInfoCar["frames"] % FRAMES_BETWEEN_BLURRING_VISITED == 0:
            self.FadeVisited(self.visitedAlpha)

    @staticmethod
    @jit(nopython=True)                             
    def UpdateVisited(position, direction_radians, visitedAlpha):
        # update a 2d numpy array that represents visited pixels.
        # it should add a circular array behind itself.
        # needs to know the angle of travel
        # needs to add but only to a level of saturation (max 255)
        circle_top_left_x = round(position[0] - CAR_VISITED_PATH_RADIUS * math.cos(direction_radians) - CAR_VISITED_PATH_RADIUS)
        circle_top_left_y = round(position[1] - CAR_VISITED_PATH_RADIUS * math.sin(direction_radians) - CAR_VISITED_PATH_RADIUS)
        #https://stackoverflow.com/questions/9886303/adding-different-sized-shaped-displaced-numpy-matrices
        sectionToUpdate = visitedAlpha[circle_top_left_x:circle_top_left_x+2*CAR_VISITED_PATH_RADIUS,circle_top_left_y:circle_top_left_y+2*CAR_VISITED_PATH_RADIUS]
        sectionToUpdate += CAR_TRAIL_CIRCLE_ALPHA  # use array slicing
        np.clip(sectionToUpdate, 0, 255, out=sectionToUpdate)      

    @staticmethod
    #@jit(nopython=True)                             
    def FadeVisited(visitedAlpha):
        # subtract from the alpha channel but stop it going below zero
        visitedAlpha -= 1
        np.clip(visitedAlpha, 0, 255, out=visitedAlpha)
        return
    
    @staticmethod
    #@jit(nopython=True)                             
    def CheckIfPositionWins(x, y):
        if -2 < x/SQUARE_SIZE-COLS < -1 and -2 < y/SQUARE_SIZE-ROWS < -1:
            return True
        else:
            return False

    @staticmethod
    #@jit(nopython=True)                             
    def CheckIfDeadEnd(track_edge_distances):
        # if any of the distances is more than the size of a square, then it's not a dead end
        for ted in track_edge_distances:
            if math.pi/-2.0<=ted[0][0]<=math.pi/2.0 and ted[1] > SQUARE_SIZE:
                return False

        return True

    def GetTrackEdgeDistances(self):    
        car_on_track = self.track.track_pixels[self.position_rounded]
        if not car_on_track:
            self.crashed = True
            self.DrawCarFinishLocation(RED)
            return [], []

        track_edge_distances = []
        whiskers = []

        for vision_angle_and_weight in CAR_VISION_ANGLES_AND_WEIGHTS:
            track_edge_distance, visited_count, visited_alpha_total, edge_x, edge_y = self.GetTrackEdgeDistance(self.track.track_pixels, self.visitedAlpha, self.direction_radians, self.position_rounded[0], self.position_rounded[1], vision_angle_and_weight[0])
            whiskers += [self.position_rounded, (edge_x, edge_y)]

            track_edge_distances.append((vision_angle_and_weight, track_edge_distance, visited_count, visited_alpha_total))
        
        return track_edge_distances, whiskers
    
    @staticmethod
    @jit(nopython=True)                             
    def GetTrackEdgeDistance(track_pixels, visitedAlpha, direction_radians, position_rounded_x, position_rounded_y, vision_angle):
        # from x,y follow a line at vision_angle until no longer on the track
        # or until CAR_VISION_DISTANCE has been reached
        search_angle_radians = direction_radians + vision_angle
        delta_x = math.cos(search_angle_radians)
        delta_y = math.sin(search_angle_radians)

        edge_distance = np.int64(0)
        visited_count = np.int64(0)
        visited_alpha_total = np.int64(0)

        for i in range(CAR_VISITED_PATH_RADIUS, CAR_VISION_DISTANCE):
            edge_distance = i
            # incrementing test_x e.g. test_x += delta_x makes the function slower by ~10%
            test_x = position_rounded_x + i * delta_x
            test_y = position_rounded_y + i * delta_y
            # saving the rounded values, rather than rounding twice improves performance of this function by ~5% and by ~3% overall
            test_x_round = round(test_x)
            test_y_round = round(test_y)
            if track_pixels[test_x_round][test_y_round] == False:
                break
            
            visited_alpha = visitedAlpha[test_x_round,test_y_round]
            visited_alpha_total += visited_alpha
            if visited_alpha > 0:
                visited_count += 1 

        return edge_distance, visited_count, visited_alpha_total, test_x_round, test_y_round

    def DrawLinesToTrackEdge(self, whiskers):
        self.trackDistancesScreen.fill((0,0,0,0))
        pygame.draw.lines(self.trackDistancesScreen, WHITE, False, whiskers)
    
    def DrawCarFinishLocation(self, highlight_colour):
        finish_radius = SQUARE_SIZE//2
        pygame.draw.circle(self.screen, highlight_colour, self.position_rounded, finish_radius, width=2)
        finish_zone = pygame.Rect(self.position_rounded[0] - finish_radius, self.position_rounded[1] - finish_radius, 2 * finish_radius, 2 * finish_radius)
        pygame.display.update(finish_zone)

class Track():
    def __init__(self, window_size, track_surface) -> None:
        self.window_size = window_size
        self.track_surface = track_surface
        self.track = []
        self.scaled_track = []
        self.interpolated_scaled_track = []
        self.track_widths = []
        self.track_pixels = []
        self.maze = []
        self.fromSaved = False
    
    def Create(self):    
        pygame.display.set_caption("New maze")
        maze = self.GetNewMaze()
        maze = self.SetMazeEndPoints(maze)
        self.maze = maze
        self.SetScaledMazeSurface(maze)
        self.track_surface = self.SetTrackPixelsFromMazeSurface(self.track_surface)

    def Save(self, timestamp):
        filename = 'maze.txt'
        if timestamp:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            filename = MAZE_DIRECTORY + 'maze_' + timestr + '.txt'
        np.savetxt(filename, self.maze, fmt='%s')

    def Load(self, filename):
        pygame.display.set_caption("Maze: " + filename)
        maze = np.loadtxt(filename, dtype=np.float32)
        self.maze = maze
        self.SetScaledMazeSurface(maze)
        self.track_surface = self.SetTrackPixelsFromMazeSurface(self.track_surface)
        self.fromSaved = True

    def GetNewMaze(self):
        MAZE_HEIGHT = ROWS + 1
        MAZE_WIDTH = COLS + 1

        backtracking = Backtracking(MAZE_HEIGHT, MAZE_WIDTH)
        maze = backtracking.createMaze()
        # remove the outer elements of the array
        maze = np.delete(maze, [0, maze.shape[0]-1], axis=0)
        maze = np.delete(maze, [0, maze.shape[1]-1], axis=1)

        return maze

    def SetMazeEndPoints(self, maze):
        maze[1][1] = maze[1][1]/2
        maze[maze.shape[0]-2][maze.shape[1]-2] = maze[maze.shape[0]-2][maze.shape[1]-2]/2
        return maze

    def SetScaledMazeSurface(self, maze):
        # draw the track in lots of red circles on a black surface
        surf = pygame.Surface((COLS,ROWS))
        surf.fill(BLACK)
        for i in range(0, maze.shape[1]):
            for j in range(0, maze.shape[0]):
                r = pygame.Rect(i,j,1,1)
                colour_val = round(maze[j][i])
                #pygame.draw.rect(surf, (colour_val,0,0), r)
                surf.set_at((i,j),(colour_val,colour_val,colour_val))
        
        pygame.transform.scale(surf, self.window_size, self.track_surface)

    def SetTrackPixelsFromMazeSurface(self, maze_surface):
        track_pixels_surface = pygame.Surface(self.window_size)
        pygame.transform.threshold(track_pixels_surface, maze_surface, search_color=(255,255,255), threshold=(128,128,128), set_color=(255,0,0), set_behavior=1, inverse_set=True)

        # get an array from the screen identifying where the track is
        tp = pygame.surfarray.pixels_red(track_pixels_surface)
        # reduce this down to an array of booleans where 255 becomes True
        self.track_pixels = tp.astype(dtype=bool)
        return track_pixels_surface

def StatsUpdate(statsSurface, statsInfoCar, statsInfoGlobal):
    statsSurface.fill((0, 0, 0, 0))
    font = pygame.font.SysFont('Arial', 12, bold=False)
    textTop = 0
    
    for stats in (statsInfoCar, statsInfoGlobal):
        for k,v in stats.items():
            img = font.render(k + ': ' + str(round(v)), True,
                    pygame.Color(BLACK),
                    pygame.Color(WHITE))
            imgSize = img.get_size()
            textTop += imgSize[1] + 10
            statsSurface.blit(img, (10,textTop))

# define a main function
def main():
    # https://stackoverflow.com/questions/18002794/local-variable-referenced-before-assignment
    global CAR_SPEED_MIN
    global CAR_SPEED_MAX

    # initialize the pygame module
    pygame.init()
    clock = pygame.time.Clock()
    startTime = time.monotonic()
    # load and set the logo
    logo = pygame.image.load("logo32x32.png")
    pygame.display.set_icon(logo)
    pygame.display.set_caption("AI car")
    
    window_size = (WINDOW_WIDTH,WINDOW_HEIGHT)
     
    # create a surface on screen that has the size defined globally
    screen = pygame.display.set_mode(window_size)
    background = pygame.Surface(window_size)

    visitedByCarScreen = pygame.Surface(window_size, pygame.SRCALPHA, 32)
    visitedByCarScreen = visitedByCarScreen.convert_alpha()
    visitedByCarScreen.fill((0,0,0,0))

    trackDistancesScreen = pygame.Surface(window_size, pygame.SRCALPHA, 32)
    trackDistancesScreen = trackDistancesScreen.convert_alpha()
    trackDistancesScreenCopy = trackDistancesScreen.copy()
    
    statsSurface = pygame.Surface(window_size)
    statsSurface = statsSurface.convert_alpha()
    statsSurface.fill((0, 0, 0, 0))
    
    # define a variable to control the main loop
    running = True
    paused = False
    drawFrame = False
    newTrackAndCarNeeded = True
    if not os.path.exists(MAZE_DIRECTORY):
        os.mkdir(MAZE_DIRECTORY)
    savedMazeFiles = [name for name in os.listdir(MAZE_DIRECTORY) if os.path.isfile(os.path.join(MAZE_DIRECTORY, name))]
    savedMazeCount = len(savedMazeFiles)
    savedMazeCounter = 0
    statsInfoGlobal = {
        "Success count" : 0,
        "Max successes in a row" : 0,
        "Total successes": 0,
        "Total frames" : 0,
        "FPS" : 0
    }

    # main loop
    while running:
        statsInfoGlobal["Total frames"] += 1
        if statsInfoGlobal["Total frames"] > 1000000:
            break
        elapsedTime = time.monotonic() - startTime
        statsInfoGlobal["FPS"] = statsInfoGlobal["Total frames"] // elapsedTime
        if newTrackAndCarNeeded:
            # create the track and draw it on the background
            try:
                car
            except NameError:
                car_exists = False
            else:
                if car.won:
                    statsInfoGlobal["Success count"] += 1
                    statsInfoGlobal["Total successes"] += 1
                    statsInfoGlobal["Max successes in a row"] = max(statsInfoGlobal["Max successes in a row"],statsInfoGlobal["Success count"])
                elif car.crashed:
                    statsInfoGlobal["Max successes in a row"] = max(statsInfoGlobal["Max successes in a row"],statsInfoGlobal["Success count"])
                    statsInfoGlobal["Success count"] = 0

            CAR_SPEED_MIN = CAR_SPEED_MIN_INITIAL # pixels per frame
            CAR_SPEED_MAX = CAR_SPEED_MAX_INITIAL # pixels per frame
            track = Track(window_size, background)
            if savedMazeCounter < savedMazeCount:
                track.Load(MAZE_DIRECTORY + savedMazeFiles[savedMazeCounter])
                savedMazeCounter += 1
            else:
                track.Create()
            car = Car(background, visitedByCarScreen, trackDistancesScreen, track)            
            newTrackAndCarNeeded = False

        # event handling, gets all event from the event queue
        for event in pygame.event.get():
            # only do something if the event is of type QUIT
            if event.type == pygame.QUIT:
                # change the value to False, to exit the main loop
                running = False
                pygame.quit
                sys.exit()
                break

            # for the next bit, on windows, you need to:
            # pip install windows-curses
            # https://stackoverflow.com/questions/35850362/importerror-no-module-named-curses-when-trying-to-import-blessings
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n:
                    newTrackAndCarNeeded = True
                    continue
                elif event.key == pygame.K_ESCAPE:
                    running = False
                    pygame.quit
                    sys.exit()
                    break
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    CAR_SPEED_MIN += 1
                    CAR_SPEED_MAX += 1
                elif event.key == pygame.K_LEFT:
                    CAR_SPEED_MIN -= 1
                    CAR_SPEED_MAX -= 1
                elif event.key == pygame.K_s:
                    track.Save(timestamp=False)
                elif event.key == pygame.K_l:
                    track = Track(window_size, background)
                    track.Load('maze.txt')
                    car = Car(background, visitedByCarScreen, trackDistancesScreen, track)  
        
        if paused:
             continue
        
        if statsInfoGlobal["Total frames"] % FRAME_DISPLAY_RATE == 0 or car.won == True or car.crashed == True:
            drawFrame = True
        else:
            drawFrame = False
            
        if not car.crashed and not car.won:
            car.Drive(drawFrame)
        
        if drawFrame:
            StatsUpdate(statsSurface, car.statsInfoCar, statsInfoGlobal)
            pygame.display.flip()
            screen.blit(background, (0,0))
            screen.blit(car.visitedByCarScreen, (0,0))
            screen.blit(car.trackDistancesScreen, (0,0))
            car.carIconGroup.draw(screen)
            screen.blit(statsSurface, (0,0))
            pygame.display.update()

        #clock.tick(400)

        if car.won:
            newTrackAndCarNeeded = True
            #pygame.time.wait(1000)

        if car.crashed:
            if not track.fromSaved: # don't re-save a track that has been loaded from a saved track - however a log of those saved ones that have failed again would be n
                track.Save(timestamp=True)
            newTrackAndCarNeeded = True
            #pygame.time.wait(1000)

# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__=="__main__":
    # call the main function
    main()
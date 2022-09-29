# A program which uses recursive backtracking to generate a maze
# https://aryanab.medium.com/maze-generation-recursive-backtracking-5981bc5cc766
# https://github.com/AryanAb/MazeGenerator/blob/master/backtracking.py
# A good alternative might be https://github.com/AryanAb/MazeGenerator/blob/master/hunt_and_kill.py

# import the pygame module, so you can use it
from asyncio.windows_events import NULL
from curses import KEY_LEFT
import pygame, sys
import numpy as np
import random
import math
import time
from scipy.interpolate import splprep, splev
from scipy.ndimage import uniform_filter1d
from collections import deque
from numba import jit, njit
#from functools import cache

from enum import Enum
import cv2

sys.setrecursionlimit(8000)

BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
START_COLOUR = (200, 200, 0)
SQUARE_SIZE = 70
ROWS = 13
COLS = 19
TRACK_MIN_WIDTH = 15
TRACK_MAX_WIDTH = 42
TRACK_CURVE_POINTS = ROWS * COLS * SQUARE_SIZE // 5
TRACK_MIDLINE_COLOUR = WHITE
TRACK_MAX_DISTANCE_TO_COMPLETE = ROWS * COLS * SQUARE_SIZE ** 1.5 # car will "crash" if it takes more than this distance to complete the maze

WINDOW_WIDTH = COLS * SQUARE_SIZE
WINDOW_HEIGHT = ROWS * SQUARE_SIZE

# cars can only look so far ahead. Needs to be somewhat larger than the maximum track width - try seting that distance to the size of a grid square
CAR_VISION_DISTANCE = round(2.0 * SQUARE_SIZE)
#CAR_VISION_ANGLES = (0, -20, 20, -45, 45, -60, 60, -90, 90) # 0 must be first
CAR_VISION_ANGLES_AND_WEIGHTS = (
    (math.radians(0.0), 1.0/2.0), 
    (math.radians(-20.0), -1.0/5.0), 
    (math.radians(20.0), 1.0/5.0), 
    (math.radians(-45.0), -1.0/6.0), 
    (math.radians(45.0), 1.0/6.0), 
    (math.radians(-60.0), -1.0/7.0), 
    (math.radians(60.0), 1.0/7.0), 
    (math.radians(-90.0), -1.0/8.0), 
    (math.radians(90.0), 1.0/8.0)
    ) # 0 must be first
CAR_SPEED_MIN_INITIAL = 2 # pixels per frame
CAR_SPEED_MAX_INITIAL = 5 # pixels per frame
CAR_SPEED_MIN = CAR_SPEED_MIN_INITIAL # pixels per frame
CAR_SPEED_MAX = CAR_SPEED_MAX_INITIAL # pixels per frame
CAR_ACCELERATION_MIN = -3 # change in speed in pixels per frame
CAR_ACCELERATION_MAX = 2 # change in speed in pixels per frame
CAR_STEERING_RADIANS_MAX = math.radians(45)
CAR_STEERING_RADIANS_DELTA_MAX = math.radians(45)
CAR_STEERING_MULTIPLIER = 1.5
CAR_PATH_COLOUR = RED
CAR_COLOUR = GREEN
CAR_VISITED_PATH_RADIUS = 20
CAR_VISITED_PATH_AVOIDANCE_FACTOR = 0.8
CAR_WHEN_TO_STEER_FACTOR = 1.2
CAR_VISITED_COLOUR = (100,100,100,50)
CAR_VISITED_FADE_LEVEL = 1.17
FRAMES_BETWEEN_BLURRING_VISITED = (ROWS * COLS)//1.5 # the bigger the maze, the longer it could be before returning to a visited bit of the maze

class Directions(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    
class Backtracking:

    def __init__(self, height, width, path, displayMaze):

        print("Using OpenCV version: " + cv2.__version__)

        if width % 2 == 0:
            width += 1
        if height % 2 == 0:
            height += 1

        self.width = width
        self.height = height
        self.path = path
        self.displayMaze = displayMaze

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

        # original
        #maze[1, 2] = 1
        #maze[self.height - 2, self.width - 3] = 1
        # set start and end to 0.5 to identify them
        #maze[2, 2] = 0.5
        #maze[self.height - 3, self.width - 3] = 0.5

        if self.displayMaze:
            cv2.namedWindow('Maze', cv2.WINDOW_NORMAL)
            cv2.imshow('Maze', maze)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        maze = maze * 255.0
        cv2.imwrite(self.path, maze)

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
        #self.image = pygame.image.load("images/green-car.png") #38x76
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
        #self.visited = np.full(screen.get_size(), False)
        
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

    #@jit
    def Drive(self):
        draw_lines = True
        track_edge_distances = self.GetTrackEdgeDistances(draw_lines)
        if draw_lines is True:
            pygame.display.update()

        if self.crashed:
            return
        
        self.won = self.CheckIfPositionWins(self.position[0], self.position[1])
        if self.won:            
            self.DrawCarFinishLocation(GREEN)
            return
        
        self.position_previous_rounded = self.position_rounded

        if self.CheckIfDeadEnd(track_edge_distances) == True:
            self.direction_radians += math.pi
        else:
            # in future, do some neural network magic to decide how much to steer and how much to change speed
            # for now, get the change in speed, based on how clear the road is straight ahead
            distance_ahead = track_edge_distances[0][1] # how far is clear straight ahead
            distance_ahead -= track_edge_distances[0][2] # subtract the distance ahead that has already been visited
            speed_delta = 4 * (distance_ahead/CAR_VISION_DISTANCE) - 2

            if speed_delta < CAR_ACCELERATION_MIN:
                speed_delta = CAR_ACCELERATION_MIN
            
            if speed_delta > CAR_ACCELERATION_MAX:
                speed_delta = CAR_ACCELERATION_MAX

            speed_new = self.speed + speed_delta
            
            if speed_new < CAR_SPEED_MIN:
                speed_new = CAR_SPEED_MIN
            
            if speed_new > CAR_SPEED_MAX:
                speed_new = CAR_SPEED_MAX
            
            self.speed = speed_new

            steering_radians_previous = self.steering_radians
            steering_angle_new = 0
            max_track_distance = 1
            #if distance_ahead + CAR_VISITED_PATH_RADIUS < CAR_VISION_DISTANCE / CAR_WHEN_TO_STEER_FACTOR:        
            for ted in track_edge_distances:
                if ted[0][0] == 0:
                    continue
                distance_for_angle = ted[1]-(CAR_VISITED_PATH_AVOIDANCE_FACTOR*ted[2])
                steering_angle_new += distance_for_angle * ted[0][1]
                if distance_for_angle > max_track_distance: max_track_distance = distance_for_angle

            self.steering_radians =  steering_angle_new * CAR_STEERING_MULTIPLIER / max_track_distance

            # restrict how much the steering can be changed per frame
            if self.steering_radians < steering_radians_previous - CAR_STEERING_RADIANS_DELTA_MAX:
                self.steering_radians = steering_radians_previous - CAR_STEERING_RADIANS_DELTA_MAX
            elif self.steering_radians > steering_radians_previous + CAR_STEERING_RADIANS_DELTA_MAX:
                self.steering_radians = steering_radians_previous + CAR_STEERING_RADIANS_DELTA_MAX

            # restrict how much the steering can be per frame
            if self.steering_radians > CAR_STEERING_RADIANS_MAX:
                self.steering_radians = CAR_STEERING_RADIANS_MAX
            elif self.steering_radians < -CAR_STEERING_RADIANS_MAX:
                self.steering_radians = -CAR_STEERING_RADIANS_MAX

            self.direction_radians += self.steering_radians #* self.speed # direction changes more per frame if you're goig faster

            self.position = (self.position[0] + self.speed * math.cos(self.direction_radians), self.position[1] + self.speed * math.sin(self.direction_radians))
            self.position_rounded = (round(self.position[0]),round(self.position[1]))    

            car_speed_colour = round(255 * self.speed / CAR_SPEED_MAX)
            car_colour = (255 - car_speed_colour, car_speed_colour, 0)
            self.screen.set_at(self.position_rounded, car_colour)
            pygame.display.update(pygame.Rect(self.position_rounded[0],self.position_rounded[1],1,1))
            pygame.display.update()

            self.UpdateVisited()

            # decided that the car has "crashed" if it has taken more than this distance to complete the maze
            if self.statsInfoCar["distance"] > TRACK_MAX_DISTANCE_TO_COMPLETE:
                self.crashed = True
                self.DrawCarFinishLocation(RED)
                return

            self.statsInfoCar["frames"] += 1
            self.statsInfoCar["distance"] += speed_new
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
            
        self.carIconGroup.update(self.position_rounded[0], self.position_rounded[1], self.direction_radians)

        if self.statsInfoCar["frames"] % FRAMES_BETWEEN_BLURRING_VISITED == 0:
            self.BlurVisited()

    def UpdateVisited(self):
        # update a 2d subarray of visited that's +/- 10 from the current position
        #self.visited[self.position_rounded[0]-CAR_VISITED_PATH_RADIUS:self.position_rounded[0]+CAR_VISITED_PATH_RADIUS, self.position_rounded[1]-CAR_VISITED_PATH_RADIUS:self.position_rounded[1]+CAR_VISITED_PATH_RADIUS] = True
        pygame.draw.circle(self.visitedByCarScreen, CAR_VISITED_COLOUR, self.position_previous_rounded, CAR_VISITED_PATH_RADIUS)

    def BlurVisited(self):
        blurSize = 3.0
        originalSize = pygame.Surface.get_size(self.visitedByCarScreen)
        scaledSize = (originalSize[0]//blurSize,originalSize[1]//blurSize)
        scaledDown = pygame.Surface(scaledSize)
        pygame.transform.smoothscale(self.visitedByCarScreen,scaledSize,scaledDown)
        pygame.transform.scale(scaledDown, originalSize, self.visitedByCarScreen)
    
    def CheckIfPositionWins(self, x, y):
        if -2 < x/SQUARE_SIZE-COLS < -1 and -2 < y/SQUARE_SIZE-ROWS < -1:
            return True
        else:
            return False

    def CheckIfDeadEnd(self, track_edge_distances):
        # if any of the distances is more than the size of a square, then it's not a dead end
        for ted in track_edge_distances:
            if math.pi/-2.0<=ted[0][0]<=math.pi/2.0 and ted[1] > SQUARE_SIZE:
                return False

        return True
    
    def GetTrackEdgeDistances(self, draw_lines):    
        car_on_track = self.track.track_pixels[self.position_rounded]
        if not car_on_track:
            self.crashed = True
            self.DrawCarFinishLocation(RED)
            return NULL

        # list of tuples: [(angle,distance)]
        track_edge_distances = []

        for vision_angle_and_weight in CAR_VISION_ANGLES_AND_WEIGHTS:
            track_edge_distance, visited_count = self.GetTrackEdgeDistance(vision_angle_and_weight[0], self.visitedByCarScreen, draw_lines)
            track_edge_distances.append((vision_angle_and_weight, track_edge_distance, visited_count))
        
        return track_edge_distances
    
    def GetTrackEdgeDistance(self, vision_angle, visitedByCarScreen, draw_line):
        # from x,y follow a line at vision_angle until no longer on the track
        # or until CAR_VISION_DISTANCE has been reached
        search_angle_radians = self.direction_radians + vision_angle
        delta_x = math.cos(search_angle_radians)
        delta_y = math.sin(search_angle_radians)

        #test_x, test_y = self.position_rounded
        edge_distance = 0
        visited_count = 0

        for i in range(CAR_VISITED_PATH_RADIUS, CAR_VISION_DISTANCE):
            edge_distance = i
            test_x = self.position_rounded[0] + i * delta_x
            test_y = self.position_rounded[1] + i * delta_y
            if self.track.track_pixels[round(test_x)][round(test_y)] == False:
                break
            
            #if i>CAR_VISITED_PATH_RADIUS:
            visited_colour = pygame.Surface.get_at(visitedByCarScreen, (round(test_x),round(test_y)))
            if CAR_VISITED_COLOUR[0]/CAR_VISITED_FADE_LEVEL<=visited_colour[0]<CAR_VISITED_COLOUR[0]*CAR_VISITED_FADE_LEVEL:
                visited_count += 1 
    
        if draw_line:
            self.DrawLineToTrackEdge(test_x, test_y)

        return edge_distance, visited_count
    
    def DrawLineToTrackEdge(self, test_x, test_y):
        pygame.draw.line(self.trackDistancesScreen, WHITE, self.position_rounded, [round(test_x), round(test_y)])

    def DrawCarFinishLocation(self, highlight_colour):
        pygame.draw.circle(self.screen, highlight_colour, self.position_rounded, TRACK_MAX_WIDTH, width=2)
        finish_zone = pygame.Rect(self.position_rounded[0] - TRACK_MAX_WIDTH, self.position_rounded[1] - TRACK_MAX_WIDTH, 2 * TRACK_MAX_WIDTH, 2 * TRACK_MAX_WIDTH)
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
    
    def Create(self):
        maze = self.GetNewMaze()
        maze = self.SetMazeEndPoints(maze)
        #maze = self.GetScaledMaze(maze)
        self.SetScaledMazeSurface(maze)
        self.track_surface = self.SetTrackPixelsFromMazeSurface(self.track_surface)
        #self.GetNewTrack()
        #self.SetScaledTrack()
        #self.SetInterpolatedScaledTrack()
        #self.SetTrackWidths()
        #self.SetTrackPixels()
        #self.DrawInterpolatedTrack()
        #self.DrawTrack()

    def GetNewMaze(self):
        MAZE_HEIGHT = ROWS + 1
        MAZE_WIDTH = COLS + 1
        MAZE_SAVE_PATH = "maze-backtracking.png"
        MAZE_DISPLAY = False

        backtracking = Backtracking(MAZE_HEIGHT, MAZE_WIDTH, MAZE_SAVE_PATH, MAZE_DISPLAY)
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
        # get an array from the screen identifying where the track is
        #tp = pygame.surfarray.array_red(surf)
        # reduce this down to an array of booleans where 255 becomes True
        #self.track_pixels = tp.astype(dtype=bool)

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
    statsInfo = statsInfoCar.copy()
    statsInfo.update(statsInfoGlobal)
    for k,v in statsInfo.items():
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
    visitedByCarScreenCopy = visitedByCarScreen.copy()

    trackDistancesScreen = pygame.Surface(window_size, pygame.SRCALPHA, 32)
    trackDistancesScreen = trackDistancesScreen.convert_alpha()
    trackDistancesScreenCopy = trackDistancesScreen.copy()
    
    statsSurface = pygame.Surface(window_size)
    statsSurface = statsSurface.convert_alpha()
    statsSurface.fill((0, 0, 0, 0))
    
    # define a variable to control the main loop
    running = True
    paused = False
    newTrackAndCarNeeded = True
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
            track.Create()
            visitedByCarScreen = visitedByCarScreenCopy.copy() # clear this surface
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
            
            #if event.type == pygame.MOUSEBUTTONDOWN:
            #    # get the mouse position
            #    mouse_pos = pygame.mouse.get_pos()
            #    car.position = mouse_pos
            #    car.GetTrackEdgeDistances(True)
            #    continue

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
        
        if paused:
             continue

        if not car.crashed and not car.won:
            car.Drive()
            StatsUpdate(statsSurface, car.statsInfoCar, statsInfoGlobal)
        
        pygame.display.flip()
        screen.blit(background, (0,0))
        screen.blit(visitedByCarScreen, (0,0))
        screen.blit(car.trackDistancesScreen, (0,0))
        car.trackDistancesScreen = trackDistancesScreenCopy.copy() # clear this surface
        car.carIconGroup.draw(screen)
        screen.blit(statsSurface, (0,0))
        pygame.display.update()
        clock.tick(400)

        #if car.crashed or car.won:
        if car.won:
            newTrackAndCarNeeded = True
            pygame.time.wait(2000)

        if car.crashed:
            teds = car.GetTrackEdgeDistances(True)

# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__=="__main__":
    # call the main function
    main()
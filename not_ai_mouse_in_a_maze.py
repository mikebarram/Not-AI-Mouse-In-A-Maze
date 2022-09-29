# A program which uses recursive backtracking to generate a maze
# https://aryanab.medium.com/maze-generation-recursive-backtracking-5981bc5cc766
# https://github.com/AryanAb/MazeGenerator/blob/master/backtracking.py
# A good alternative might be https://github.com/AryanAb/MazeGenerator/blob/master/hunt_and_kill.py

import math
import random
import sys
# import the pygame module, so you can use it
from asyncio.windows_events import NULL
from collections import deque
from curses import KEY_LEFT
from enum import Enum

import cv2
import numpy as np
import pygame
from scipy.interpolate import splev, splprep
from scipy.ndimage import uniform_filter1d

#from functools import cache


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

WINDOW_WIDTH = COLS * SQUARE_SIZE
WINDOW_HEIGHT = ROWS * SQUARE_SIZE

# cars can only look so far ahead. Needs to be somewhat larger than the maximum track width - try seting that distance to the size of a grid square
CAR_VISION_DISTANCE = round(3.0 * SQUARE_SIZE)
#CAR_VISION_ANGLES = (0, -10, 10, -20, 20, -30, 30, -45, 45, -60, 60, -80, 80, -90, 90) # 0 must be first
CAR_VISION_ANGLES = (0, -20, 20, -45, 45, -60, 60, -90, 90) # 0 must be first
CAR_SPEED_MIN_INITIAL = 2 # pixels per frame
CAR_SPEED_MAX_INITIAL = 7 # pixels per frame
CAR_SPEED_MIN = CAR_SPEED_MIN_INITIAL # pixels per frame
CAR_SPEED_MAX = CAR_SPEED_MAX_INITIAL # pixels per frame
CAR_ACCELERATION_MIN = -3 # change in speed in pixels per frame
CAR_ACCELERATION_MAX = 2 # change in speed in pixels per frame
CAR_STEERING_RADIANS_MAX = math.radians(45)
CAR_STEERING_RADIANS_DELTA_MAX = math.radians(45)
CAR_PATH_COLOUR = RED
CAR_COLOUR = GREEN

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
        self.image = pygame.image.load("images/green-car.png") #38x76
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
    def __init__(self, screen, track):
        self.screen = screen
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
        #self.position = (self.track.scaled_track[0][0],self.track.scaled_track[0][1]) #this should be a tuple of integers. interpolated_scaled_track[0] will be float
        self.position = (SQUARE_SIZE*1.5, SQUARE_SIZE*1.5) # middle of top left square inside the border
        self.position_rounded = (round(self.position[0]),round(self.position[1]))
        self.speed = SQUARE_SIZE/50  # pixels per frame
        #self.direction_radians = track.GetInitialDirectionRadians()
        self.direction_radians = math.pi/3
        self.steering_radians = 0
        self.crashed = False
        self.position_previous_rounded = self.position_rounded
        
        self.statsInfo = {
            "distance":0.0,
            "frames":0,
            "average speed":0.0,
            "rotations":0.0,
            "CAR_SPEED_MIN":CAR_SPEED_MIN,
            "CAR_SPEED_MAX":CAR_SPEED_MAX
            }

        self.instructions = {
            "speed":0.0,
            "speed_delta":0.0,
            "direction_radians":0.0,
            "steering_radians":0.0,
            }
        self.latestInstructions = deque(maxlen=20)
        
        self.carIcon = CarIcon(self.position_rounded[0],self.position_rounded[1])
        self.carIconGroup = pygame.sprite.Group()
        self.carIconGroup.add(self.carIcon)

    def Drive(self):
        track_edge_distances = self.GetTrackEdgeDistances(False)
        if self.crashed:
            return
        
        self.position_previous_rounded = self.position_rounded

        # in future, do some neural network magic to decide how much to steer and how much to change speed
        # for now, get the change in speed, based on how clear the road is straight ahead
        distance_ahead = track_edge_distances[0][1] # how far is clear straight ahead
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
        if distance_ahead < CAR_VISION_DISTANCE / 1.8:        
            for ted in track_edge_distances:
                if ted[0] == 0:
                    continue
                steering_angle_new += ted[1] / ted[0]
                if ted[1] > max_track_distance: max_track_distance = ted[1]

        self.steering_radians = 5 * steering_angle_new / max_track_distance

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
        
        #self.carIconGroup.draw(self.screen)
        self.carIconGroup.update(self.position_rounded[0], self.position_rounded[1], self.direction_radians)
        
        self.statsInfo["frames"] += 1
        self.statsInfo["distance"] += speed_new
        self.statsInfo["average speed"] = self.statsInfo["distance"] // self.statsInfo["frames"]
        self.statsInfo["rotations"] += self.steering_radians / (2 * math.pi)
        self.statsInfo["CAR_SPEED_MIN"] = CAR_SPEED_MIN
        self.statsInfo["CAR_SPEED_MAX"] = CAR_SPEED_MAX

        self.instructions = {
            "speed":self.speed,
            "speed_delta":speed_delta,
            "direction_radians":self.direction_radians,
            "steering_radians":self.steering_radians,
            "track_edge_distances":track_edge_distances
            }
        self.latestInstructions.appendleft(self.instructions)

    def GetTrackEdgeDistances(self, draw_lines):    
        car_on_track = self.track.track_pixels[self.position_rounded]
        if not car_on_track:
            self.crashed = True
            self.DrawCrashedCar()
            return NULL

        # list of tuples: [(angle,distance)]
        track_edge_distances = []

        for vision_angle in CAR_VISION_ANGLES:
            track_edge_distance = self.GetTrackEdgeDistance(vision_angle, draw_lines)
            track_edge_distances.append((vision_angle, track_edge_distance))

        #if draw_lines is True:
        pygame.display.update()
        
        self.crashed = False
        return track_edge_distances
    
    def GetTrackEdgeDistance(self, vision_angle, draw_line):
        # from x,y follow a line at vision_angle until no longer on the track
        # or until CAR_VISION_DISTANCE has been reached
        search_angle_radians = self.direction_radians + math.radians(vision_angle)
        delta_x = math.cos(search_angle_radians)
        delta_y = math.sin(search_angle_radians)

        test_x, test_y = self.position_rounded
        edge_distance = 0

        for i in range(1, CAR_VISION_DISTANCE):
            edge_distance = i
            test_x += delta_x
            test_y += delta_y
            if self.track.track_pixels[round(test_x)][round(test_y)] == False:
                break
        
        if draw_line:
            pygame.draw.line(self.screen, RED, self.position_rounded, [round(test_x), round(test_y)])

        return edge_distance
    
    def DrawCrashedCar(self):
        pygame.draw.circle(self.screen, RED, self.position_rounded, TRACK_MAX_WIDTH, width=2)
        crash_zone = pygame.Rect(self.position_rounded[0] - TRACK_MAX_WIDTH, self.position_rounded[1] - TRACK_MAX_WIDTH, 2 * TRACK_MAX_WIDTH, 2 * TRACK_MAX_WIDTH)
        pygame.display.update(crash_zone)

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

    #def GetScaledMaze(maze):    
    #    return [[t[0] * SQUARE_SIZE,t[1] * SQUARE_SIZE] for t in maze]

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

    def GetNewTrack(self):
        # python version of the answer to this 
        # https://answers.unity.com/questions/394991/generate-closed-random-path-in-c.html
        
        start_row = random.randint(1, COLS-2)
        start_col = random.randint(1, ROWS-2)
        
        # create a list of the coordinates of 4 central points that make up a square
        track = [(start_row,start_col),(start_row+1,start_col),(start_row+1,start_col+1),(start_row,start_col+1)]
        
        # create 2D array of booleans for the vertices in the grid
        # Elements will be set to True to indicate the vertex is part of the track and so the track can't be expanded into it
        # Elements around the outside are also set to True, so that the track won't expand all the way to the edge
        # https://stackoverflow.com/questions/13614452/python-multidimensional-boolean-array
        grid_vertices = np.zeros((COLS+1,ROWS+1), dtype=bool)   

        # set the first and last rows
        for i in range(COLS):
            grid_vertices[i][0] = True        
            grid_vertices[i][ROWS] = True
    
        # set the first and last columns
        for i in range(1, ROWS-1):
            grid_vertices[0][i] = True        
            grid_vertices[COLS][i] = True

        # set the grid vertices for the starting square of track
        for track_vertex in track:
            grid_vertices[track_vertex] = True

        # keep expanding the track until it won't expand any more
        while True:
            # get a random track segment to expand but, if that won't expand, need to get another until all have been tested.
            # so, get a randomly sorted list of indices and work through the segments indexed by them until a segment can be expanded or none can be expanded and end the loop
            track_len = len(track)
            shuffled_indices = list(range(0,track_len))
            random.shuffle(shuffled_indices)

            # loop though the shuffled_indices until a segment can be expanded or all have been tried
            for track_segment_start_index in shuffled_indices:
                track_segment_end_index = track_segment_start_index + 1
                if track_segment_end_index > track_len-1:
                    track_segment_end_index = 0

                track_segment_start = track[track_segment_start_index]
                track_segment_end = track[track_segment_end_index]

                track_segment_start_x, track_segment_start_y = track_segment_start
                track_segment_end_x, track_segment_end_y = track_segment_end
                
                # the track started off going round clockwise and so always will
                # the track should expand outwards, which is always to the left of the current track segment (when looking from the start of the segment to the end of it)
                # coordinates start (0,0) at the top left
                # if the segment goes left to right, it will expand upwards and y will decrease by 1
                # if the segment goes up to down, it will expand to the right and x will increase by 1
                # if the segment goes right to left, it will expand downwards and y will increase by 1
                # if the segment goes down to up, it will expand to the left and x wil decrease by 1

                delta_x = track_segment_end_y - track_segment_start_y
                delta_y = track_segment_start_x - track_segment_end_x
                
                track_vertex_extra_1 = (track_segment_start_x + delta_x, track_segment_start_y + delta_y)
                # check if this is already in use
                if grid_vertices[track_vertex_extra_1]:
                    # in use, so can't expand this segment
                    continue

                track_vertex_extra_2 = (track_segment_end_x + delta_x, track_segment_end_y + delta_y)
                # check if this is already in use
                if grid_vertices[track_vertex_extra_2]:
                    # in use, so can't expand this segment
                    continue

                # neither new vertex has been used by the track, so insert the new vertices after the start one
                # this method inserts a list (the 2 new vertices) into the track before the vertex for the end of the segment
                track[track_segment_end_index:track_segment_end_index] = [track_vertex_extra_1,track_vertex_extra_2]

                # flag that the new vertices are part of the track
                grid_vertices[track_vertex_extra_1] = True
                grid_vertices[track_vertex_extra_2] = True                
                # if the track has been expanded at this segment, break from this loop
                break
            else:
                # gone through all of the track segments and the track wasn't expanded, so stop trying to expand it any more
                break
        
        self.track = track

    def SetScaledTrack(self):    
        self.scaled_track = [[t[0] * SQUARE_SIZE,t[1] * SQUARE_SIZE] for t in self.track]

    def DrawTrackStart(self):
        pygame.draw.circle(self.track_surface, START_COLOUR, self.scaled_track[0], TRACK_MAX_WIDTH + 5)

    def DrawTrack(self):
        pygame.draw.lines(self.track_surface, BLACK, True, self.scaled_track, 1)

    def SetInterpolatedScaledTrack(self):
        # https://stackoverflow.com/questions/31464345/fitting-a-closed-curve-to-a-set-of-points
        
        # loop the track back round to the strart
        interpolated_scaled_track = self.scaled_track + [self.scaled_track[0]]

        # change the list of coordinates to a numpy array
        pts = np.array(interpolated_scaled_track)

        # magic happens
        tck, u = splprep(pts.T, u=None, s=0.0, per=1) 
        u_new = np.linspace(u.min(), u.max(), TRACK_CURVE_POINTS)
        x_new, y_new = splev(u_new, tck, der=0)

        self.interpolated_scaled_track = [list(a) for a in zip(x_new , y_new)]

    def SetTrackWidths(self):
        # track width should vary between TRACK_MIN_WIDTH to TRACK_MAX_WIDTH
        # to get the width to vary smoothly but randomly, a random walk is created and then smoothed over 20 points, normalised to the range [0,1] and scaled
        # the width starts at TRACK_MIN_WIDTH. If the end of the track is wide, it can cause problems with driving when it loops round to the narrow start.
        # The list of widths could be halved in length and mirrored, so that the end is the same width as the start, but this makes for an extra challenge.
        x = np.linspace(0, TRACK_CURVE_POINTS, TRACK_CURVE_POINTS)

        def RandomWalk(x):
            y = 0
            result = []
            for _ in x:
                result.append(y)
                y += np.random.normal(scale=1)
            return np.array(result)

        def NormalizeData(data):
            return (data - np.min(data)) / (np.max(data) - np.min(data))

        track_width = uniform_filter1d(RandomWalk(x), size=20)
        track_width = (TRACK_MAX_WIDTH - TRACK_MIN_WIDTH) * NormalizeData(track_width) + TRACK_MIN_WIDTH
        self.track_widths = track_width
    
    def SetTrackPixels(self):
        # draw the track in lots of red circles on a black surface
        surf = pygame.Surface(self.window)
        surf.fill(BLACK)
        for i in range(0, len(self.interpolated_scaled_track)):
            pygame.draw.circle(surf, RED, self.interpolated_scaled_track[i], self.track_widths[i])
        
        # get an array from the screen identifying where the track is
        tp = pygame.surfarray.array_red(surf)
        # reduce this down to an array of booleans where 255 becomes True
        self.track_pixels = tp.astype(dtype=bool)

    def DrawInterpolatedTrack(self):
        # now make the track look nice
        self.track_surface.fill(WHITE)

        self.DrawTrackStart()

        # draw lots of black circles of varying widths to create a smooth track that matches the backing array track_pixels
        for i in range(0, len(self.interpolated_scaled_track)):
            pygame.draw.circle(self.track_surface, BLACK, self.interpolated_scaled_track[i], self.track_widths[i])
            
        # draw track centre line
        #for t in self.scaled_track:
        #    self.screen.set_at((round(t[0]),round(t[1])), TRACK_MIDLINE_COLOUR)

    def GetInitialDirectionRadians(self):
        initial_angle = math.atan2(self.track[1][1]-self.track[0][1], self.track[1][0]-self.track[0][0])
        return initial_angle

def StatsUpdate(statsSurface, statsInfo):
    statsSurface.fill((0, 0, 0, 0))
    font = pygame.font.SysFont('Arial', 12, bold=False)
    textTop = 0
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
    # load and set the logo
    logo = pygame.image.load("logo32x32.png")
    pygame.display.set_icon(logo)
    pygame.display.set_caption("AI car")
    
    window_size = (WINDOW_WIDTH,WINDOW_HEIGHT)
     
    # create a surface on screen that has the size defined globally
    screen = pygame.display.set_mode(window_size)
    background = pygame.Surface(window_size)
    statsSurface = pygame.Surface(window_size)
    statsSurface = statsSurface.convert_alpha()
    statsSurface.fill((0, 0, 0, 0))
    
    # define a variable to control the main loop
    running = True
    paused = False
    newTrackAndCarNeeded = True

    # main loop
    while running:
        if newTrackAndCarNeeded:
            # create the track and draw it on the background
            CAR_SPEED_MIN = CAR_SPEED_MIN_INITIAL # pixels per frame
            CAR_SPEED_MAX = CAR_SPEED_MAX_INITIAL # pixels per frame
            track = Track(window_size, background)
            track.Create()
            car = Car(background, track)            
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

        if not car.crashed:
            car.Drive()
            StatsUpdate(statsSurface, car.statsInfo)   
        
        screen.blit(background, (0,0))
        car.carIconGroup.draw(screen)
        screen.blit(statsSurface, (0,0))
        pygame.display.flip()
        clock.tick(200)


# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__=="__main__":
    # call the main function
    main()

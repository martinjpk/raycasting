import pyglet
# from pyglet.gl import *
import numpy as np
from math import sin, cos, atan, atan2, acos, sqrt, pi
import logging
import matplotlib.pyplot as plt
import multiprocessing

class Object:
    def __init__(self, pos: np.array):
        self.pos = pos

    def update(self):
        pass

    def intersect(self):
        pass

    def move(self, dir):
        self.pos += dir

    def scale(self):
        pass 

    def rotate(self):
        pass

class Sphere(Object):
    def __init__(self, center, radius, color='gray'):
        super().__init__(np.array(center))
        self.rad = radius
        self.color = color

    def intersect(self, ray):
        oc = ray.org - self.pos
        loc = ray.dir.dot(oc)
        # print('ray dir: {}'.format(ray.dir))
        # print('oc: {}\tloc: {}'.format(oc, loc))
        # print('sqrt({})'.format(loc**2 - np.linalg.norm(oc)**2 + self.rad**2))
        return - loc - (loc**2 - np.linalg.norm(oc)**2 + self.rad**2)**(1/2)

class Polygon(Object):
    def __init__(self, vert, color='red'):
        super().__init__(np.array(vert[0]))
        self.vert = vert
        self.color = color
    
    def intersect(self, ray):
        pass

class Ray:
    def __init__(self, origin, ah, av):
        self.org = origin
        self.ah = ah
        self.av = av
        # print('ray -> ah: {:.4f}\tav: {:.4f}'.format(self.ah, self.av))

    def computeRay(self):
        self.dir = np.array([
            sin(self.av)*cos(self.ah),
            sin(self.av)*sin(self.ah),
            cos(self.av)
            ])

class Vision:
    def __init__(self, dir, fov=((6/4)*pi/4, pi/4), dist=10, size=(40,40)):
        self.fov_h, self.fov_v = fov
        self.dir = np.array(dir)
        r = np.linalg.norm(self.dir)
        self.ang_v = acos(self.dir[2]/r)
        self.ang_h = atan2(self.dir[1], self.dir[2])
        self.dist = dist
        self.m, self.n = size
        self.pos = np.array([0,0,2])
        # self.data = []
        self.data = np.ndarray((self.n, self.m))

    def cast(self, objects):
        # print('casting dir: {:,.4f}, {:,.4f}'.format(self.ang_h, self.ang_v))
        for i in range(self.n):
            for j in range(self.m):
                # print('\n===== i: {}\tj: {} ====='.format(i,j))
                ray = Ray(self.pos,
                    self.ang_h - self.fov_h + j*self.fov_h*2/self.m, 
                    self.ang_v - self.fov_v + i*self.fov_v*2/self.n
                    )
                ray.computeRay()
                minDist = np.inf
                for obj in objects:
                    dist = obj.intersect(ray)
                    if dist < minDist:
                        minDist = dist
                        cls_obj = obj
                minDist = 0 if minDist == np.inf else minDist
                # self.data.append(int(255*minDist/10))
                self.data[i][j] = minDist  

    def __cast_parallel(self, pos, angle, objects):
        p = multiprocessing.Process(target=self.__cast_ray, args=())

    def __cast_ray(self, pos, angle, objects):
        ray = Ray(pos, *angle)
        ray.computeRay()
        minDist = np.inf
        for obj in objects: 
            dist = obj.intersect(ray)
            if dist < minDist:
                minDist = dist
                cls_obj = obj
        minDist = 0 if minDist == np.inf else minDist
        # self.data.append(int(255*minDist/10))
        self.data[i][j] = minDist  

    def render(self):
        # raw = (GLubyte * len(self.data))(*self.data)
        # img = pyglet.image.ImageData(self.m, self.n, 'L', raw)
        # img.blit(0,0)
        plt.imshow(self.data, cmap='gray_r', vmin=0, vmax=6)
        plt.show()
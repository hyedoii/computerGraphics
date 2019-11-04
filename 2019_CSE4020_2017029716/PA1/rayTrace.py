#!/usr/bin/env python3
# -*- coding: utf-8 -*
# sample_python aims to allow seamless integration with lua.
# see examples below

import os
import sys
import pdb  # use pdb.set_trace() for debugging
import code  # or use code.interact(local=dict(globals(), **locals()))  for debugging.
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

def swap(one, onother):
    temp = one
    one = onother
    onother = temp


class Color:
    def __init__(self, R, G, B):
        self.color = np.array([R, G, B]).astype(np.float)
    
    # Gamma corrects this color.
    # @param gamma the gamma value to use (2.2 is generally used).
    def gammaCorrect(self, gamma):
        inverseGamma = 1.0 / gamma
        self.color = np.power(self.color, inverseGamma)
    
    def toUINT8(self):
        return (np.clip(self.color, 0, 1) * 255).astype(np.uint8)


class Sphere:
    def __init__(self, stype, center, radius, ref):
        self.stype = stype
        self.center = center
        self.radius = radius
        self.ref = ref
    
    def intersects(self, camera):
        sphere_to_ray = camera.viewPoint - self.center
        b = 2 * camera.direction * sphere_to_ray
        c = sphere_to_ray ** 2 - self.radius ** 2
        discriminant = b ** 2 - 4 * c
        
        if discriminant >= 0:
            dist = (-b - np.sqrt(discriminant)) / 2
            if dist > 0:
                return dist
    # dist = 거리


# normal vector of intersect surface
# pt (접점의 point ) sphere랑 box 둘 다 해당
# pt - ray.center == ray.direction
def surface_norm(self, pt):
    return (pt - self.center).normalize()


class Box:
    def __init__(self, stype, minPt, maxPt, ref):
        self.stype = stype
        self.minPt = minPt
        self.maxPt = maxPt
        self.center = minPt.center(maxPt)
        self.ref = ref
    
    # p is vector of box's center, d is ray's direction
    # e is inner product of normal vector and p
    # f is inner product of normal vector and d
    # h is distance of center to side
    # hx = (maxPt.x - minPt.x) / 2, hy = (maxPt.y - minPt.y) / 2, hz = (maxPt.z - minPt.z) / 2
    # if e == h not intersects
    def intersects(self, ray):
        tMin = -10000
        tMax = 10000
        p = self.center - ray.center
        d = ray.direction
        axis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        h = [(self.maxPt.x - self.minPt.x) / 2, (self.maxPt.y - self.minPt.y) / 2, (self.maxPt.z - self.minPt.z) / 2]
        
        for i in range(2):
            e = axis[i] * p
            f = axis[i] * d
            if abs(f) > 0:
                first = (e + h[i]) / f
                second = (e - h[i]) / f
                if first > second:
                    swap(first, second)
                if first > tMin:
                    tMin = first
                if second < tMax:
                    tMax = second
                if tMin > tMax:
                    return
                if tMax < 0:
                    return
            elif -e - h[i] > 0 or -e + h[i] < 0:
                return
        if tMin > 0:
            return tMin
        else:
            return tMax


class Shader:
    def __init__(self, name, typee, color):
        self.color = color
        self.name = name
        self.typee = typee
    
    #무광
    def lam(self, ray, n):
        l = ray.direction
        L = self.color * ray.intensity * max(0, n * l)
        return L
    
    #유광
    def pho(self, ray, n, eye):
        h = n + eye / (n + eye).norm()    #단위벡터
        L = self.color * ray.intensity * pow((max(0, n * h)), 50)
        return L

# L = 최종 색깔
# n = 법선 벡터 ( 박스일 경우 x,y,z 법선 중 하나)


class Vector:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    
    # find size of vector
    def norm(self):
        return np.sqrt(sum(num * num for num in self))
    
    # find unit vector
    def normalize(self):
        return self / self.norm()
    
    def reflect(self, other):
        other = other.normalize()
        return self - 2 * (self * other) * other
    
    # find box's center
    def center(self, other):
        return Vector((self.x + other.x) / 2, (self.y + other.y) / 2, (self.z + other.z) / 2)
    
    # operator overide
    def __str__(self):
        return "Vector({},{},{})".format(*self)
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, other):
        if isinstance(other, Vector):
            return self.x * other.x + self.y * other.y + self.z * other.z
        else:
            return Vector(self.x * other, self.y * other, self.z * other)

def __rmul__(self, other):
    return self.__mul__(other)
    
    def __truediv__(self, other):
        return Vector(self.x / other, self.y / other, self.z / other)
    
    def __pow__(self, exp):
        if exp != 2:
            raise ValueError("Exponent can only be two")
        else:
            return self * self

def __iter__(self):
    yield self.x
        yield self.y
        yield self.z


class Ray:
    def __init__(self, center, intensity):
        self.center = center
        #self.direction = direction.normalize()
        self.intensity = intensity
    
    def point_at_dist(self, dist):
        return self.center + self.direction * dist


class Camera:
    def __init__(self, viewPoint, viewDir, viewUp, viewWidth, viewHeight):
        self.center = viewPoint
        self.viewDir = viewDir
        self.viewUp = viewUp
        self.viewWidth = viewWidth
        self.viewHeight = viewHeight
    
    def camera_to_surface(self, i, j, imgSize):
        ratio = 1 / np.linalg.norm(self.viewDir)
        
        screenHor = np.cross(-self.viewDir, self.viewUp)
        screenVer = np.cross(screenHor, -self.viewDir)
        screenHorRatio = (self.viewWidth / 2) / np.linalg.norm(screenHor)
        screenVerRatio = (self.viewHeight / 2) / np.linalg.norm(screenVer)
        
        screenHor = screenHorRatio * screenHor
        screenVer = screenVerRatio * screenVer
        
        staring_point = -self.viewDir * ratio * self.viewDir + screenHor + screenVer
        
        result = staring_point + (((-screenVer) / (imgSize[1] / 2)) * i) + (((-screenHor) / (imgSize[0] / 2)) * j)
        
        vec = Vector(result[0], result[1], result[2])
        
        return vec

"""
    class Light:
    def __init__(self, position, intensity):
    self.position = position
    self.intensity = intensity
    """


Point = Vector


def main():
    tree = ET.parse(sys.argv[1])
    root = tree.getroot()
    
    shaders = []
    objects = []
    lights = []
    
    # set default values
    viewDir = np.array([0, 0, -1]).astype(np.float)
    viewUp = np.array([0, 1, 0]).astype(np.float)
    viewProjNormal = -1 * viewDir  # you can safely assume this. (no examples will use shifted perspective camera)
    viewWidth = 1.0
    viewHeight = 1.0
    projDistance = 1.0
    intensity = np.array([1, 1, 1]).astype(np.float)  # how bright the light is.
    # print(np.cross(viewDir, viewUp))
    
    # <image> </image>
    imgSize = np.array(root.findtext('image').split()).astype(np.int)
    
    # <camera> </camera>
    for c in root.findall('camera'):
        viewPoint = np.array(c.findtext('viewPoint').split()).astype(np.float)
        viewDir = np.array(c.findtext('viewDir').split()).astype(np.float)
        # projNormal = np.array(c.findtext('projNormal').split()).astype(np.float)
        viewUp = np.array(c.findtext('viewUp').split()).astype(np.int)
        viewWidth = np.array(c.findtext('viewWidth')).astype(np.float)
        viewHeight = np.array(c.findtext('viewHeight')).astype(np.float)
        camera = Camera(viewPoint, viewDir, viewUp, viewWidth, viewHeight)
    
    # <shader> </shader>
    for c in root.findall('shader'):
        diffuseColor_c = np.array(c.findtext('diffuseColor').split()).astype(np.float)
        name = c.get('name')
        shaderType = c.get('type')
        shaders.append(Shader(name, shaderType, Color(diffuseColor_c[0], diffuseColor_c[1], diffuseColor_c[2])))
    # print('name', c.get('name'))
    # print('diffuseColor', diffuseColor_c)
    # print(shaders[0].name)
    # code.interact(local=dict(globals(), **locals()))

    # <light> </light>
    for c in root.findall('light'):
        lPosition = np.array(c.findtext('position').split()).astype(np.float)
        lIntensity = np.array(c.findtext('intensity').split()).astype(np.float)
        # print('position', lPosition)
        # print('intensity', lIntensity)
        
        lights.append(Ray(lPosition, lIntensity))

# <surface> </surface>
for c in root.findall('surface'):
    # Sphere
    if c.get('type') == 'Sphere':
        sType = c.get('type')
        sCenter = np.array(c.findtext('center').split()).astype(np.float)
        for j in c.findall('shader'):
            sShader = j.get('ref')
            #sShader = c.get('ref')
            sRadius = c.findtext('radius')
            choosedShader = Shader
            for i in range(len(shaders)):
                # print(shaders[i].name)
                if shaders[i].name == sShader:
                    choosedShader = shaders[i]
                    objects.append(Sphere(sType, sCenter, sRadius, choosedShader))
        # Box
        if c.get('type') == 'Box':
            bType = c.get('type')
            bMinPt = np.array(c.findtext('minPt').split()).astype(np.float)
            bShader = c.get('ref')
            bMaxPt = np.array(root.find('maxPt').split()).astype(np.float)
            choosedShader = Shader
                for i in range(len(shaders)):
                    if shaders[i].name == bShader:
                        choosedShader = shaders[i]
                    objects.append(Box(bType, bMinPt, bMaxPt, choosedShader))

        # Create an empty image
        channels = 3
            img = np.zeros((imgSize[1], imgSize[0], channels), dtype=np.uint8)
            img[:, :] = 0
            
            #####################################################################################################
            somelight = Ray
            
            for i in np.arange(imgSize[1]):
                for j in np.arange(imgSize[0]):
                    # 카메라가 네모칸 하나에 써는 그 벡터[i][j]= 물체를 향한 벡터
                    # intersects를 써서 거리를 리턴받는다.
                    
                    ratio = 1 / np.linalg.norm(viewDir)
                    
                    screenHor = np.cross(-viewDir, viewUp)
                    screenVer = np.cross(screenHor, -viewDir)
                    screenHorRatio = (viewWidth / 2) / np.linalg.norm(screenHor)
                    screenVerRatio = (viewHeight / 2) / np.linalg.norm(screenVer)
                    
                    screenHor = screenHorRatio * screenHor
                    screenVer = screenVerRatio * screenVer
                    
                    staring_point = -viewDir * ratio * viewDir + screenHor + screenVer
                    
                    result = staring_point + (((-screenVer) / (imgSize[1] / 2)) * i) + (((-screenHor) / (imgSize[0] / 2)) * j)
                    for h in range(i*j):
                        objects[h].intersects(result)
                    
                    
                    
                    """
                        camera.center = viewPoint
                        camera.direction = camera.camera_to_surface(i, j, imgSize).normalize()
                        distance = 0
                        pt = Vector(0, 0, 0)
                        
                        for a in np.arange(len(objects)):
                        distance = objects[a].intersects(camera)
                        pt = camera.point_at_dist(distance)
                        light = lights[0]
                        light.direction = pt - light.center
                        if(objects[a].stype == 'Sphere'):
                        if(objects[a].ref.typee == 'Lambertian'):
                        img[i][j] = objects[a].ref.lam(light, objects.surface_norm(light.point_at_dist())).toUnit8()
                        print('000000',somelight.center)
                        print(objects[a].ref)
                        if(objects[a].ref.typee == 'Phong'):
                        print('phooooooong')
                        img[i][j] = objects[a].ref.pho(light, objects[a].surface_norm(light.point_at_dist()), camera.direction).toUnit8()
                        print(light.center)
                        print(objects[a].ref)
                        """

# replace the code block below!
rawimg = Image.fromarray(img, 'RGB')

rawimg.save('out.png')
rawimg.save(sys.argv[1] + '.png')


if __name__ == "__main__":
    main()

import glfw
import sys
import pdb
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import ArrayDatatype
import time
import numpy as np
import ctypes
from PIL.Image import open
import OBJ
from Ray import *

# global variables
wld2cam = []
cam2wld = []

cow2wld = None
cursorOnCowBoundingBox = False
#pickInfo = None
floorTexID = 0
cameras = [
    [28, 18, 28, 0, 2, 0, 0, 1, 0],
    [28, 18, -28, 0, 2, 0, 0, 1, 0],
    [-28, 18, 28, 0, 2, 0, 0, 1, 0],
    [-12, 12, 0, 0, 2, 0, 0, 1, 0],
    [0, 100, 0, 0, 0, 0, 1, 0, 0]
]

camModel = None
cowModel = None
H_DRAG = 1
V_DRAG = 2

# dragging state
isDrag = 0

DrawLine = 0
animstartTime = 0

IsNotFirst = 0
Initial = 0

tempCow = []
tempDrawCow = []

before_h = None
before_v = None

before_x = 0
before_y = 0

end = None

class PickInfo:
    def __init__(self, cursorRayT, cowPickPosition, cowPickConfiguration, cowPickPositionLocal):
        self.cursorRayT = cursorRayT
        self.cowPickPosition = cowPickPosition.copy()
        self.cowPickConfiguration = cowPickConfiguration.copy()
        self.cowPickPositionLocal = cowPickPositionLocal.copy()


pickInfo = PickInfo(0, np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]))


def vector3(x, y, z):
    return np.array((x, y, z))


def position3(v):
    # divide by w
    w = v[3]
    return vector3(v[0] / w, v[1] / w, v[2] / w)


def vector4(x, y, z):
    return np.array((x, y, z, 1))


#################################
def angle(v1, v2):
    global argument1, argument2
    argument1 = np.sqrt(np.dot(v1, v1))
    argument2 = np.sqrt(np.dot(v2, v2))
    if argument1 == 0:
        argument1 = 0.0000001
    if argument2 == 0:
        argument2 = 0.0000001
    return np.arccos(np.dot(v1, v2) / (argument1 * argument2))


def distance(f, t):
    x = f[0] - t[0]
    x = x ** 2
    y = f[1] - t[1]
    y = y ** 2
    z = f[2] - t[2]
    z = z ** 2
    return np.sqrt(x + y + z)


def CMS(p0, p1, p2, p3, per):
    t0 = 0
    t1 = np.sqrt(distance(p0, p1)) + t0
    t2 = np.sqrt(distance(p1, p2)) + t1
    t3 = np.sqrt(distance(p2, p3)) + t2

    t = float(t2 * per + t1 * (1 - per))

    if t3 == t2:
        t3 = t2 + 0.0000001
    A3 = p2 * ((t3 - t) / (t3 - t2)) + p3 * ((t - t2) / (t3 - t2))

    if t2 == t1:
        t2 = t1 + 0.0000001
    A2 = p1 * ((t2 - t) / (t2 - t1)) + p2 * ((t - t1) / (t2 - t1))

    if t1 == t0:
        t1 = t0 + 0.0000001
    A1 = p0 * ((t1 - t) / (t1 - t0)) + p1 * ((t - t0) / (t1 - t0))

    if t3 == t1:
        t3 = t1 + 0.0000001
    B2 = A2 * ((t3 - t) / (t3 - t1)) + A3 * ((t - t1) / (t3 - t1))

    if t2 == t0:
        t2 = t0 + 0.0000001
    B1 = A1 * ((t2 - t) / (t2 - t0)) + A2 * ((t - t0) / (t2 - t0))

    if t2 == t1:
        t2 = t1 + 0.0000001
    C = B1 * ((t2 - t) / (t2 - t1)) + B2 * ((t - t1) / (t2 - t1))

    return C


def points_to_spline(tempMatrix4):
    vecSaver = []
    for i in range(0, len(tempMatrix4)):
        vecSaver.append(vector3(tempMatrix4[i][0][3], tempMatrix4[i][1][3], tempMatrix4[i][2][3]))
    # print(vecSaver)
    point_saver = []
    for i in range(0, len(vecSaver)):
        u0 = i - 1
        u1 = i
        u2 = i + 1
        u3 = i + 2

        u0 = (u0 + len(vecSaver)) % len(vecSaver)
        u2 = u2 % len(vecSaver)
        u3 = u3 % len(vecSaver)

        for j in range(0, 100):
            per = float(j) / float(100)
            point_saver.append(CMS(vecSaver[u0], vecSaver[u1], vecSaver[u2], vecSaver[u3], per))

    return_point = []
    for k in range(0, 3):
        for i in range(0, len(point_saver)):
            return_point.append(point_saver[i])
    return return_point


def mult(mat, v):
    w = mat[3][0] * v[0] + mat[3][1] * v[1] + mat[3][2] * v[2] + mat[3][3]
    x = mat[0][0] * v[0] + mat[0][1] * v[1] + mat[0][2] * v[2] + mat[0][3]
    y = mat[1][0] * v[0] + mat[1][1] * v[1] + mat[1][2] * v[2] + mat[1][3]
    z = mat[2][0] * v[0] + mat[2][1] * v[1] + mat[2][2] * v[2] + mat[2][3]
    if w != 1.0:
        x = x / w
        y = y / w
        z = z / w
    return np.array([x, y, z])


def rotate(m, v):
    return m[0:3, 0:3] @ v


def transform(m, v):
    return position3(m @ np.append(v, 1))


def getTranslation(m):
    return m[0:3, 3]


def setTranslation(m, v):
    m[0:3, 3] = v


def makePlane(a, b, n):
    v = a.copy()
    for i in range(3):
        if n[i] == 1.0:
            v[i] = b[i]
        elif n[i] == -1.0:
            v[i] = a[i]
        else:
            assert (n[i] == 0.0)

    return Plane(rotate(cow2wld, n), transform(cow2wld, v))


def onKeyPress(window, key, scancode, action, mods):
    global cameraIndex, wld2cam
    if action == glfw.RELEASE:
        return  # do nothing
    # If 'c' or space bar are pressed, alter the camera.
    # If a number is pressed, alter the camera corresponding the number.
    if key == glfw.KEY_C or key == glfw.KEY_SPACE:
        print("Toggle camera %s\n" % cameraIndex)
        cameraIndex += 1

    if key == glfw.KEY_1:
        cameraIndex = 0

    if key == glfw.KEY_2:
        cameraIndex = 1

    if key == glfw.KEY_3:
        cameraIndex = 2

    if key == glfw.KEY_4:
        cameraIndex = 3

    if key == glfw.KEY_5:
        cameraIndex = 4

    if cameraIndex >= len(wld2cam):
        cameraIndex = 0


def drawOtherCamera():
    global cameraIndex, wld2cam, camModel
    for i in range(len(wld2cam)):
        if (i != cameraIndex):
            glPushMatrix()  # Push the current matrix on GL to stack. The matrix is wld2cam[cameraIndex].matrix().
            glMultMatrixd(cam2wld[i].T)
            drawFrame(5)  # Draw x, y, and z axis.
            frontColor = [0.2, 0.2, 0.2, 1.0]
            glEnable(GL_LIGHTING)
            glMaterialfv(GL_FRONT, GL_AMBIENT, frontColor)  # Set ambient property frontColor.
            glMaterialfv(GL_FRONT, GL_DIFFUSE, frontColor)  # Set diffuse property frontColor.
            glScaled(0.5, 0.5, 0.5)  # Reduce camera size by 1/2.
            glTranslated(1.1, 1.1, 0.0)  # Translate it (1.1, 1.1, 0.0).
            camModel.render()
            glPopMatrix()  # Call the matrix on stack. wld2cam[cameraIndex].matrix() in here.


def drawFrame(leng):
    glDisable(GL_LIGHTING)  # Lighting is not needed for drawing axis.
    glBegin(GL_LINES)  # Start drawing lines.
    glColor3d(1, 0, 0)  # color of x-axis is red.
    glVertex3d(0, 0, 0)
    glVertex3d(leng, 0, 0)  # Draw line(x-axis) from (0,0,0) to (len, 0, 0).
    glColor3d(0, 1, 0)  # color of y-axis is green.
    glVertex3d(0, 0, 0)
    glVertex3d(0, leng, 0)  # Draw line(y-axis) from (0,0,0) to (0, len, 0).
    glColor3d(0, 0, 1)  # color of z-axis is  blue.
    glVertex3d(0, 0, 0)
    glVertex3d(0, 0, leng)  # Draw line(z-axis) from (0,0,0) - (0, 0, len).
    glEnd()  # End drawing lines.


# *********************************************************************************
# Draw 'cow' object.
# *********************************************************************************/
def drawCow(_cow2wld, drawBB):
    glPushMatrix()  # Push the current matrix of GL into stack. This is because the matrix of GL will be change while drawing cow.

    # The information about location of cow to be drawn is stored in cow2wld matrix.
    # (Project2 hint) If you change the value of the cow2wld matrix or the current matrix, cow would rotate or move.
    glMultMatrixd(_cow2wld.T)

    drawFrame(5)  # Draw x, y, and z axis.
    frontColor = [0.8, 0.2, 0.9, 1.0]
    glEnable(GL_LIGHTING)
    glMaterialfv(GL_FRONT, GL_AMBIENT, frontColor)  # Set ambient property frontColor.
    glMaterialfv(GL_FRONT, GL_DIFFUSE, frontColor)  # Set diffuse property frontColor.
    cowModel.render()  # Draw cow.
    glDisable(GL_LIGHTING)
    if drawBB:
        glBegin(GL_LINES)
        glColor3d(1, 1, 1)
        cow = cowModel
        glVertex3d(cow.bbmin[0], cow.bbmin[1], cow.bbmin[2])
        glVertex3d(cow.bbmax[0], cow.bbmin[1], cow.bbmin[2])
        glVertex3d(cow.bbmin[0], cow.bbmax[1], cow.bbmin[2])
        glVertex3d(cow.bbmax[0], cow.bbmax[1], cow.bbmin[2])
        glVertex3d(cow.bbmin[0], cow.bbmin[1], cow.bbmax[2])
        glVertex3d(cow.bbmax[0], cow.bbmin[1], cow.bbmax[2])
        glVertex3d(cow.bbmin[0], cow.bbmax[1], cow.bbmax[2])
        glVertex3d(cow.bbmax[0], cow.bbmax[1], cow.bbmax[2])

        glColor3d(1, 1, 1)
        glVertex3d(cow.bbmin[0], cow.bbmin[1], cow.bbmin[2])
        glVertex3d(cow.bbmin[0], cow.bbmax[1], cow.bbmin[2])
        glVertex3d(cow.bbmax[0], cow.bbmin[1], cow.bbmin[2])
        glVertex3d(cow.bbmax[0], cow.bbmax[1], cow.bbmin[2])
        glVertex3d(cow.bbmin[0], cow.bbmin[1], cow.bbmax[2])
        glVertex3d(cow.bbmin[0], cow.bbmax[1], cow.bbmax[2])
        glVertex3d(cow.bbmax[0], cow.bbmin[1], cow.bbmax[2])
        glVertex3d(cow.bbmax[0], cow.bbmax[1], cow.bbmax[2])

        glColor3d(1, 1, 1)
        glVertex3d(cow.bbmin[0], cow.bbmin[1], cow.bbmin[2])
        glVertex3d(cow.bbmin[0], cow.bbmin[1], cow.bbmax[2])
        glVertex3d(cow.bbmax[0], cow.bbmin[1], cow.bbmin[2])
        glVertex3d(cow.bbmax[0], cow.bbmin[1], cow.bbmax[2])
        glVertex3d(cow.bbmin[0], cow.bbmax[1], cow.bbmin[2])
        glVertex3d(cow.bbmin[0], cow.bbmax[1], cow.bbmax[2])
        glVertex3d(cow.bbmax[0], cow.bbmax[1], cow.bbmin[2])
        glVertex3d(cow.bbmax[0], cow.bbmax[1], cow.bbmax[2])

        glColor3d(1, 1, 1)
        glVertex3d(cow.bbmin[0], cow.bbmin[1], cow.bbmin[2])
        glVertex3d(cow.bbmin[0], cow.bbmax[1], cow.bbmin[2])
        glVertex3d(cow.bbmax[0], cow.bbmin[1], cow.bbmin[2])
        glVertex3d(cow.bbmax[0], cow.bbmax[1], cow.bbmin[2])
        glVertex3d(cow.bbmin[0], cow.bbmin[1], cow.bbmax[2])
        glVertex3d(cow.bbmin[0], cow.bbmax[1], cow.bbmax[2])
        glVertex3d(cow.bbmax[0], cow.bbmin[1], cow.bbmax[2])
        glVertex3d(cow.bbmax[0], cow.bbmax[1], cow.bbmax[2])

        glColor3d(1, 1, 1)
        glVertex3d(cow.bbmin[0], cow.bbmin[1], cow.bbmin[2])
        glVertex3d(cow.bbmin[0], cow.bbmin[1], cow.bbmax[2])
        glVertex3d(cow.bbmax[0], cow.bbmin[1], cow.bbmin[2])
        glVertex3d(cow.bbmax[0], cow.bbmin[1], cow.bbmax[2])
        glVertex3d(cow.bbmin[0], cow.bbmax[1], cow.bbmin[2])
        glVertex3d(cow.bbmin[0], cow.bbmax[1], cow.bbmax[2])
        glVertex3d(cow.bbmax[0], cow.bbmax[1], cow.bbmin[2])
        glVertex3d(cow.bbmax[0], cow.bbmax[1], cow.bbmax[2])
        glEnd()
    glPopMatrix()  # Pop the matrix in stack to GL. Change it the matrix before drawing cow.


def drawFloor():
    glDisable(GL_LIGHTING)

    # Set color of the floor.
    # Assign checker-patterned texture.
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, floorTexID)

    # Draw the floor. Match the texture's coordinates and the floor's coordinates resp.
    nrep = 4
    glBegin(GL_POLYGON)
    glTexCoord2d(0, 0)
    glVertex3d(-12, -0.1, -12)  # Texture's (0,0) is bound to (-12,-0.1,-12).
    glTexCoord2d(nrep, 0)
    glVertex3d(12, -0.1, -12)  # Texture's (1,0) is bound to (12,-0.1,-12).
    glTexCoord2d(nrep, nrep)
    glVertex3d(12, -0.1, 12)  # Texture's (1,1) is bound to (12,-0.1,12).
    glTexCoord2d(0, nrep)
    glVertex3d(-12, -0.1, 12)  # Texture's (0,1) is bound to (-12,-0.1,12).
    glEnd()

    glDisable(GL_TEXTURE_2D)
    drawFrame(5)  # Draw x, y, and z axis.


##################################################################################################################
##################################################################################################################
##################################################################################################################

def display():
    global cameraIndex, cow2wld, DrawLine, animstartTime, cursorOnCowBoundingBox, wld2cam, tempDrawCow, tempCow
    glClearColor(0.8, 0.9, 0.9, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear the screen
    # set viewing transformation.
    glLoadMatrixd(wld2cam[cameraIndex].T)

    drawOtherCamera()  # Locate the camera's position, and draw all of them.
    drawFloor()  # Draw floor.

    # TODO:
    # update cow2wld here to animate the cow.
    # animTime=glfw.get_time()-animStartTime;
    # you need to modify both the translation and rotation parts of the cow2wld matrix.
    if not DrawLine:
        drawCow(cow2wld, cursorOnCowBoundingBox)  # Draw cow.
        for i in range(0, len(tempCow)):
            drawCow(tempCow[i], False)

    else:
        animTime = float(glfw.get_time() - animstartTime)
        idx = int(animTime / 0.01)
        drawCow(tempDrawCow[idx % len(tempDrawCow)], False)

        if idx > len(tempDrawCow):
            DrawLine = False
            tempCow.clear()
            tempDrawCow.clear()

    glFlush()


def reshape(window, w, h):
    width = w
    height = h
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)  # Select The Projection Matrix
    glLoadIdentity()  # Reset The Projection Matrix
    # Define perspective projection frustum
    aspect = width / (float)(height)
    gluPerspective(45, aspect, 1, 1024)
    matProjection = glGetDoublev(GL_PROJECTION_MATRIX).T
    glMatrixMode(GL_MODELVIEW)  # Select The Modelview Matrix
    glLoadIdentity()  # Reset The Projection Matrix


def initialize(window):
    global cursorOnCowBoundingBox, floorTexID, cameraIndex, camModel, cow2wld, cowModel
    cursorOnCowBoundingBox = False
    # Set up OpenGL state
    glShadeModel(GL_SMOOTH)  # Set Smooth Shading
    glEnable(GL_DEPTH_TEST)  # Enables Depth Testing
    glDepthFunc(GL_LEQUAL)  # The Type Of Depth Test To Do
    # Use perspective correct interpolation if available
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
    # Initialize the matrix stacks
    width, height = glfw.get_window_size(window)
    reshape(window, width, height)
    # Define lighting for the scene
    lightDirection = [1.0, 1.0, 1.0, 0]
    ambientIntensity = [0.1, 0.1, 0.1, 1.0]
    lightIntensity = [0.9, 0.9, 0.9, 1.0]
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambientIntensity)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightIntensity)
    glLightfv(GL_LIGHT0, GL_POSITION, lightDirection)
    glEnable(GL_LIGHT0)

    # initialize floor
    im = open('bricks.bmp')
    try:
        ix, iy, image = im.size[0], im.size[1], im.tobytes("raw", "RGB", 0, -1)
    except SystemError:
        ix, iy, image = im.size[0], im.size[1], im.tobytes("raw", "RGBX", 0, -1)

    # Make texture which is accessible through floorTexID.
    floorTexID = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, floorTexID)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
    glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, ix, 0, GL_RGB, GL_UNSIGNED_BYTE, image)
    # initialize cow
    cowModel = OBJ.OBJrenderer("cow.obj")

    # initialize cow2wld matrix
    glPushMatrix()  # Push the current matrix of GL into stack.
    glLoadIdentity()  # Set the GL matrix Identity matrix.
    glTranslated(0, -cowModel.bbmin[1], -8)  # Set the location of cow.
    glRotated(-90, 0, 1, 0)  # Set the direction of cow. These information are stored in the matrix of GL.
    cow2wld = glGetDoublev(GL_MODELVIEW_MATRIX).T  # convert column-major to row-major
    glPopMatrix()  # Pop the matrix on stack to GL.

    # intialize camera model.
    camModel = OBJ.OBJrenderer("camera.obj")

    # initialize camera frame transforms.

    cameraCount = len(cameras)
    for i in range(cameraCount):
        # 'c' points the coordinate of i-th camera.
        c = cameras[i]
        glPushMatrix()  # Push the current matrix of GL into stack.
        glLoadIdentity()  # Set the GL matrix Identity matrix.
        gluLookAt(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8])  # Setting the coordinate of camera.
        wld2cam.append(glGetDoublev(GL_MODELVIEW_MATRIX).T)
        glPopMatrix()  # Transfer the matrix that was pushed the stack to GL.
        cam2wld.append(np.linalg.inv(wld2cam[i]))
    cameraIndex = 0

xxx = 0
def onMouseButton(window, button, state, mods):
    global isDrag, V_DRAG, H_DRAG, before_x, before_y, IsNotFirst, DrawLine, animstartTime, Initial
    global cursorOnCowBoundingBox, tempCow, tempDrawCow, spline_points, cow2wld, xxx
    GLFW_DOWN = 1
    GLFW_UP = 0
    x, y = glfw.get_cursor_pos(window)

    if button == glfw.MOUSE_BUTTON_LEFT:
        if state == GLFW_DOWN:
            isDrag = V_DRAG
            xxx = x

        elif state == GLFW_UP:
            if IsNotFirst:
                if not DrawLine:
                    tempCow.append(cow2wld)
                    if len(tempCow) == 6:

                        spline_points = points_to_spline(tempCow)
                        cow_now = tempCow[0]

                        for i in range(0, len(spline_points)):
                            tempvec3 = vector3(cow_now[0][3], cow_now[1][3], cow_now[2][3])
                            dir = spline_points[i] - tempvec3
                            back_dir = vector3(0, 0, 0) - tempvec3

                            backT = np.eye(4)
                            setTranslation(backT, back_dir)
                            frontT = np.eye(4)
                            setTranslation(frontT, tempvec3)

                            tangentVec = spline_points[(i + 1) % len(spline_points)] - spline_points[i]
                            tv_xz = vector3(tangentVec[0], 0, tangentVec[2])
                            orivec = vector3(0, 0, 1)
                            orivec2 = vector3(0, 1, 0)

                            angle_xz = angle(orivec, tv_xz)
                            angle_yz = angle(orivec2, tangentVec)

                            angle_yz = np.pi / 2 - angle_yz
                            angle_yz = 2 * np.pi - angle_yz

                            crv_y = np.cross(orivec, tv_xz)

                            if crv_y[1] < 0:
                                angle_xz = 2 * np.pi - angle_xz

                            rot_x = np.array([[1, 0, 0, 0],
                                              [0, np.cos(angle_yz), -np.sin(angle_yz), 0],
                                              [0, np.sin(angle_yz), np.cos(angle_yz), 0],
                                              [0, 0, 0, 1]])

                            rot_y = np.array([[np.cos(angle_xz), 0, np.sin(angle_xz), 0],
                                              [0, 1, 0, 0],
                                              [-np.sin(angle_xz), 0, np.cos(angle_xz), 0],
                                              [0, 0, 0, 1]])

                            currentMatrix = cow_now
                            currentMatrix = backT @ currentMatrix
                            currentMatrix = rot_x @ currentMatrix
                            currentMatrix = rot_y @ currentMatrix
                            currentMatrix = frontT @ currentMatrix

                            T = np.eye(4)
                            setTranslation(T, dir)
                            currentMatrix = T @ currentMatrix
                            tempDrawCow.append(currentMatrix)

                        animstartTime = glfw.get_time()
                        DrawLine = True
            if not IsNotFirst:
                IsNotFirst = True
                Initial = True

            isDrag = H_DRAG
            
            # print("Left mouse up")
            # start horizontal dragging using mouse-move events.
    elif button == glfw.MOUSE_BUTTON_RIGHT:
        if state == GLFW_DOWN:
            print("Right mouse click at ", (x, y))


def onMouseDrag(window, x, y):
    global isDrag, cursorOnCowBoundingBox, pickInfo, cow2wld, Initial, IsNotFirst, before_h, before_v, tempCow, tempDrawCow, cowPickLocalPos, cowPickPosition
    if isDrag:
        print( "in drag mode %d\n"% isDrag)
        if isDrag == V_DRAG:
            # vertical dragging
            # TODO:
            # create a dragging plane perpendicular to the ray direction,
            # and test intersection with the screen ray.
            if cursorOnCowBoundingBox:
                ray = screenCoordToRay(window, xxx, y)
                pp = pickInfo

                if IsNotFirst and not Initial:
                    print("Down change\n")
                    pp.cowPickPosition = before_h
                    pp.cowPickConfiguration = cow2wld
                    invWorld = np.linalg.inv(cow2wld)
                    pp.cowPickPositionLocal = mult(invWorld, pp.cowPickPosition)
                    Initial = True

                p = Plane(np.array([1, 0, 0]), pp.cowPickPosition)
                c = ray.intersectsPlane(p)

                currentPos = ray.getPoint(c[1])
                print(pp.cowPickPosition, currentPos)
                print(pp.cowPickConfiguration, cow2wld)

                T = np.eye(4)
                setTranslation(T, currentPos - pp.cowPickPosition)
                cow2wld = T @ pp.cowPickConfiguration
                before_v = currentPos

        else:
            # horizontal dragging
            # Hint: read carefully the following block to implement vertical dragging.
            if cursorOnCowBoundingBox:
                ray = screenCoordToRay(window, x, y)
                pp = pickInfo

                if Initial:
                    pp.cowPickPosition = before_v
                    pp.cowPickConfiguration = cow2wld
                    invWord = np.linalg.inv(cow2wld)
                    pp.cowPickPositionLocal = mult(invWord, pp.cowPickPosition)
                    Initial = False

                p = Plane(np.array([0, 1, 0]), pp.cowPickPosition)
                c = ray.intersectsPlane(p)

                currentPos = ray.getPoint(c[1])

                T = np.eye(4)
                setTranslation(T, currentPos - pp.cowPickPosition)
                cow2wld = T @ pp.cowPickConfiguration

    else:
        ray = screenCoordToRay(window, x, y)

        planes = []
        cow = cowModel
        bbmin = cow.bbmin
        bbmax = cow.bbmax

        planes.append(makePlane(bbmin, bbmax, vector3(0, 1, 0)))
        planes.append(makePlane(bbmin, bbmax, vector3(0, -1, 0)))
        planes.append(makePlane(bbmin, bbmax, vector3(1, 0, 0)))
        planes.append(makePlane(bbmin, bbmax, vector3(-1, 0, 0)))
        planes.append(makePlane(bbmin, bbmax, vector3(0, 0, 1)))
        planes.append(makePlane(bbmin, bbmax, vector3(0, 0, -1)))

        o = ray.intersectsPlanes(planes)
        cursorOnCowBoundingBox = o[0]
        pp = pickInfo
        pp.cursorRayT = o[1]
        pp.cowPickPosition = ray.getPoint(pp.cursorRayT)
        pp.cowPickConfiguration = cow2wld

        invWord = np.linalg.inv(cow2wld)
        pp.cowPickPositionLocal = mult(invWord, pp.cowPickPosition)
        before_v = before_h = pp.cowPickPosition


def screenCoordToRay(window, x, y):
    width, height = glfw.get_window_size(window)

    matProjection = glGetDoublev(GL_PROJECTION_MATRIX).T
    matProjection = matProjection @ wld2cam[cameraIndex]  # use @ for matrix mult.
    invMatProjection = np.linalg.inv(matProjection)
    # -1<=v.x<1 when 0<=x<width
    # -1<=v.y<1 when 0<=y<height
    vecAfterProjection = vector4(
        (float(x - 0)) / (float(width)) * 2.0 - 1.0,
        -1 * (((float(y - 0)) / float(height)) * 2.0 - 1.0),
        -10)

    vecBeforeProjection = position3(invMatProjection @ vecAfterProjection)

    rayOrigin = getTranslation(cam2wld[cameraIndex])
    return Ray(rayOrigin, normalize(vecBeforeProjection - rayOrigin))


def main():
    if not glfw.init():
        print('GLFW initialization failed')
        sys.exit(-1)
    width = 800
    height = 600
    window = glfw.create_window(width, height, '2017029716-PA2', None, None)
    if not window:
        glfw.terminate()
        sys.exit(-1)

    glfw.make_context_current(window)
    glfw.set_key_callback(window, onKeyPress)
    glfw.set_mouse_button_callback(window, onMouseButton)
    glfw.set_cursor_pos_callback(window, onMouseDrag)
    glfw.swap_interval(1)

    initialize(window)
    while not glfw.window_should_close(window):
        glfw.poll_events()
        display()

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()

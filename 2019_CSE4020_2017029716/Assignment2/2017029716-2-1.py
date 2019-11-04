import glfw
import numpy as np
from OpenGL.GL import *
GL = [GL_POLYGON, GL_POINTS, GL_LINES, GL_LINE_STRIP, GL_LINE_LOOP, GL_TRIANGLES, GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN, GL_QUADS, GL_QUAD_STRIP]
type_key = 4

def render():
    global type_key
    
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()
    glBegin(GL[type_key])
    
    for v in np.linspace(0.0, np.pi*11/6, 12):
        glVertex2f(np.cos(v), np.sin(v))
    glEnd()

def key_callback(window, key, scancode, action, mods):
    
    global type_key
    
    #입력값에 따른 결과 지정
    if action==glfw.PRESS:
        if key == glfw.KEY_0:
           type_key = 0
        elif key == glfw.KEY_1:
           type_key = 1
        elif key == glfw.KEY_2:
           type_key = 2
        elif key == glfw.KEY_3:
           type_key = 3
        elif key == glfw.KEY_4:
           type_key = 4
        elif key == glfw.KEY_5:
           type_key = 5
        elif key == glfw.KEY_6:
           type_key = 6
        elif key == glfw.KEY_7:
           type_key = 7
        elif key == glfw.KEY_8:
           type_key = 8
        elif key == glfw.KEY_9:
           type_key = 9
    
def main():

    if not glfw.init():
        return

    window = glfw.create_window(480,480,"2017029716-2-1", None,None)
    if not window:
        glfw.terminate()
        return

    glfw.set_key_callback(window, key_callback)

    glfw.make_context_current(window)

    while not glfw.window_should_close(window):

        glfw.poll_events()

        render()

        glfw.swap_buffers(window)
        
    glfw.terminate()

if __name__ == "__main__":
    main()

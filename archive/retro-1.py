#!/usr/bin/env python3
import moderngl
import numpy as np
import pyglet
from pyglet.window import key


window = pyglet.window.Window(fullscreen=True, caption="Retro BASIC")
ctx = moderngl.get_context()

prog = ctx.program(
    vertex_shader="""
        #version 330

        in vec2 in_vert;
        in vec3 in_color;

        out vec3 v_color;

        void main() {
            v_color = in_color;
            gl_Position = vec4(in_vert, 0.0, 1.0);
        }
    """,
    fragment_shader="""
        #version 330

        in vec3 v_color;

        out vec3 f_color;

        void main() {
            f_color = v_color;
        }
    """,
)

vertices = np.asarray([

    -0.75, -0.75,  1, 0, 0,
    0.75, -0.75,  0, 1, 0,
    0.0, 0.649,  0, 0, 1

], dtype='f4')

vbo = ctx.buffer(vertices.tobytes())
vao = ctx.vertex_array(prog, vbo, "in_vert", "in_color")


@window.event
def on_key_press(symbol, modifiers):
    if symbol == key.A:
        print('The "A" key was pressed.')
    elif symbol == key.LEFT:
        print('The left arrow key was pressed.')
    elif symbol == key.ENTER:
        print('The enter key was pressed.')


@window.event
def on_resize(width, height):
    # print(f"The window was resized to {width},{height}")
    pass


@window.event
def on_draw():
    ctx.clear()
    vao.render()


pyglet.app.run()

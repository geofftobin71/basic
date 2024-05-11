#!/usr/bin/env python3
import moderngl
import numpy as np
import pyglet
from pyglet.window import key


class Window(pyglet.window.Window):

    def __init__(self):
        super().__init__(fullscreen=True)
        self.ctx = moderngl.get_context()

        self.prog = self.ctx.program(
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

        vertices = np.asarray(
            [-0.75, -0.75, 1, 0, 0, 0.75, -0.75, 0, 1, 0, 0.0, 0.649, 0, 0, 1],
            dtype='f4')

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, self.vbo, "in_vert",
                                         "in_color")

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.close()

    def on_resize(self, width, height):
        pass

    def on_draw(self):
        self.ctx.clear()
        self.vao.render()


if __name__ == '__main__':
    window = Window()
    pyglet.app.run()

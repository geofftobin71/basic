#!/usr/bin/env python3
import moderngl
import numpy as np
import pyglet
from pyglet.window import key


class Display():

    def __init__(self, width, height, ctx):
        self.width = width
        self.height = height
        self.columns = width // 8
        self.rows = height // 8
        self.width = self.columns * 8
        self.height = self.rows * 8
        self.aspect = width / height
        self.ctx = ctx

        self.prog = self.ctx.program(
            vertex_shader="""
            #version 330

            in vec2 in_vert;
            in vec2 in_uv;

            out vec2 v_uv;

            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
                v_uv = in_uv;
            }
            """,
            fragment_shader="""
            #version 330

            uniform sampler2D Texture;

            in vec2 v_uv;

            out vec4 f_color;

            void main() {
                f_color = vec4(texture(Texture, v_uv).rgb, 1.0);
                // vec3(0.5, 0.5, 0.5);
            }
            """,
        )

        pixels = pyglet.image.load("doom.png").get_data()
        self.texture = self.ctx.texture((320, 240), 3, pixels)
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self.vbo = self.ctx.buffer(reserve=4 * 4 * 4)
        self.vao = self.ctx.vertex_array(self.prog, self.vbo, "in_vert",
                                         "in_uv")

    def resize(self, window_aspect):
        display_size = 0.9

        if window_aspect < self.aspect:  # Portrait Device
            vertices = np.array([
                display_size, display_size * window_aspect / self.aspect, 1.0,
                0.0, -display_size, display_size * window_aspect / self.aspect,
                0.0, 0.0, display_size, -display_size * window_aspect /
                self.aspect, 1.0, 1.0, -display_size,
                -display_size * window_aspect / self.aspect, 0.0, 1.0
            ], dtype='f4')
        else:  # Landscape Device
            vertices = np.array([
                display_size * self.aspect / window_aspect, display_size, 1.0,
                0.0, -display_size * self.aspect / window_aspect, display_size,
                0.0, 0.0, display_size * self.aspect / window_aspect,
                -display_size, 1.0, 1.0, -display_size * self.aspect /
                window_aspect, -display_size, 0.0, 1.0
            ], dtype='f4')

        self.vbo.write(vertices.tobytes())


class Window(pyglet.window.Window):

    def __init__(self):
        super().__init__(resizable=True)  # fullscreen=True)
        self.ctx = moderngl.get_context()

        (w, h) = self.get_framebuffer_size()
        self.aspect = w / h

        self.dsp = Display(320, 240, self.ctx)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.close()

    def on_resize(self, width, height):
        self.viewport = (0, 0, *self.get_framebuffer_size())
        self.aspect = width / height
        self.dsp.resize(self.aspect)

    def on_draw(self):
        self.ctx.clear()
        self.dsp.texture.use()
        self.dsp.vao.render(moderngl.TRIANGLE_STRIP)


if __name__ == '__main__':
    window = Window()
    pyglet.app.run()

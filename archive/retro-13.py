#!/usr/bin/env python3
import moderngl
import numpy as np
import pyglet
from pyglet.window import key

import colors
import fonts
import shaders


class Display():

    def __init__(self, width, height, ctx):
        self.columns = width // 8
        self.rows = height // 8
        self.width = self.columns * 8
        self.height = self.rows * 8

        self.display_texture = ctx.texture((self.width, self.height), 3)
        self.display_texture.repeat_x = False
        self.display_texture.repeat_y = False
        self.display_texture.repeat_z = False

        self.display_fbo = ctx.framebuffer(self.display_texture)
        self.display_fbo.viewport = (0, 0, self.width, self.height)

        self.char_memory = bytearray(self.columns * self.rows * 3)
        for i in range(self.columns * self.rows):
            self.char_memory[i * 3 + 0] = i & 0xFF
            self.char_memory[i * 3 + 1] = i & 0xFF
            self.char_memory[i * 3 + 2] = 0xFF - (i & 0xFF)

        self.char_texture = ctx.texture((self.columns, self.rows),
                                        components=3,
                                        dtype="u1")
        self.update_chars()

        self.font_bitmap = fonts.c64
        self.font_texture = ctx.texture((128, 128), components=1, dtype="f1")
        self.update_font()

        self.colors = colors.xterm

        self.colors_texture = ctx.texture((256, 1), components=3, dtype='f1')
        self.update_colors()

        self.display_prog = ctx.program(shaders.display_vs, shaders.display_fs)

        self.display_prog['font_sampler'].value = 1
        self.display_prog['char_sampler'].value = 2
        self.display_prog['colors_sampler'].value = 3

        display_vertices = np.array(
            [1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0], dtype='f4')

        self.display_vbo = ctx.buffer(display_vertices)
        self.display_vao = ctx.vertex_array(self.display_prog,
                                            self.display_vbo,
                                            "position",
                                            mode=moderngl.TRIANGLE_STRIP)

    def update_font(self):
        font_image = bytearray(128 * 128)
        for char in range(256):
            char_x = char & 0xF
            char_y = char >> 4

            for py in range(8):
                for px in range(8):
                    if self.font_bitmap[char * 8 + py] & (1 << px):
                        font_image[128 * (char_y * 8 + py) +
                                   (char_x * 8 + px)] = 0xFF

        self.font_texture.write(font_image)

    def update_chars(self):
        self.char_texture.write(self.char_memory)

    def update_colors(self):
        self.colors_texture.write(self.colors)

    def on_draw(self):
        self.display_fbo.use()
        self.display_fbo.clear()  # clear bg color
        self.font_texture.use(1)
        self.char_texture.use(2)
        self.colors_texture.use(3)
        self.display_vao.render()
        self.display_texture.use(4)


class Window(pyglet.window.Window):

    def __init__(self):
        super().__init__(fullscreen=True)
        self.ctx = moderngl.get_context()

        self.dsp = Display(320, 240, self.ctx)

        self.pixel_upscale_prog = self.ctx.program(shaders.pixel_upscale_vs,
                                                   shaders.pixel_upscale_fs)

        self.pixel_upscale_prog['display_sampler'].value = 4
        self.pixel_upscale_prog['display_size'].value = (self.dsp.width,
                                                         self.dsp.height)

        self.pixel_upscale_vbo = self.ctx.buffer(reserve=4 * 4 * 4)
        self.pixel_upscale_vao = self.ctx.vertex_array(
            self.pixel_upscale_prog,
            self.pixel_upscale_vbo,
            "position",
            "uv",
            mode=moderngl.TRIANGLE_STRIP)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.close()

    def on_resize(self, width, height):
        window_aspect = width / height
        display_aspect = self.dsp.width / self.dsp.height
        display_scale = 0.9

        if window_aspect < display_aspect:  # Portrait Device
            pixel_upscale_vertices = np.array([
                display_scale, display_scale * window_aspect / display_aspect,
                1.0, 0.0, -display_scale, display_scale * window_aspect /
                display_aspect, 0.0, 0.0, display_scale, -display_scale *
                window_aspect / display_aspect, 1.0, 1.0, -display_scale,
                -display_scale * window_aspect / display_aspect, 0.0, 1.0
            ],
                                              dtype='f4')
        else:  # Landscape Device
            pixel_upscale_vertices = np.array([
                display_scale * display_aspect / window_aspect, display_scale,
                1.0, 0.0, -display_scale * display_aspect / window_aspect,
                display_scale, 0.0, 0.0, display_scale * display_aspect /
                window_aspect, -display_scale, 1.0, 1.0, -display_scale *
                display_aspect / window_aspect, -display_scale, 0.0, 1.0
            ],
                                              dtype='f4')

        self.pixel_upscale_vbo.write(pixel_upscale_vertices.tobytes())

    def on_draw(self):
        self.dsp.on_draw()

        self.ctx.screen.use()
        self.viewport = (0, 0, *self.get_framebuffer_size())
        self.ctx.clear()  # clear border color
        self.pixel_upscale_vao.render()


if __name__ == '__main__':
    window = Window()
    pyglet.app.run()

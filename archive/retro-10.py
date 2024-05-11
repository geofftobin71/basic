#!/usr/bin/env python3
import moderngl
import numpy as np
import pyglet
from pyglet.window import key

import fonts
import shaders


class Display():

    def __init__(self, width, height, ctx):
        self.columns = width // 8
        self.rows = height // 8
        self.width = self.columns * 8
        self.height = self.rows * 8
        self.aspect = self.width / self.height
        self.ctx = ctx

        self.screen_texture = self.ctx.texture((self.width, self.height), 3)
        self.screen_texture.repeat_x = False
        self.screen_texture.repeat_y = False
        self.screen_texture.repeat_z = False

        self.screen_fbo = self.ctx.framebuffer(self.screen_texture)
        self.screen_fbo.viewport = (0, 0, self.width, self.height)

        self.char_memory = bytearray(self.columns * self.rows * 3)
        for i in range(len(self.char_memory) // 3):
            self.char_memory[i * 3 + 0] = i & 0xFF  # random.randint(0, 255)
            self.char_memory[i * 3 + 1] = self.char_memory[i * 3 + 0]
            self.char_memory[i * 3 + 2] = self.char_memory[i * 3 + 0]

        self.char_texture = self.ctx.texture((self.columns, self.rows),
                                             components=3,
                                             dtype="u1")
        self.char_texture.write(self.char_memory)

        self.font_bitmap = fonts.c64

        font_image = bytearray(128 * 128)
        for char in range(256):
            char_x = char & 0xF
            char_y = char >> 4

            for py in range(8):
                for px in range(8):
                    if self.font_bitmap[char * 8 + py] & (1 << px):
                        font_image[128 * (char_y * 8 + py) +
                                   (char_x * 8 + px)] = 0xFF

        self.font_texture = self.ctx.texture((128, 128),
                                             components=1,
                                             dtype="f1")
        self.font_texture.write(font_image)

        self.screen_prog = self.ctx.program(shaders.screen_vs,
                                            shaders.screen_fs)

        self.screen_prog['font_sampler'].value = 1
        self.screen_prog['char_sampler'].value = 2

        screen_vertices = np.array(
            [1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0], dtype='f4')
        self.screen_vbo = self.ctx.buffer(screen_vertices)
        self.screen_vao = self.ctx.vertex_array(self.screen_prog,
                                                self.screen_vbo,
                                                "position",
                                                mode=moderngl.TRIANGLE_STRIP)

        self.pixel_upscale_prog = self.ctx.program(shaders.pixel_upscale_vs,
                                                   shaders.pixel_upscale_fs)

        self.pixel_upscale_prog['screen_sampler'].value = 3
        self.pixel_upscale_prog['screen_size'].value = (self.width,
                                                        self.height)

        self.pixel_upscale_vbo = self.ctx.buffer(reserve=4 * 4 * 4)
        self.pixel_upscale_vao = self.ctx.vertex_array(
            self.pixel_upscale_prog,
            self.pixel_upscale_vbo,
            "position",
            "uv",
            mode=moderngl.TRIANGLE_STRIP)

    def on_draw(self):
        self.screen_fbo.use()
        self.screen_fbo.clear()  # clear bg color
        self.font_texture.use(1)
        self.char_texture.use(2)
        self.screen_vao.render()

    def on_resize(self, window_width, window_height):
        window_aspect = window_width / window_height
        display_scale = 0.9

        if window_aspect < self.aspect:  # Portrait Device
            pixel_upscale_vertices = np.array([
                display_scale, display_scale * window_aspect / self.aspect,
                1.0, 0.0, -display_scale, display_scale * window_aspect /
                self.aspect, 0.0, 0.0, display_scale, -display_scale *
                window_aspect / self.aspect, 1.0, 1.0, -display_scale,
                -display_scale * window_aspect / self.aspect, 0.0, 1.0
            ],
                                              dtype='f4')
        else:  # Landscape Device
            pixel_upscale_vertices = np.array([
                display_scale * self.aspect / window_aspect, display_scale,
                1.0, 0.0, -display_scale * self.aspect / window_aspect,
                display_scale, 0.0, 0.0, display_scale * self.aspect /
                window_aspect, -display_scale, 1.0, 1.0, -display_scale *
                self.aspect / window_aspect, -display_scale, 0.0, 1.0
            ],
                                              dtype='f4')

        self.pixel_upscale_vbo.write(pixel_upscale_vertices.tobytes())


class Window(pyglet.window.Window):

    def __init__(self):
        super().__init__(fullscreen=True)
        self.ctx = moderngl.get_context()

        self.dsp = Display(320, 240, self.ctx)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.close()

    def on_resize(self, width, height):
        self.dsp.on_resize(width, height)

    def on_draw(self):
        self.dsp.on_draw()

        self.ctx.screen.use()
        self.viewport = (0, 0, *self.get_framebuffer_size())
        self.ctx.clear()  # clear border color
        self.dsp.screen_texture.use(3)
        self.dsp.pixel_upscale_vao.render()


if __name__ == '__main__':
    window = Window()
    pyglet.app.run()

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

        self.fg_color = 7
        self.bg_color = 0
        self.clear_color = 0
        self.border_color = 232

        self.cursor_x = 0
        self.cursor_y = 0

        self.display_texture = ctx.texture((self.width, self.height), 3)
        self.display_texture.repeat_x = False
        self.display_texture.repeat_y = False
        self.display_texture.repeat_z = False

        self.display_fbo = ctx.framebuffer(self.display_texture)
        self.display_fbo.viewport = (0, 0, self.width, self.height)

        self.char_memory = bytearray(self.columns * self.rows * 3)
        self.char_texture = ctx.texture((self.columns, self.rows),
                                        components=3,
                                        dtype="u1")
        self.cls()

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

    def cls(self):
        for i in range(self.columns * self.rows):
            self.char_memory[i * 3 + 0] = 0
            self.char_memory[i * 3 + 1] = self.fg_color
            self.char_memory[i * 3 + 2] = self.clear_color
        self.update_chars()
        self.cursor_x = 0
        self.cursor_y = 0

    def get_border_color_3f(self):
        return (float(self.colors[self.border_color * 3 + 0]) / 0xFF,
                float(self.colors[self.border_color * 3 + 1]) / 0xFF,
                float(self.colors[self.border_color * 3 + 2]) / 0xFF)

    def scroll(self):
        tmp = self.char_memory[self.columns * 3:]
        self.cls()
        self.char_memory[:-self.columns * 3] = tmp
        self.update_chars()

    def print(self, string):
        for c in string:
            if c == '\n':
                self.cursor_x = 0
                self.cursor_y += 1
                if self.cursor_y == self.rows:
                    self.scroll()
                    self.cursor_y -= 1
            else:
                address = (self.columns * self.cursor_y + self.cursor_x) * 3
                self.char_memory[address] = ord(c)
                self.char_memory[address+1] = self.fg_color
                self.char_memory[address+2] = self.bg_color
                self.cursor_x += 1

                if self.cursor_x == self.columns:
                    self.cursor_x = 0
                    self.cursor_y += 1

                if self.cursor_y == self.rows:
                    self.scroll()
                    self.cursor_y -= 1

        self.update_chars()

    def on_draw(self):
        self.display_fbo.use()
        self.display_fbo.clear()  # clear bg color
        self.font_texture.use(1)
        self.char_texture.use(2)
        self.colors_texture.use(3)
        self.display_vao.render()
        self.display_texture.use(4)


class Window(pyglet.window.Window):

    def __init__(self, width, height):
        super().__init__(fullscreen=True)
        self.ctx = moderngl.get_context()

        self.dsp = Display(width, height, self.ctx)

        self.pixel_upscale_prog = self.ctx.program(shaders.pixel_upscale_vs,
                                                   shaders.pixel_upscale_fs)

        self.pixel_upscale_prog['display_sampler'].value = 4
        self.pixel_upscale_prog['display_size'].value = (width, height)

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
        self.ctx.clear(*self.dsp.get_border_color_3f())
        self.pixel_upscale_vao.render()

    def cls(self):
        self.dsp.cls()

    def print(self, string):
        self.dsp.print(string)

    def fg_color(self, index):
        self.dsp.fg_color = index

    def bg_color(self, index):
        self.dsp.bg_color = index

    def border_color(self, index):
        self.dsp.border_color = index

    def clear_color(self, index):
        self.dsp.clear_color = index


if __name__ == '__main__':
    window = Window(320, 240)

    window.print("01\n")
    window.print("02\n")
    window.print("03\n")
    window.print("04\n")
    window.print("05\n")
    window.print("06\n")
    window.print("07\n")
    window.print("08\n")
    window.print("09\n")
    window.print("10\n")
    window.print("11\n")
    window.print("12\n")
    window.print("13\n")
    window.print("14\n")
    window.print("15\n")
    window.print("16\n")
    window.print("17\n")
    window.print("18\n")
    window.print("19\n")
    window.print("20\n")
    window.print("21\n")
    window.print("22\n")
    window.print("23\n")
    window.print("24\n")
    window.print("25\n")
    window.print("26\n")
    window.print("27\n")
    window.print("28\n")
    window.print("29\n")
    window.fg_color(9)
    window.bg_color(14)
    window.print("30\n")
    window.fg_color(7)
    window.bg_color(0)
    window.print("31\n")

    pyglet.app.run()

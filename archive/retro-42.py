#!/usr/bin/env python3
from enum import Enum

import moderngl
import numpy as np
import pyglet
from pyglet.window import key

import colors
import fonts
import shaders


class InputMode(Enum):
    READY = 0
    EDITING = 1
    DONE = 2


class Window(pyglet.window.Window):

    def __init__(self, display_width, display_height, display_aspect=None):
        super().__init__(fullscreen=True)
        self.ctx = moderngl.get_context()

        self.columns = display_width // 8
        self.rows = display_height // 8
        self.display_width = self.columns * 8
        self.display_height = self.rows * 8
        self.display_aspect = (display_aspect if display_aspect is not None
                               else self.display_width / self.display_height)

        self.fg_color = 14
        self.bg_color = 6
        self.border_color = 14
        self.cursor_color = 14
        self.clear_color = self.bg_color

        self.print_cursor_x = 0
        self.print_cursor_y = 0

        self.display_texture = self.ctx.texture(
            (self.display_width, self.display_height), 3)
        self.display_texture.repeat_x = False
        self.display_texture.repeat_y = False
        self.display_texture.repeat_z = False

        self.display_fbo = self.ctx.framebuffer(self.display_texture)
        self.display_fbo.viewport = (0, 0, self.display_width,
                                     self.display_height)

        self.char_memory = bytearray(self.columns * self.rows * 3)
        self.char_texture = self.ctx.texture((self.columns, self.rows),
                                             components=3,
                                             dtype="u1")
        self.cls()

        self.font_bitmap = fonts.c64
        self.font_texture = self.ctx.texture((128, 128),
                                             components=1,
                                             dtype="f1")
        self.update_font()

        self.colors = colors.c64
        self.colors_texture = self.ctx.texture((256, 1),
                                               components=3,
                                               dtype='f1')
        self.update_colors()

        self.display_prog = self.ctx.program(shaders.display_vs,
                                             shaders.display_fs)

        self.display_prog['font_sampler'].value = 1
        self.display_prog['char_sampler'].value = 2
        self.display_prog['colors_sampler'].value = 3

        display_vertices = np.array(
            [1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0], dtype='f4')

        self.display_vbo = self.ctx.buffer(display_vertices)
        self.display_vao = self.ctx.vertex_array(self.display_prog,
                                                 self.display_vbo,
                                                 "position",
                                                 mode=moderngl.TRIANGLE_STRIP)

        self.pixel_upscale_prog = self.ctx.program(shaders.pixel_upscale_vs,
                                                   shaders.pixel_upscale_fs)

        self.pixel_upscale_prog['display_sampler'].value = 4
        self.pixel_upscale_prog['display_size'].value = (display_width,
                                                         display_height)

        self.pixel_upscale_vbo = self.ctx.buffer(reserve=4 * 4 * 4)
        self.pixel_upscale_vao = self.ctx.vertex_array(
            self.pixel_upscale_prog,
            self.pixel_upscale_vbo,
            "position",
            "uv",
            mode=moderngl.TRIANGLE_STRIP)

        self.input_mode = InputMode.READY
        self.input_buffer = bytearray()
        self.input_buffer_cursor = 0
        self.input_start_x = self.print_cursor_x
        self.input_start_y = self.print_cursor_y

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.close()

    def on_text(self, text):
        if self.input_mode != InputMode.EDITING:
            return

        if ord(text) == 13:
            self.end_input()
            return

        self.poke_bg(self.input_start_x + self.input_buffer_cursor,
                     self.input_start_y, self.bg_color)
        self.input_buffer.insert(self.input_buffer_cursor, ord(text))
        self.input_buffer_cursor += 1

        self.print_cursor_x = self.input_start_x
        self.print_cursor_y = self.input_start_y
        self.print(f"{self.input_buffer.decode()} ")

        self.poke_bg(self.input_start_x + self.input_buffer_cursor,
                     self.input_start_y, self.cursor_color)
        self.update_chars()

    def on_text_motion(self, motion):
        if self.input_mode != InputMode.EDITING:
            return

        # print(key.motion_string(motion))
        self.poke_bg(self.input_start_x + self.input_buffer_cursor,
                     self.input_start_y, self.bg_color)

        if (motion == key.MOTION_BACKSPACE and self.input_buffer_cursor > 0):
            del self.input_buffer[self.input_buffer_cursor -
                                  1:self.input_buffer_cursor]
            self.input_buffer_cursor -= 1

        elif (motion == key.MOTION_DELETE
              and self.input_buffer_cursor < len(self.input_buffer)):
            del self.input_buffer[self.input_buffer_cursor:self.
                                  input_buffer_cursor + 1]

        elif (motion == key.MOTION_LEFT and self.input_buffer_cursor > 0):
            self.input_buffer_cursor -= 1

        elif (motion == key.MOTION_RIGHT
              and self.input_buffer_cursor < len(self.input_buffer)):
            self.input_buffer_cursor += 1

        elif (motion == key.MOTION_BEGINNING_OF_LINE):
            self.input_buffer_cursor = 0

        elif (motion == key.MOTION_END_OF_LINE):
            self.input_buffer_cursor = len(self.input_buffer)

        self.print_cursor_x = self.input_start_x
        self.print_cursor_y = self.input_start_y
        self.print(f"{self.input_buffer.decode()} ")

        self.poke_bg(self.input_start_x + self.input_buffer_cursor,
                     self.input_start_y, self.cursor_color)
        self.update_chars()

    def on_resize(self, window_width, window_height):
        window_aspect = window_width / window_height
        display_scale = 0.9

        if window_aspect < self.display_aspect:  # Portrait Device
            aspect = window_aspect / self.display_aspect
            pixel_upscale_vertices = np.array([
                display_scale, display_scale * aspect, 1.0, 0.0,
                -display_scale, display_scale * aspect, 0.0, 0.0,
                display_scale, -display_scale * aspect, 1.0, 1.0,
                -display_scale, -display_scale * aspect, 0.0, 1.0
            ],
                                              dtype='f4')
        else:  # Landscape Device
            aspect = self.display_aspect / window_aspect
            pixel_upscale_vertices = np.array([
                display_scale * aspect, display_scale, 1.0, 0.0,
                -display_scale * aspect, display_scale, 0.0, 0.0,
                display_scale * aspect, -display_scale, 1.0, 1.0,
                -display_scale * aspect, -display_scale, 0.0, 1.0
            ],
                                              dtype='f4')

        self.pixel_upscale_vbo.write(pixel_upscale_vertices.tobytes())

    def on_draw(self):
        if self.input_mode == InputMode.EDITING:
            self.poke_bg(self.input_start_x + self.input_buffer_cursor,
                         self.input_start_y, self.cursor_color)

        self.update_chars()
        self.display_fbo.use()
        self.display_fbo.clear()
        self.font_texture.use(1)
        self.char_texture.use(2)
        self.colors_texture.use(3)
        self.display_vao.render()

        self.display_texture.use(4)
        self.ctx.screen.use()
        self.viewport = (0, 0, *self.get_framebuffer_size())
        self.ctx.clear(*self.get_border_color_3f())
        self.pixel_upscale_vao.render()

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
        self.clear_color = self.bg_color
        for i in range(self.columns * self.rows):
            self.char_memory[i * 3 + 0] = 0
            self.char_memory[i * 3 + 1] = self.fg_color
            self.char_memory[i * 3 + 2] = self.clear_color
        self.update_chars()
        self.print_cursor_x = 0
        self.print_cursor_y = 0

    def get_border_color_3f(self):
        return (float(self.colors[self.border_color * 3 + 0]) / 0xFF,
                float(self.colors[self.border_color * 3 + 1]) / 0xFF,
                float(self.colors[self.border_color * 3 + 2]) / 0xFF)

    def scroll(self):
        tmp = self.char_memory[self.columns * 3:]
        for i in range(self.columns * self.rows):
            self.char_memory[i * 3 + 0] = 0
            self.char_memory[i * 3 + 1] = self.fg_color
            self.char_memory[i * 3 + 2] = self.clear_color
        self.char_memory[:-self.columns * 3] = tmp
        self.update_chars()
        self.print_cursor_y -= 1
        self.input_start_y -= 1

    def start_input(self, prompt):
        self.print(f"{prompt}? ")
        self.input_buffer = bytearray()
        self.input_buffer_cursor = 0
        self.input_start_x = self.print_cursor_x
        self.input_start_y = self.print_cursor_y
        self.input_mode = InputMode.EDITING

    def end_input(self):
        # print(f'"{self.input_buffer.decode()}"')
        self.poke_bg(self.input_start_x + self.input_buffer_cursor,
                     self.input_start_y, self.bg_color)
        self.input_mode = InputMode.DONE
        self.print_cursor_x = 0
        self.print_cursor_y += 1
        if self.print_cursor_y == self.rows:
            self.scroll()

    def reset_input(self):
        self.input_mode = InputMode.READY

    def input_ready(self):
        return self.input_mode == InputMode.READY

    def input_editing(self):
        return self.input_mode == InputMode.EDITING

    def input_done(self):
        return self.input_mode == InputMode.DONE

    def spc(self, x):
        self.print_cursor_x += x

        while self.print_cursor_x >= self.columns:
            self.print_cursor_x -= self.columns
            self.print_cursor_y += 1

            if self.print_cursor_y == self.rows:
                self.scroll()

    def tab(self, x):
        self.print_cursor_x = x % self.columns

    def print(self, string):
        for c in string:
            if c == '\n':
                self.print_cursor_x = 0
                self.print_cursor_y += 1
                if self.print_cursor_y == self.rows:
                    self.scroll()
            else:
                address = (self.columns * self.print_cursor_y +
                           self.print_cursor_x) * 3
                self.char_memory[address] = ord(c)
                self.char_memory[address + 1] = self.fg_color
                self.char_memory[address + 2] = self.bg_color
                self.print_cursor_x += 1

                if self.print_cursor_x == self.columns:
                    self.print_cursor_x = 0
                    self.print_cursor_y += 1

                if self.print_cursor_y == self.rows:
                    self.scroll()

        self.update_chars()

    def poke_char(self, x, y, c):
        address = (self.columns * y + x) * 3
        self.char_memory[address] = c

    def poke_fg(self, x, y, fg):
        address = (self.columns * y + x) * 3
        self.char_memory[address + 1] = fg

    def poke_bg(self, x, y, bg):
        address = (self.columns * y + x) * 3
        self.char_memory[address + 2] = bg

    def peek_char(self, x, y):
        address = (self.columns * y + x) * 3
        return self.char_memory[address]

    def peek_fg(self, x, y):
        address = (self.columns * y + x) * 3
        return self.char_memory[address + 1]

    def peek_bg(self, x, y):
        address = (self.columns * y + x) * 3
        return self.char_memory[address + 2]


if __name__ == '__main__':
    window = Window(320, 240)

    window.cls()
    window.print("Retro BASIC v1.0\n")
    window.print("READY\n")

    pyglet.app.run()

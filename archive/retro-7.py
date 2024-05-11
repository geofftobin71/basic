#!/usr/bin/env python3
import random

import moderngl
import numpy as np
import pyglet
from pyglet.window import key


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

        self.vpu_fbo = self.ctx.framebuffer(self.screen_texture)
        self.vpu_fbo.viewport = (0, 0, self.width, self.height)

        self.char_memory = bytearray(self.columns * self.rows * 3)
        for i in range(len(self.char_memory) // 3):
            self.char_memory[i * 3 + 0] = i & 0xFF  # random.randint(0, 255)
            self.char_memory[i * 3 + 1] = self.char_memory[i * 3 + 0]
            self.char_memory[i * 3 + 2] = self.char_memory[i * 3 + 0]

        self.char_texture = self.ctx.texture((self.columns, self.rows),
                                             components=3,
                                             dtype="u1")
        self.char_texture.write(self.char_memory)

        self.font_bitmap = bytearray(
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x18\x18\x18\x18\x18\x00\x18\x00"
            b"\x36\x36\x36\x00\x00\x00\x00\x00\x36\x36\x7F\x36\x7F\x36\x36\x00"
            b"\x18\x7C\x06\x3C\x60\x3E\x18\x00\x00\x66\x35\x1B\x6C\x56\x33\x00"
            b"\x1C\x36\x36\x1C\xB6\x66\xDC\x00\x18\x18\x0C\x00\x00\x00\x00\x00"
            b"\x30\x18\x0C\x0C\x0C\x18\x30\x00\x0C\x18\x30\x30\x30\x18\x0C\x00"
            b"\x00\x66\x3C\xFF\x3C\x66\x00\x00\x00\x18\x18\x7E\x18\x18\x00\x00"
            b"\x00\x00\x00\x00\x00\x18\x18\x0C\x00\x00\x00\x7E\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x18\x18\x00\xC0\x60\x30\x18\x0C\x06\x03\x00"
            b"\x3C\x66\x76\x7E\x6E\x66\x3C\x00\x18\x1C\x18\x18\x18\x18\x3C\x00"
            b"\x3C\x66\x60\x30\x18\x0C\x7E\x00\x3C\x66\x60\x38\x60\x66\x3C\x00"
            b"\x38\x3C\x36\x33\x7F\x30\x30\x00\x7E\x06\x3E\x60\x60\x66\x3C\x00"
            b"\x38\x0C\x06\x3E\x66\x66\x3C\x00\x7E\x60\x30\x18\x18\x18\x18\x00"
            b"\x3C\x66\x66\x3C\x66\x66\x3C\x00\x3C\x66\x66\x7C\x60\x30\x1C\x00"
            b"\x00\x18\x18\x00\x00\x18\x18\x00\x00\x18\x18\x00\x00\x18\x18\x0C"
            b"\x30\x18\x0C\x06\x0C\x18\x30\x00\x00\x00\x7E\x00\x7E\x00\x00\x00"
            b"\x0C\x18\x30\x60\x30\x18\x0C\x00\x3C\x66\x60\x30\x18\x00\x18\x00"
            b"\x3C\x66\x76\x56\x76\x06\x3C\x00\x3C\x66\x66\x7E\x66\x66\x66\x00"
            b"\x3E\x66\x66\x3E\x66\x66\x3E\x00\x3C\x66\x06\x06\x06\x66\x3C\x00"
            b"\x1E\x36\x66\x66\x66\x36\x1E\x00\x7E\x06\x06\x1E\x06\x06\x7E\x00"
            b"\x7E\x06\x06\x1E\x06\x06\x06\x00\x3C\x66\x06\x76\x66\x66\x3C\x00"
            b"\x66\x66\x66\x7E\x66\x66\x66\x00\x3C\x18\x18\x18\x18\x18\x3C\x00"
            b"\x78\x30\x30\x30\x30\x36\x1C\x00\x66\x36\x1E\x0E\x1E\x36\x66\x00"
            b"\x06\x06\x06\x06\x06\x06\x7E\x00\x63\x77\x7F\x6B\x63\x63\x63\x00"
            b"\x66\x66\x6E\x7E\x76\x66\x66\x00\x3C\x66\x66\x66\x66\x66\x3C\x00"
            b"\x3E\x66\x66\x3E\x06\x06\x06\x00\x3C\x66\x66\x66\x76\x3E\x6C\x00"
            b"\x3E\x66\x66\x3E\x36\x66\x66\x00\x3C\x66\x06\x3C\x60\x66\x3C\x00"
            b"\x7E\x18\x18\x18\x18\x18\x18\x00\x66\x66\x66\x66\x66\x66\x3C\x00"
            b"\x66\x66\x66\x66\x66\x3C\x18\x00\x63\x63\x63\x6B\x7F\x77\x63\x00"
            b"\x66\x66\x3C\x18\x3C\x66\x66\x00\x66\x66\x66\x3C\x18\x18\x18\x00"
            b"\x7E\x60\x30\x18\x0C\x06\x7E\x00\x3C\x0C\x0C\x0C\x0C\x0C\x3C\x00"
            b"\x03\x06\x0C\x18\x30\x60\xC0\x00\x3C\x30\x30\x30\x30\x30\x3C\x00"
            b"\x18\x3C\x66\x42\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7F\x00"
            b"\x18\x18\x30\x00\x00\x00\x00\x00\x00\x00\x3C\x60\x7C\x66\x7C\x00"
            b"\x06\x06\x3E\x66\x66\x66\x3E\x00\x00\x00\x3C\x66\x06\x66\x3C\x00"
            b"\x60\x60\x7C\x66\x66\x66\x7C\x00\x00\x00\x3C\x66\x7E\x06\x3C\x00"
            b"\x38\x0C\x0C\x3E\x0C\x0C\x0C\x00\x00\x00\x7C\x66\x66\x7C\x60\x3C"
            b"\x06\x06\x3E\x66\x66\x66\x66\x00\x18\x00\x1C\x18\x18\x18\x3C\x00"
            b"\x18\x00\x1C\x18\x18\x18\x18\x0E\x06\x06\x66\x36\x1E\x36\x66\x00"
            b"\x1C\x18\x18\x18\x18\x18\x3C\x00\x00\x00\x36\x7F\x6B\x63\x63\x00"
            b"\x00\x00\x3E\x66\x66\x66\x66\x00\x00\x00\x3C\x66\x66\x66\x3C\x00"
            b"\x00\x00\x3E\x66\x66\x3E\x06\x06\x00\x00\x7C\x66\x66\x7C\x60\xE0"
            b"\x00\x00\x36\x6E\x06\x06\x06\x00\x00\x00\x7C\x06\x3C\x60\x3E\x00"
            b"\x0C\x0C\x3E\x0C\x0C\x0C\x38\x00\x00\x00\x66\x66\x66\x66\x7C\x00"
            b"\x00\x00\x66\x66\x66\x3C\x18\x00\x00\x00\x63\x63\x6B\x7F\x36\x00"
            b"\x00\x00\x66\x3C\x18\x3C\x66\x00\x00\x00\x66\x66\x66\x7C\x60\x3C"
            b"\x00\x00\x7E\x30\x18\x0C\x7E\x00\x30\x18\x18\x0C\x18\x18\x30\x00"
            b"\x18\x18\x18\x18\x18\x18\x18\x00\x0C\x18\x18\x30\x18\x18\x0C\x00"
            b"\x00\x6E\x3B\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        )

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

        self.vpu_prog = self.ctx.program(
            vertex_shader="""
            #version 330

            uniform vec2 screen_size;

            in vec2 position;
            in vec2 uv;

            out vec2 pixel;

            void main()
            {
                gl_Position = vec4(position, 0.0, 1.0);
                pixel = uv * screen_size;
            }
            """,
            fragment_shader="""
            #version 330

            uniform sampler2D font_sampler;
            uniform usampler2D char_sampler;

            in vec2 pixel;

            out vec4 color;

            void main()
            {
                uint cell = texelFetch(char_sampler, ivec2(pixel) >> 3, 0).r;

                uint cell_x = (cell & 0x0FU) << 3;
                uint cell_y = (cell & 0xF0U) >> 1;

                uint pixel_x = uint(pixel.x) & 0x07U;
                uint pixel_y = uint(pixel.y) & 0x07U;

                float c = texelFetch(font_sampler, ivec2(cell_x + pixel_x,
                                cell_y + pixel_y), 0).r;

                // float c = float(cell) / 255.0;

                vec4 background_color = vec4(0.1, 0.1, 0.1, 1.0);
                vec4 foreground_color = vec4(0.9, 0.9, 0.9, 1.0);

                color = mix(background_color, foreground_color, c);
            }
            """,
        )

        self.vpu_prog['font_sampler'].value = 1
        self.vpu_prog['char_sampler'].value = 2
        self.vpu_prog['screen_size'].value = (self.width, self.height)

        vpu_vertices = np.array([
            1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, -1.0, 1.0, 0.0, -1.0,
            -1.0, 0.0, 0.0
        ],
                                dtype='f4')
        self.vpu_vbo = self.ctx.buffer(vpu_vertices)
        self.vpu_vao = self.ctx.vertex_array(self.vpu_prog,
                                             self.vpu_vbo,
                                             "position",
                                             "uv",
                                             mode=moderngl.TRIANGLE_STRIP)

        self.pixel_upscale_prog = self.ctx.program(
            vertex_shader="""
#version 330

            uniform vec2 screen_size;

            in vec2 position;
            in vec2 uv;

            out vec2 pixel;

            void main()
            {
                gl_Position = vec4(position, 0.0, 1.0);
                pixel = uv * screen_size;
            }
            """,
            fragment_shader="""
#version 330

            uniform vec2 screen_size;
            uniform sampler2D screen_sampler;

            in vec2 pixel;

            out vec4 color;

            void main()
            {
                vec2 seam = floor(pixel + 0.5);
                vec2 dudv = fwidth(pixel);
                vec2 uv = (seam + clamp((pixel - seam) / dudv, -0.5, 0.5))
                            / screen_size;

                color = texture(screen_sampler, uv);
            }
            """,
        )

        self.pixel_upscale_prog['screen_sampler'].value = 3
        self.pixel_upscale_prog['screen_size'].value = (self.width,
                                                        self.height)

        self.screen_vbo = self.ctx.buffer(reserve=4 * 4 * 4)
        self.screen_vao = self.ctx.vertex_array(self.pixel_upscale_prog,
                                                self.screen_vbo,
                                                "position",
                                                "uv",
                                                mode=moderngl.TRIANGLE_STRIP)

    def resize(self, window_aspect):
        display_size = 0.9

        if window_aspect < self.aspect:  # Portrait Device
            screen_vertices = np.array([
                display_size, display_size * window_aspect / self.aspect, 1.0,
                0.0, -display_size, display_size * window_aspect / self.aspect,
                0.0, 0.0, display_size, -display_size * window_aspect /
                self.aspect, 1.0, 1.0, -display_size,
                -display_size * window_aspect / self.aspect, 0.0, 1.0
            ],
                                       dtype='f4')
        else:  # Landscape Device
            screen_vertices = np.array([
                display_size * self.aspect / window_aspect, display_size, 1.0,
                0.0, -display_size * self.aspect / window_aspect, display_size,
                0.0, 0.0, display_size * self.aspect / window_aspect,
                -display_size, 1.0, 1.0, -display_size * self.aspect /
                window_aspect, -display_size, 0.0, 1.0
            ],
                                       dtype='f4')

        self.screen_vbo.write(screen_vertices.tobytes())


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
        # self.viewport = (0, 0, *self.get_framebuffer_size())
        self.aspect = width / height
        self.dsp.resize(self.aspect)

    def on_draw(self):
        self.dsp.vpu_fbo.use()
        self.dsp.vpu_fbo.clear()  # clear bg color
        self.dsp.font_texture.use(1)
        self.dsp.char_texture.use(2)
        self.dsp.vpu_vao.render()
        self.ctx.screen.use()
        self.viewport = (0, 0, *self.get_framebuffer_size())
        self.ctx.clear()  # clear border color
        self.dsp.screen_texture.use(3)
        self.dsp.screen_vao.render()


if __name__ == '__main__':
    window = Window()
    pyglet.app.run()

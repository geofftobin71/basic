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

        self.font_bitmap = bytearray(256 * 8)
  \x00000000, \x00000000, \x00000000, \x00000000, \x00000000, \x00000000, \x00000000, \x00000000,
  \x00000000, \x00000000, \x00000000, \x00000000, \x00000000, \x00000000, \x00000000, \x00000000,
  \x00000000, \x00000000, \x00000000, \x00000000, \x00000000, \x00000000, \x00000000, \x00000000,
  \x00000000, \x00000000, \x00000000, \x00000000, \x00000000, \x00000000, \x00000000, \x00000000,
  \x00000000, \x00000000, \x00000000, \x00000000, \x00000000, \x00000000, \x00000000, \x00000000,
  \x00000000, \x00000000, \x00000000, \x00000000, \x00000000, \x00000000, \x00000000, \x00000000,
  \x00000000, \x00000000, \x00000000, \x00000000, \x00000000, \x00000000, \x00000000, \x00000000,
  \x00000000, \x00000000, \x00000000, \x00000000, \x00000000, \x00000000, \x00000000, \x00000000,

  \x00000000, \x00000000, \x18181818, \x00180018, \x00363636, \x00000000, \x367F3636, \x0036367F,   #  !"#
  \x3C067C18, \x00183E60, \x1B356600, \x0033566C, \x1C36361C, \x00DC66B6, \x000C1818, \x00000000,   # $%&'
  \x0C0C1830, \x0030180C, \x3030180C, \x000C1830, \xFF3C6600, \x0000663C, \x7E181800, \x00001818,   # ()*+
  \x00000000, \x0C181800, \x7E000000, \x00000000, \x00000000, \x00181800, \x183060C0, \x0003060C,   # ,-./
  \x7E76663C, \x003C666E, \x18181C18, \x003C1818, \x3060663C, \x007E0C18, \x3860663C, \x003C6660,   # 0123
  \x33363C38, \x0030307F, \x603E067E, \x003C6660, \x3E060C38, \x003C6666, \x1830607E, \x00181818,   # 4567
  \x3C66663C, \x003C6666, \x7C66663C, \x001C3060, \x00181800, \x00181800, \x00181800, \x0C181800,   # 89:;
  \x060C1830, \x0030180C, \x007E0000, \x0000007E, \x6030180C, \x000C1830, \x3060663C, \x00180018,   # <=>?

  \x5676663C, \x003C0676, \x7E66663C, \x00666666, \x3E66663E, \x003E6666, \x0606663C, \x003C6606,   # @ABC
  \x6666361E, \x001E3666, \x1E06067E, \x007E0606, \x1E06067E, \x00060606, \x7606663C, \x003C6666,   # DEFG
  \x7E666666, \x00666666, \x1818183C, \x003C1818, \x30303078, \x001C3630, \x0E1E3666, \x0066361E,   # HIJK
  \x06060606, \x007E0606, \x6B7F7763, \x00636363, \x7E6E6666, \x00666676, \x6666663C, \x003C6666,   # LMNO
  \x3E66663E, \x00060606, \x6666663C, \x006C3E76, \x3E66663E, \x00666636, \x3C06663C, \x003C6660,   # PQRS
  \x1818187E, \x00181818, \x66666666, \x003C6666, \x66666666, \x00183C66, \x6B636363, \x0063777F,   # TUVW
  \x183C6666, \x0066663C, \x3C666666, \x00181818, \x1830607E, \x007E060C, \x0C0C0C3C, \x003C0C0C,   # XYZ[
  \x180C0603, \x00C06030, \x3030303C, \x003C3030, \x42663C18, \x00000000, \x00000000, \x007F0000,   # \]^_

  \x00301818, \x00000000, \x603C0000, \x007C667C, \x663E0606, \x003E6666, \x663C0000, \x003C6606,   # `abc
  \x667C6060, \x007C6666, \x663C0000, \x003C067E, \x3E0C0C38, \x000C0C0C, \x667C0000, \x3C607C66,   # defg
  \x663E0606, \x00666666, \x181C0018, \x003C1818, \x181C0018, \x0E181818, \x36660606, \x0066361E,   # hijk
  \x1818181C, \x003C1818, \x7F360000, \x0063636B, \x663E0000, \x00666666, \x663C0000, \x003C6666,   # lmno
  \x663E0000, \x06063E66, \x667C0000, \xE0607C66, \x6E360000, \x00060606, \x067C0000, \x003E603C,   # pqrs
  \x0C3E0C0C, \x00380C0C, \x66660000, \x007C6666, \x66660000, \x00183C66, \x63630000, \x00367F6B,   # tuvw
  \x3C660000, \x00663C18, \x66660000, \x3C607C66, \x307E0000, \x007E0C18, \x0C181830, \x00301818,   # xyz{
  \x18181818, \x00181818, \x3018180C, \x000C1818, \x003B6E00, \x00000000, \x00000000, \x00000000    # |}~

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

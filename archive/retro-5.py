#!/usr/bin/env python3
import moderngl
import numpy as np
import pyglet
import random
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
            self.char_memory[i * 3 + 0] = random.randint(0, 255)
            self.char_memory[i * 3 + 1] = self.char_memory[i * 3 + 0]
            self.char_memory[i * 3 + 2] = self.char_memory[i * 3 + 0]

        self.char_texture = self.ctx.texture((self.columns, self.rows),
                                             components=3,
                                             dtype="i1")
        self.char_texture.write(self.char_memory)

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

                /*
                float c = texelFetch(font_sampler, ivec2(cell_x + pixel_x,
                                cell_y + pixel_y), 0).r;
                                */

                float c = float(cell) / 255.0;

                vec4 background_color = vec4(0.2, 0.2, 0.3, 1.0);
                vec4 foreground_color = vec4(0.8, 0.8, 0.9, 1.0);

                color = mix(background_color, foreground_color, c);
            }
            """,
        )

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

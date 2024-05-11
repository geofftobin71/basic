display_vs = """
#version 330

in vec2 position;

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

display_fs = """
#version 330

uniform sampler2D font_sampler;     // Font image. 16 x 16 grid of 8 x 8 pixel characters. Single channel.
uniform usampler2D char_sampler;    // Screen characters image. Columns x Rows. RGB. R = char, G = fg, B = bg.
uniform sampler2D colors_sampler;   // Color palette image. 256 x 1. RGB.

in vec4 gl_FragCoord;

out vec4 color;

void main()
{
    // Get the character at this screen pixel (x / 8, y / 8)
    uvec3 char = texelFetch(char_sampler, ivec2(gl_FragCoord.xy) >> 3, 0).rgb;

    // Get the tile of this character in the font image
    uvec2 tile = uvec2((char.r & 0x0FU) << 3, (char.r & 0xF0U) >> 1);

    // Get the pixel offset within the tile
    uvec2 offset = uvec2(uint(gl_FragCoord.x) & 0x07U, uint(gl_FragCoord.y) & 0x07U);

    // Get the value (on or off) at this pixel in the font image
    float value = texelFetch(font_sampler, ivec2(tile.x + offset.x, tile.y + offset.y), 0).r;

    // Get the colors from the palette
    vec4 foreground_color = texelFetch(colors_sampler, ivec2(char.g, 0), 0);
    vec4 background_color = texelFetch(colors_sampler, ivec2(char.b, 0), 0);

    color = mix(background_color, foreground_color, value);
}
"""

pixel_upscale_vs = """
#version 330

uniform vec2 display_size;

in vec2 position;
in vec2 uv;

out vec2 pixel;

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    pixel = uv * display_size;
}
"""

pixel_upscale_fs = """
#version 330

uniform vec2 display_size;
uniform sampler2D display_sampler;

in vec2 pixel;

out vec4 color;

void main()
{
    vec2 seam = floor(pixel + 0.5);
    vec2 dudv = fwidth(pixel);
    vec2 uv = (seam + clamp((pixel - seam) / dudv, -0.5, 0.5)) / display_size;

    color = texture(display_sampler, uv);
}
"""

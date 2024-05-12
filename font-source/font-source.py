#!/usr/bin/env python3
import argparse

from pyglet import image


def main():
    arg_parser = argparse.ArgumentParser(
        description="Convert font image to python bytecode")
    arg_parser.add_argument("filename", help="Source Image")
    arg_parser.add_argument("fontname", help="Font Name")
    arg_parser.add_argument(
        "-pw",
        "--pixel-width",
        type=int,
        help="Source image pixels per font pixel (default 2)")
    arg_parser.add_argument("-sx",
                            "--start-x",
                            type=int,
                            help="Source image X pixel offset (default 0)")
    arg_parser.add_argument("-sy",
                            "--start-y",
                            type=int,
                            help="Source image Y pixel offset (default 0)")
    arg_parser.add_argument(
        "-f",
        "--first-char",
        type=int,
        help="First ASCII character in the image (default 33)")
    arg_parser.add_argument(
        "-l",
        "--last-char",
        type=int,
        help="Last ASCII character in the image (default 127)")
    arg_parser.add_argument("-width",
                            "--width",
                            type=int,
                            help="Width in characters (default 32)")
    arg_parser.add_argument("-height",
                            "--height",
                            type=int,
                            help="Height in characters (default 3)")
    arg_parser.add_argument("-i",
                            "--inverse",
                            action="store_true",
                            help="Inverse (Dark text on light background)")
    arg_parser.add_argument("-cw",
                            "--char-width",
                            type=int,
                            help="Width of a single character (default 8)")

    args = arg_parser.parse_args()

    pic = image.load(args.filename)
    rawimage = pic.get_image_data()
    format = "I"
    pitch = rawimage.width * len(format)
    pixels = rawimage.get_data(format, -pitch)

    font_name = args.fontname
    pixel_width = args.pixel_width or 2
    start_x = args.start_x or 0
    start_y = args.start_y or 0
    first_char = args.first_char or 33
    last_char = args.last_char or 127
    width = args.width or 32
    height = args.height or 3
    char_width = args.char_width or 8

    result = bytearray()
    for _ in range(first_char * 8):
        result.append(0)

    char = first_char
    for cy in range(height):
        for cx in range(width):
            y = cy * pixel_width * 8 + start_y
            for py in range(8):
                value = 0
                x = cx * pixel_width * char_width + start_x
                for px in range(char_width):
                    address = pitch * y + x
                    if args.inverse:
                        if pixels[address] < 128:
                            value += 1 << px
                    else:
                        if pixels[address] > 128:
                            value += 1 << px
                    x += pixel_width
                result.append(int(value))
                y += pixel_width
            char += 1
            if char > last_char:
                break

    for _ in range((255 - last_char) * 8):
        result.append(0)

    print(f"{font_name} = {result}")


if __name__ == '__main__':
    main()

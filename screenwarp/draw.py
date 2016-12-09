import logging
import cStringIO

import wand.image
import wand.color
import wand.drawing

logger = logging.getLogger(__name__)


def drawChessboard(screen_width, screen_height, cols, rows, color1='#fff', color2='#000', x=0, y=0, square_width=None, square_height=None, bgcolor='#000'):
    if square_width is None: square_width = (screen_width-x) / cols
    if square_height is None: square_height = (screen_height-y) / rows

    logger.debug("drawChessboard %s x %s", cols, rows)
    image = wand.image.Image(width=screen_width, height=screen_height, background=wand.color.Color(bgcolor))
    with wand.drawing.Drawing() as draw:
        draw.fill_color = wand.color.Color(color2)
        draw.color = wand.color.Color(color2)
        for r in range(rows):
            for c in range(cols):
                if (c + r) % 2:
                    draw.fill_color = wand.color.Color(color2)
                    draw.color = wand.color.Color(color2)
                else:
                    draw.fill_color = wand.color.Color(color1)
                    draw.color = wand.color.Color(color1)
                x2 = x + square_width * c
                y2 = y + square_height * r

                #logger.debug("%s %s %s %s", x, y, square_width, square_height)
                draw.rectangle(x2, y2, width=square_width-1, height=square_height-1)

        draw.draw(image)
        blob = image.make_blob('png')
    return cStringIO.StringIO(blob)


def draw_points(points, width, height, xoff, yoff):
    image = wand.image.Image(width=width, height=height, background=wand.color.Color('#fff'))

    with wand.drawing.Drawing() as draw:
        draw.fill_color = wand.color.Color('#f00')
        for p in range(len(points)):
            # draw.fill_color = wand.color.Color('#{0:x}{0:x}{0:x}'.format(r*2))
            x = points[p][0] + xoff
            y = points[p][1] + yoff
            draw_point(draw, x, y)

        draw.draw(image)
        blob = image.make_blob('png')
    return cStringIO.StringIO(blob)


def draw_grid(grid, width, height, xoff, yoff):
    image = wand.image.Image(width=width, height=height, background=wand.color.Color('#fff'))

    with wand.drawing.Drawing() as draw:
        draw.fill_color = wand.color.Color('#f00')
        for r in range(len(grid)):
            # draw.fill_color = wand.color.Color('#{0:x}{0:x}{0:x}'.format(r*2))
            for c in range(len(grid[r])):
                #logger.info("r: %s, c: %s", r, c)
                x = grid[r][c][0] + xoff
                y = grid[r][c][1] + yoff
                draw_point(draw, x, y)

        draw.draw(image)
        blob = image.make_blob('png')
    return cStringIO.StringIO(blob)


def draw_point(draw, x, y):
    draw.point(x, y)
    draw.point(x+1, y)
    draw.point(x-1, y)
    draw.point(x, y+1)
    draw.point(x, y-1)


def draw_text(text, width, height):
    font_size = 400
    image = wand.image.Image(width=width, height=height, background=wand.color.Color('#fff'))
    with wand.drawing.Drawing() as draw:
        draw.fill_color = wand.color.Color('#000')
        draw.font_size = font_size
        draw.text_alignment = 'center'
        draw.text(width/2, height/2+font_size/2, text)

        draw.draw(image)
        blob = image.make_blob('png')
    return cStringIO.StringIO(blob)


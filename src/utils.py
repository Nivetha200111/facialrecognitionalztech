
import cv2

def load_image(file_path):
    """Load an image from a given file path."""
    return cv2.imread(file_path)

def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Resize an image to a given width or height while maintaining aspect ratio."""
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

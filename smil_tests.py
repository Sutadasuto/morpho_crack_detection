import cv2
import numpy as np
import smilPython as sp

test_images = ["/media/winbuntu/databases/my_photos/from_papers/001.png",
               "/media/winbuntu/databases/CrackForestDataset/image/107.jpg"]
path_sizes = [3, 5, 10, 20, 50, 100]

for idx, path in enumerate(test_images):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    original = sp.Image()
    original.fromNumArray(image.transpose())

    closed = sp.Image()
    opened = sp.Image()
    for size in path_sizes:
        sp.ImPathClosing(original, size, closed)
        sp.ImPathOpening(original, size, opened)
        sp.write(original, "/media/winbuntu/databases/my_photos/smil/%s.png" % idx)
        sp.write(closed, "/media/winbuntu/databases/my_photos/smil/%s_closing_%s.png" % (idx, size))
        sp.write(opened, "/media/winbuntu/databases/my_photos/smil/%s_opening_%s.png" % (idx, size))

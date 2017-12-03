import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

def make_graph_image(x, y):
    fig = plt.figure()
    plt.plot(x,y)
    fig.canvas.draw()

    w, h  = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(h,w,3)
    t = transforms.ToTensor()

    return t(buf)



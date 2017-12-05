import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

plt.switch_backend('agg')

def make_graph_image(x, y):
    fig = plt.figure()
    plt.plot(x,y)
    plt.xticks(x)
    plt.axis([x[0], x[-1], 0, 1])
    fig.canvas.draw()

    w, h  = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(h,w,3)
    t = transforms.ToTensor()

    return t(buf)



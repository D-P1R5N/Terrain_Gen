import numpy as np
import random
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.colors as colors

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def pick_a_number():
    _l = np.linspace(.4,.8,16)
    _v = random.choice(_l)
    return _v

def create_border(shape):
    x = random.randrange(2,np.shape(shape)[0])
    y = random.randrange(2,np.shape(shape)[1])
    #x,y= np.shape(shape)
    border = np.ones((x,y))
    delta = np.ones((x-2,y-2))
    delta = np.negative(delta)
    r,c = 1,1
    border[r:r + delta.shape[0], c:c +delta.shape[1]] += delta
    border = rectify_border(border)
    delta = rectify_delta(delta)
    border[r:r + delta.shape[0], c:c + delta.shape[1]] += delta
    return border

def rectify_border(border):
    out = np.where(border>0)
    coords = [i for i in zip(*out)]
    chosen = random.sample(coords,2*len(coords)//3)
    remainder = [i for i in coords if i not in chosen]
    for _r,_c in coords:
        if (_r,_c) in chosen:
            border[_r][_c] += pick_a_number()
        else:
            pass
        border[_r][_c] -=1
    return border

def rectify_delta(delta):
    delta += 1
    delta += (np.random.randint(0,35, size=np.shape(delta))/100)
    return delta

def evolve(world,feature):
    x = create_border(feature)
    x_max = np.shape(world)[0] - np.shape(x)[0]
    x_val = random.randrange(0, x_max)
    y_max = np.shape(world)[1] - np.shape(x)[1]
    y_val = random.randrange(0, y_max)
    world[x_val:x_val + np.shape(x)[0], y_val:y_val + np.shape(x)[1]] += x

    return world
def colour(world):
    plt.figure()

    top = plt.cm.terrain(np.linspace(.36,1,256))
    #mid = plt.cm.terrain(np.linspace(.36,.72,256))
    bot = plt.cm.terrain(np.linspace(0,.36,256))
    all = np.vstack([bot,top])

    spec = colors.LinearSegmentedColormap.from_list("Terra", all)
    divnorm = MidpointNormalize(vmin = np.min(world), vcenter= np.mean(world), vmax=np.amax(world))
    plt.imshow(world, alpha=.9, origin='lower', cmap=spec, interpolation='spline36', norm=divnorm)
    return plt.show()

r,c = 1,1
# World:Feature:Generation ~~ 5:1:20
map = []
world = np.zeros((35,35))
feature = np.zeros((7,7))
generation = 125
while generation:
    world + evolve(world,feature)
    map.append(world)
    generation -= 1
map = np.vstack(map)
np.save("map", map)
colour(world)

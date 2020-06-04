from assets import Sphere, Vision
import pyglet 

# w = pyglet.window.Window()

eye = Vision([1,0,0], size=(600,400))
sph = Sphere([7,3,2], 3)


eye.cast([sph])
eye.render()

# pyglet.app.run()
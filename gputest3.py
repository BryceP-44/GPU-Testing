import torch
import time
import numpy as np
import math
import vispy
from vispy import app, scene
from vispy.scene import visuals

print(torch.cuda.is_available())
nx, ny = 2000, 2500

def laplacian(T):
    return (torch.roll(T, 1, dims=0) + torch.roll(T, -1, dims=0) +
            torch.roll(T, 1, dims=1) + torch.roll(T, -1, dims=1) -
            4 * T)

x = torch.arange(0, nx).unsqueeze(1).repeat(1, ny)  # x-coordinates
y = torch.arange(0, ny).unsqueeze(0).repeat(nx, 1)  # y-coordinates
T = torch.zeros(nx, ny, device='cuda')  # Tensor on GPU
xplot = x.numpy().flatten()
yplot = y.numpy().flatten()


obs = torch.tensor([-10,-10,20], device='cuda')
tot = math.tan(120*math.pi/180)

gpu_memory_size = T.numel() * T.element_size()  
print("Memory size of T:", gpu_memory_size / (1024 ** 2),"MB")
print("Starting GPU calculation")
torch.cuda.empty_cache()

dt = .1
ds = 1
f = dt/(ds**2)

# Initialize temperature
T[:, 0] = T[:, -1] = torch.linspace(100, 0, nx)  # Gradient on sides
T[0, :] = 100  # Top edge hot
T[-1, :] = 0   # Bottom edge cold

xplot = x.cpu().numpy().flatten()
yplot = y.cpu().numpy().flatten()
Tplot = T.cpu().numpy().flatten()
pos = np.column_stack((xplot, yplot, Tplot))

canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
scatter = visuals.Markers()
scatter.set_data(np.empty((0, 3)))
view.add(scatter)
scatter.set_data(pos, edge_width=0, face_color=(1, 1, 1, .5), size=2)
view.camera = 'turntable'  # or try 'arcball'
#view.camera = scene.TurntableCamera(elevation=30, azimuth=45)

def update(event):
    global xplot
    global yplot
    global T
    ti = time.time()
    T += laplacian(T) * f
    T[:, 0] = T[:, -1] = 0#torch.linspace(100, 0, nx)  # Gradient on sides
    T[0,:] = 100
    T[-1,:] = 0
    torch.cuda.synchronize()
    print("GPU Laplacian time:", time.time() - ti)

    ti2=time.time()
    
    Tplot = T.cpu().numpy().flatten()
    pos = np.column_stack((xplot, yplot, Tplot))

    
    #if pos.size>0:
    scatter.set_data(pos, edge_width=0, face_color=(1, 1, 1, .5), size=2)
    #view.camera = 'turntable'  # or try 'arcball'
    
    print("Graphing time: ", time.time()-ti2)
    print("")

@canvas.events.key_press.connect
def on_key_press(event):
    if event.key == 'Q':
        app.quit()
        
timer = app.Timer(interval=0.01, connect=update, start=True)

if __name__ == '__main__':
    app.run()





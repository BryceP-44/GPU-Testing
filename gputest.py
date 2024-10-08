import torch
import time
import random
import numpy as np
import math
import vispy
from vispy import scene
import keyboard

print(torch.cuda.is_available())

nx, ny = 200, 250



def laplacian(T):
    return (torch.roll(T, 1, dims=0) + torch.roll(T, -1, dims=0) +
            torch.roll(T, 1, dims=1) + torch.roll(T, -1, dims=1) -
            4 * T)

x = torch.arange(0, nx, device='cuda').unsqueeze(1).repeat(1, ny)  # x-coordinates
y = torch.arange(0, ny, device='cuda').unsqueeze(0).repeat(nx, 1)  # y-coordinates
T = torch.zeros(nx, ny, device='cuda')  # Tensor on GPU

canvas = scene.SceneCanvas(keys="interactive", size=(800, 600), show=True)
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=(0, 0, nx, ny))
image = scene.Image(cmap='viridis', parent=view.scene)

obs=torch.tensor([-10,-10,20], device='cuda')
tot=math.tan(120*math.pi/180)

gpu_memory_size = T.numel() * T.element_size()  
print("Memory size of T:", gpu_memory_size / (1024 ** 2),"MB")
print("Starting GPU calculation")
torch.cuda.empty_cache()
#print(T)
cont=1
dt=.001
ds=1
f=dt/(ds**2)
speed=1000


def update(T,f,obs,x,y,tot):
    #torch.cuda.empty_cache()
    ti = time.time()
    T[0,:]=100
    T[nx-1,:]=0
    T += laplacian(T)*f
    torch.cuda.synchronize()
    print("GPU Laplacian time:", time.time() - ti)

    T_normalized = (T - T.min()) / (T.max() - T.min())
    image.set_data(T_normalized.cpu().numpy())
    
    xr=obs[0]-x #relative x distance to the observer
    yr=obs[1]-y
    zr=obs[2]-T
    dyz=(yr**2+zr**2+10**-6)**.5
    xp=1920*xr/(2*dyz)*tot+.5

    dxz=(xr**2+zr**2+10**-6)**.5
    yp=1080*yr/(2*dxz)*tot+.5
    #print("XP: ",xp)
    #print("YP: ",yp)
    positions = np.vstack((xp.cpu().numpy().flatten(), yp.cpu().numpy().flatten())).T

    if keyboard.is_pressed("up arrow"):
        obs[1]+=speed
    if keyboard.is_pressed("down arrow"):
        obs[1]-=speed
    if keyboard.is_pressed("left arrow"):
        obs[0]+=speed
    if keyboard.is_pressed("right arrow"):
        obs[0]-=speed
    if keyboard.is_pressed("w"):
        obs[2]+=speed
    if keyboard.is_pressed("s"):
        obs[2]-=speed


#view.camera = scene.PanZoomCamera(aspect=1)
#view.camera.set_range()

while cont==1:
    update(T,f,obs,x,y,tot)
    canvas.update()
    vispy.app.process_events()
    time.sleep(.01)
    if keyboard.is_pressed("q"):
        cont=0
        
print(T)














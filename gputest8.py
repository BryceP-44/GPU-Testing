import pygame
import moderngl
import numpy as np
import time

num_points =9000
L=2
ds=L/num_points
c=2
dt=.5*ds/c #cfl condition for stability

# Initialize positions and velocities
x_values = np.linspace(-1.0, 1.0, num_points)
# Initial shape (e.g., Gaussian pulse)
y_values = 0.5 * np.exp(-50.0 * x_values**2)
# Initial velocities (can be zero or set to create specific waves)
vy_values = np.zeros(num_points)

# Create buffers
line_vertices = np.column_stack((x_values, y_values)).astype('f4')
velocities = np.column_stack((x_values, vy_values)).astype('f4')



# Initialize Pygame
pygame.init()
width, height = 800, 600
pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)

# Create ModernGL context
ctx = moderngl.create_context(require=460)

# Set viewport to match the window dimensions
ctx.viewport = (0, 0, width, height)

# Vertex Shader with animation
vertex_shader_source = '''
#version 460
in vec2 in_vert;
//uniform float time;

void main() {
    /*
    float y_offset=.1;
    if (in_vert.x<-.5){
        y_offset=.5;
        }
    else{
        y_offset=.3;
        }*/
    
    // Apply the offset to the y-coordinate
    //vec2 pos = vec2(in_vert.x, in_vert.y + y_offset);
    //float in_vert.y=in_vert.y+.01;
    vec2 pos = vec2(in_vert.x, in_vert.y);
    
    gl_Position = vec4(pos, 0.0, 1.0);
}
'''

# Fragment Shader
fragment_shader_source = '''
#version 460
out vec4 fragColor;
void main() {
    fragColor = vec4(1.0, 0.0, 0.0, 1.0);  // Red color
}
'''

# Compute Shader for updating vertex positions
compute_shader_source = '''
#version 460
layout(local_size_x = 256) in; // Define workgroup size not 1024 anymore

layout(std430, binding = 0) buffer Data {
    vec2 positions[];  // Buffer of 2D positions (x, y)
};

layout(std430, binding = 1) buffer Velocities {
    vec2 velocities[];  // Buffer of 2D velocities (x, y)
};

shared vec2 shared_pos[256];
uniform float dt;
uniform float c;
uniform float ds;

void main() {
    uint idx = gl_GlobalInvocationID.x;  // Global index in buffer
    uint local_idx = gl_LocalInvocationID.x;
    
    // Check if the index is within the valid range
    if (idx < positions.length()) {
        shared_pos[local_idx]=positions[idx];  // Prevent out-of-bounds access
    }

    barrier(); //ensure all threads have loaded data

    
    if (idx>0 && idx<positions.length()-1){

    float d2y=(positions[idx-1].y+positions[idx+1].y-2*shared_pos[local_idx].y)/(ds*ds);
    velocities[idx].y+=c*c*d2y*dt;
    positions[idx].y+=velocities[idx].y*dt;
    
    }
    
    if (positions[idx].x > 1.0) {
        positions[idx].x = -1.0;  // Wrap-around logic for x position
    }
}
'''
# Create compute shader program
compute_prog = ctx.compute_shader(compute_shader_source)

# Compile shaders and create a program
render_prog = ctx.program(
    vertex_shader=vertex_shader_source,
    fragment_shader=fragment_shader_source,
)

vbo = ctx.buffer(line_vertices) #vertex buffer object
# Bind buffer to the compute shader (binding point 0)
vbo.bind_to_storage_buffer(0)

# Create a Vertex Array Object
vao = ctx.simple_vertex_array(render_prog, vbo, 'in_vert')

#create velocity buffer
velbo=ctx.buffer(velocities)
velbo.bind_to_storage_buffer(1)

# Set uniforms
compute_prog['c'].value = c
compute_prog['ds'].value = ds
compute_prog['dt'].value = dt

# Main loop

running=True
while running:
    ti=time.time()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running=False
    
    compute_prog.run(group_x=(num_points+255)//256)

    # Clear the screen (black background)
    ctx.clear()
    
    # Draw the line
    vao.render(moderngl.LINE_STRIP)
    
    # Swap the buffers
    pygame.display.flip()

    if time.time()-ti>0:
        #print(1/(time.time()-ti))
        g=4



pygame.quit()



    

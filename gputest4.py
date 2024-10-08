import pygame
import moderngl
import numpy as np

# Initialize Pygame
pygame.init()
width, height = 800, 600
pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)

# Create ModernGL context
ctx = moderngl.create_context()

# Set viewport to match the window dimensions
ctx.viewport = (0, 0, width, height)

# Vertex Shader with animation
vertex_shader_source = '''
#version 330
in vec2 in_vert;
uniform float time;

void main() {
    // Calculate a sine wave based on time and x-position
    float y_offset = sin(in_vert.x * 10.0 + time*10) * 1;
    
    // Apply the offset to the y-coordinate
    vec2 pos = vec2(in_vert.x, in_vert.y + y_offset);
    
    gl_Position = vec4(pos, 0.0, 1.0);
}
'''

# Fragment Shader
fragment_shader_source = '''
#version 330
out vec4 fragColor;
void main() {
    fragColor = vec4(1.0, 0.0, 0.0, 1.0);  // Red color
}
'''

# Compile shaders and create a program
prog = ctx.program(
    vertex_shader=vertex_shader_source,
    fragment_shader=fragment_shader_source,
)

# Create a vertex buffer for a line with more points
num_points = 100000
x_values = np.linspace(-1.0, 1.0, num_points)
line_vertices = np.column_stack((x_values, np.zeros(num_points))).astype('f4')

vbo = ctx.buffer(line_vertices)

# Create a Vertex Array Object
vao = ctx.simple_vertex_array(prog, vbo, 'in_vert')

# Get the uniform location for time
time_loc = prog['time']

# Main loop
clock = pygame.time.Clock()
start_time = pygame.time.get_ticks()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    # Calculate time
    current_time = (pygame.time.get_ticks() - start_time) / 1000.0

    # Clear the screen (black background)
    ctx.clear()

    # Update the time uniform
    time_loc.value = current_time

    # Draw the line
    vao.render(moderngl.LINE_STRIP)

    # Swap the buffers
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

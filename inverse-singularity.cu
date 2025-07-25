// Compile with:
// nvcc -O3 -o inverse-singularity inverse-singularity.cu -lGL -lGLU -lglut

#include <GL/glut.h>
#include <curand_kernel.h>

const int WIDTH = 1920, HEIGHT = 1080;
const int NUM_PARTICLES = 9999999;
const float SCHWARZSCHILD_R = 1.0f;

struct Particle {
    float r, theta, v;
    float color[3];
};

__global__ void update_particles(Particle* particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_PARTICLES) return;
    particles[i].r *= particles[i].v;
}

__global__ void init_particles(Particle* particles, curandState* states, int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_PARTICLES) return;

    curand_init(seed, i, 0, &states[i]);
    float theta = curand_uniform(&states[i]) * 2.0f * 3.1415926f;
    float v = 1.001f + curand_uniform(&states[i]) * 0.01f;
    particles[i] = {
        SCHWARZSCHILD_R, theta, v,
        { curand_uniform(&states[i]),
          curand_uniform(&states[i]),
          curand_uniform(&states[i]) }
    };
}

Particle* d_particles;
curandState* d_states;

void polar_to_cartesian_host(float r, float theta, float& x, float& y) {
    x = r * cosf(theta);
    y = r * sinf(theta);
}

void display() {
    update_particles<<<(NUM_PARTICLES + 255) / 256, 256>>>(d_particles);
    cudaDeviceSynchronize();

    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POINTS);

    static Particle* particles = new Particle[NUM_PARTICLES];
    cudaMemcpy(particles, d_particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);

    for (int i = 0; i < NUM_PARTICLES; ++i) {
        float x, y;
        polar_to_cartesian_host(particles[i].r, particles[i].theta, x, y);
        x = x / (WIDTH / 2);
        y = y / (HEIGHT / 2);
        glColor3fv(particles[i].color);
        glVertex2f(x, y);
    }

    glEnd();
    glutSwapBuffers();
}

void idle() {
    glutPostRedisplay();
}

void init_cuda() {
    cudaMalloc(&d_particles, NUM_PARTICLES * sizeof(Particle));
    cudaMalloc(&d_states, NUM_PARTICLES * sizeof(curandState));
    init_particles<<<(NUM_PARTICLES + 255) / 256, 256>>>(d_particles, d_states, time(0));
    cudaDeviceSynchronize();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Inverse Schwarzschild");
    glClearColor(0, 0, 0, 0);
    gluOrtho2D(-1, 1, -1, 1);
    glPointSize(2.0);

    init_cuda();
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutMainLoop();
    return 0;
}

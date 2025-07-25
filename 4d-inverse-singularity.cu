// nvcc -O3 -o 4d-inverse-singularity 4d-inverse-singularity.cu -lGL -lGLU -lglut

#include <GL/glut.h>
#include <curand_kernel.h>
#include <cmath>
#include <ctime>

const int WIDTH = 1920, HEIGHT = 1080;
const int NUM_PARTICLES = 999999;
const float SCHWARZSCHILD_R = 5.0f;
const float PI = 3.1415926f;
const float W_PROJECTION_DIST = 20.0f;

struct Particle {
    float r, theta, phi, psi, v;
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
    float theta = curand_uniform(&states[i]) * 2 * PI;
    float phi = curand_uniform(&states[i]) * PI;
    float psi = curand_uniform(&states[i]) * PI;
    float v = 1.002f + curand_uniform(&states[i]) * 0.003f;

    particles[i] = {
        SCHWARZSCHILD_R, theta, phi, psi, v,
        { curand_uniform(&states[i]),
          curand_uniform(&states[i]),
          curand_uniform(&states[i]) }
    };
}

Particle* d_particles;
curandState* d_states;
static Particle* h_particles = new Particle[NUM_PARTICLES];

// Camera
float camX = 0, camY = 0, camZ = 50;
float pitch = 0, yaw = -90;
float lastX = WIDTH / 2, lastY = HEIGHT / 2;
bool keys[256] = {0};

void process_input() {
    float speed = 0.5;
    float dx = cosf(yaw * PI / 180.0f), dz = sinf(yaw * PI / 180.0f);
    float dy = sinf(pitch * PI / 180.0f);

    if (keys['w']) { camX += dx * speed; camZ += dz * speed; camY += dy * speed; }
    if (keys['s']) { camX -= dx * speed; camZ -= dz * speed; camY -= dy * speed; }
    if (keys['a']) { camX += dz * speed; camZ -= dx * speed; }
    if (keys['d']) { camX -= dz * speed; camZ += dx * speed; }
}

void display() {
    process_input();
    update_particles<<<(NUM_PARTICLES + 255) / 256, 256>>>(d_particles);
    cudaMemcpy(h_particles, d_particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    float frontX = cosf(pitch * PI / 180.0f) * cosf(yaw * PI / 180.0f);
    float frontY = sinf(pitch * PI / 180.0f);
    float frontZ = cosf(pitch * PI / 180.0f) * sinf(yaw * PI / 180.0f);
    gluLookAt(camX, camY, camZ, camX + frontX, camY + frontY, camZ + frontZ, 0, 1, 0);

    glBegin(GL_POINTS);
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        Particle& p = h_particles[i];

        // 4D → 3D projection
        float x4 = p.r * sinf(p.psi) * sinf(p.phi) * cosf(p.theta);
        float y4 = p.r * sinf(p.psi) * cosf(p.phi);
        float z4 = p.r * sinf(p.psi) * sinf(p.phi) * sinf(p.theta);
        float w4 = p.r * cosf(p.psi);

        float denom = (W_PROJECTION_DIST - w4);
        if (denom <= 0.01f) continue;

        float x = x4 / denom;
        float y = y4 / denom;
        float z = z4 / denom;

        glColor3fv(p.color);
        glVertex3f(x, y, z);
    }
    glEnd();

    glutSwapBuffers();
}

void idle() {
    glutPostRedisplay();
}

void keyDown(unsigned char key, int, int) {
    keys[key] = true;
}
void keyUp(unsigned char key, int, int) {
    keys[key] = false;
}

void mouse_motion(int x, int y) {
    float sensitivity = 0.15f;
    float dx = x - lastX;
    float dy = lastY - y;
    lastX = x;
    lastY = y;
    yaw += dx * sensitivity;
    pitch += dy * sensitivity;
    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;
}

void init_cuda() {
    cudaMalloc(&d_particles, NUM_PARTICLES * sizeof(Particle));
    cudaMalloc(&d_states, NUM_PARTICLES * sizeof(curandState));
    init_particles<<<(NUM_PARTICLES + 255) / 256, 256>>>(d_particles, d_states, time(0));
    cudaDeviceSynchronize();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Inverse Schwarzschild 4D");

    glEnable(GL_DEPTH_TEST);
    glClearColor(0, 0, 0, 1);
    glPointSize(1.0f);

    glMatrixMode(GL_PROJECTION);
    gluPerspective(75, (float)WIDTH / HEIGHT, 0.1f, 1000.0f);
    glMatrixMode(GL_MODELVIEW);

    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyDown);
    glutKeyboardUpFunc(keyUp);
    glutPassiveMotionFunc(mouse_motion);

    init_cuda();
    glutMainLoop();
    return 0;
}

// Compile with:
// nvcc -O3 -o schwarzschild-3d-inversed schwarzschild-3d-inversed.cu -lGL -lGLU -lglut

#include <GL/glut.h>
#include <curand_kernel.h>
#include <cmath>
#include <ctime>

const int WIDTH = 1000, HEIGHT = 1000;
const int NUM_PARTICLES = 300000;
const float GM = 5.0f;
const float DT = 0.02f;

struct Particle {
    float r, theta, phi;
    float dr, dtheta, dphi;
    float color;
};

Particle* d_particles;
Particle* h_particles = new Particle[NUM_PARTICLES];
curandState* d_states;

// Camera
float rotX = 20.0f, rotY = 0.0f, zoom = -250.0f;
int lastX, lastY;
bool dragging = false;

__device__ float3 spherical_to_cartesian(float r, float theta, float phi) {
    return make_float3(
        r * sinf(phi) * cosf(theta),
        r * cosf(phi),
        r * sinf(phi) * sinf(theta)
    );
}

__global__ void init_particles(Particle* p, curandState* states, int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_PARTICLES) return;
    curand_init(seed, i, 0, &states[i]);

    float r0 = 2.0f + curand_uniform(&states[i]) * 10.0f;
    float theta = curand_uniform(&states[i]) * 2 * M_PI;
    float phi = curand_uniform(&states[i]) * M_PI;

    p[i] = {
        r0, theta, phi,
        0.0f, 0.002f, 0.002f,
        1.0f
    };
}

__global__ void update_geodesics(Particle* p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_PARTICLES) return;

    float r = p[i].r;
    float f = 1.0f + 2.0f * GM / r;
    float acc = -GM / (r * r);

    p[i].dr += acc * DT;
    p[i].r += p[i].dr * f * DT;
    p[i].theta += p[i].dtheta;
    p[i].phi += p[i].dphi;

    if (p[i].r > 100.0f) p[i].color = 0.2f;
}

void display() {
    update_geodesics<<<(NUM_PARTICLES + 255) / 256, 256>>>(d_particles);
    cudaMemcpy(h_particles, d_particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glTranslatef(0, 0, zoom);
    glRotatef(rotX, 1, 0, 0);
    glRotatef(rotY, 0, 1, 0);

    glBegin(GL_POINTS);
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        Particle& p = h_particles[i];
        float3 pos = make_float3(
            p.r * sinf(p.phi) * cosf(p.theta),
            p.r * cosf(p.phi),
            p.r * sinf(p.phi) * sinf(p.theta)
        );
        glColor3f(p.color, p.color, p.color);
        glVertex3f(pos.x, pos.y, pos.z);
    }
    glEnd();
    glutSwapBuffers();
}

void idle() { glutPostRedisplay(); }

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        dragging = (state == GLUT_DOWN);
        lastX = x;
        lastY = y;
    }
}

void motion(int x, int y) {
    if (dragging) {
        rotY += (x - lastX);
        rotX += (y - lastY);
        lastX = x;
        lastY = y;
    }
}

void keys(unsigned char key, int, int) {
    if (key == 'w') zoom += 5.0f;
    if (key == 's') zoom -= 5.0f;
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
    glutCreateWindow("Inverted Schwarzschild 3D - CUDA");

    glEnable(GL_DEPTH_TEST);
    glClearColor(0, 0, 0, 1);
    glPointSize(1.0f);

    glMatrixMode(GL_PROJECTION);
    gluPerspective(60.0, (float)WIDTH / HEIGHT, 0.1, 1000.0);
    glMatrixMode(GL_MODELVIEW);

    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(keys);

    init_cuda();
    glutMainLoop();
    return 0;
}

// Compile with:
// nvcc -O3 -o schwarzschild-3d-inversed schwarzschild-3d-inversed.cu -lGL -lGLU -lglut

#include <GL/glut.h>
#include <curand_kernel.h>
#include <cmath>
#include <ctime>

const int WIDTH = 1920, HEIGHT = 1080;
const int NUM_PARTICLES = 9999999;
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
float camX = 0, camY = 0, camZ = 200;
float camYaw = 0, camPitch = 0;
bool keys[256] = { false };
int lastX, lastY;
bool dragging = false;

// Host version of spherical to cartesian
float3 spherical_to_cartesian_host(float r, float theta, float phi) {
    return {
        r * sinf(phi) * cosf(theta),
        r * cosf(phi),
        r * sinf(phi) * sinf(theta)
    };
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
    float acc = -GM / (r * r);  // repulsive

    p[i].dr += acc * DT;
    p[i].r += p[i].dr * f * DT;
    p[i].theta += p[i].dtheta;
    p[i].phi += p[i].dphi;

    if (p[i].r > 100.0f) p[i].color = 0.2f;
}

void update_camera() {
    float speed = 2.0f;
    float yawRad = camYaw * M_PI / 180.0f;
    float pitchRad = camPitch * M_PI / 180.0f;

    float dx = cosf(yawRad);
    float dz = sinf(yawRad);

    if (keys['w']) { camX += dx * speed; camZ += dz * speed; }
    if (keys['s']) { camX -= dx * speed; camZ -= dz * speed; }
    if (keys['a']) { camX += dz * speed; camZ -= dx * speed; }
    if (keys['d']) { camX -= dz * speed; camZ += dx * speed; }
    if (keys['q']) { camY -= speed; }
    if (keys['e']) { camY += speed; }
}

void display() {
    update_camera();
    update_geodesics<<<(NUM_PARTICLES + 255) / 256, 256>>>(d_particles);
    cudaMemcpy(h_particles, d_particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    float yawRad = camYaw * M_PI / 180.0f;
    float pitchRad = camPitch * M_PI / 180.0f;
    float lx = cosf(pitchRad) * cosf(yawRad);
    float ly = sinf(pitchRad);
    float lz = cosf(pitchRad) * sinf(yawRad);

    gluLookAt(camX, camY, camZ, camX + lx, camY + ly, camZ + lz, 0, 1, 0);

    glBegin(GL_POINTS);
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        Particle& p = h_particles[i];
        float3 pos = spherical_to_cartesian_host(p.r, p.theta, p.phi);
        glColor3f(p.color, p.color, p.color);
        glVertex3f(pos.x, pos.y, pos.z);
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

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        dragging = (state == GLUT_DOWN);
        lastX = x;
        lastY = y;
    }
}

void motion(int x, int y) {
    if (dragging) {
        camYaw += (x - lastX) * 0.3f;
        camPitch -= (y - lastY) * 0.3f;
        if (camPitch > 89.0f) camPitch = 89.0f;
        if (camPitch < -89.0f) camPitch = -89.0f;
        lastX = x;
        lastY = y;
    }
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
    glutCreateWindow("Inverted Schwarzschild 3D - Free Camera");

    glEnable(GL_DEPTH_TEST);
    glClearColor(0, 0, 0, 1);
    glPointSize(1.0f);

    glMatrixMode(GL_PROJECTION);
    gluPerspective(60.0, (float)WIDTH / HEIGHT, 0.1, 2000.0);
    glMatrixMode(GL_MODELVIEW);

    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyDown);
    glutKeyboardUpFunc(keyUp);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);

    init_cuda();
    glutMainLoop();
    return 0;
}

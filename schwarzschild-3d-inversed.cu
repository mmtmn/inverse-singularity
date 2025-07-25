// Compile with:
// nvcc -O3 -o schwarzschild_3d_inverted_colored schwarzschild_3d_inverted_colored.cu -lGL -lGLU -lglut

#include <GL/glut.h>
#include <curand_kernel.h>
#include <cmath>
#include <ctime>

const int WIDTH = 2560, HEIGHT = 1440;
const int NUM_PARTICLES = 999999;
const float GM = 5.0f;
const float DT = 0.02f;

struct Particle {
    float r, theta, phi;
    float dr, dtheta, dphi;
    float color_r, color_g, color_b;
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
bool slowMode = false;

float baseSpeed = 1.0f;

// 3D inverse Schwarzschild projection
float3 inverted_cartesian(float r, float theta, float phi) {
    float inv_r = 1.0f / (r + 0.001f);
    float inv_sin = 1.0f / (sinf(theta) + 0.01f);
    float inv_phi = 1.0f / (sinf(phi) + 0.01f);
    return {
        inv_r * inv_phi * cosf(theta),
        inv_r * cosf(phi),
        inv_r * inv_phi * sinf(theta)
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
        0.0f, -0.002f, -0.002f,
        curand_uniform(&states[i]),  // R
        curand_uniform(&states[i]),  // G
        curand_uniform(&states[i])   // B
    };
}

__global__ void update_inverted(Particle* p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_PARTICLES) return;

    float r = p[i].r;
    float f = -(1.0f + 2.0f * GM / r);
    float acc = +GM / (r * r);

    p[i].dr += acc * DT;
    p[i].r += p[i].dr * f * DT;
    p[i].theta += p[i].dtheta;
    p[i].phi += p[i].dphi;

    // mutate colors slightly
    p[i].color_r = fmodf(p[i].color_r + 0.0001f * p[i].dr, 1.0f);
    p[i].color_g = fmodf(p[i].color_g + 0.0002f * p[i].phi, 1.0f);
    p[i].color_b = fmodf(p[i].color_b + 0.0001f * p[i].r, 1.0f);
}

void update_camera() {
    float speed = slowMode ? baseSpeed * 0.1f : baseSpeed;
    float yawRad = camYaw * M_PI / 180.0f;
    float pitchRad = camPitch * M_PI / 180.0f;

    // Forward vector
    float fx = cosf(pitchRad) * cosf(yawRad);
    float fy = sinf(pitchRad);
    float fz = cosf(pitchRad) * sinf(yawRad);

    // Right vector (cross with global up)
    float rx = -sinf(yawRad);
    float ry = 0;
    float rz = cosf(yawRad);

    // Up vector (recomputed)
    float ux = ry * fz - rz * fy;
    float uy = rz * fx - rx * fz;
    float uz = rx * fy - ry * fx;

    if (keys['w']) { camX += fx * speed; camY += fy * speed; camZ += fz * speed; }
    if (keys['s']) { camX -= fx * speed; camY -= fy * speed; camZ -= fz * speed; }
    if (keys['a']) { camX -= rx * speed; camY -= ry * speed; camZ -= rz * speed; }
    if (keys['d']) { camX += rx * speed; camY += ry * speed; camZ += rz * speed; }
    if (keys['q']) { camX -= ux * speed; camY -= uy * speed; camZ -= uz * speed; }
    if (keys['e']) { camX += ux * speed; camY += uy * speed; camZ += uz * speed; }
}


void display() {
    update_camera();
    update_inverted<<<(NUM_PARTICLES + 255) / 256, 256>>>(d_particles);
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
        float3 pos = inverted_cartesian(p.r, p.theta, p.phi);
        glColor3f(p.color_r, p.color_g, p.color_b);
        glVertex3f(pos.x * 100.0f, pos.y * 100.0f, pos.z * 100.0f);
    }
    glEnd();

    glutSwapBuffers();
}

void idle() { glutPostRedisplay(); }

void keyDown(unsigned char key, int, int) {
    if (key == 32) slowMode = true;  // spacebar
    else keys[key] = true;
}

void keyUp(unsigned char key, int, int) {
    if (key == 32) slowMode = false;
    else keys[key] = false;
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
    glutCreateWindow("Inverted Schwarzschild 3D - Colored + Fly Cam");

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

// Compile with:
// nvcc -O3 -o inverted inverted.cu -lGL -lGLU -lglut

#include <GL/glut.h>
#include <curand_kernel.h>
#include <cmath>
#include <ctime>

const int WIDTH = 1000, HEIGHT = 1000;
const int NUM_PARTICLES = 200000;
const float GM = 5.0f;
const float DT = 0.02f;

struct Particle {
    float r, t;
    float dr, dtau;
    float color;
};

Particle* d_particles;
Particle* h_particles = new Particle[NUM_PARTICLES];
curandState* d_states;

// Camera state
float camX = 0.0f, camY = 0.0f;
float zoom = 1.0f;
bool keys[256] = { false };
int lastMouseX = 0, lastMouseY = 0;
bool dragging = false;

__global__ void init_particles(Particle* p, curandState* states, int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_PARTICLES) return;
    curand_init(seed, i, 0, &states[i]);

    float r0 = 2.0f + curand_uniform(&states[i]) * 20.0f;
    p[i] = {
        r0, 0.0f, 0.0f, 1.0f, 1.0f
    };
}

__global__ void update_inverted_geodesics(Particle* p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_PARTICLES) return;

    float r = p[i].r;
    float f = 1.0f + 2.0f * GM / r;
    float finv = 1.0f / f;
    float acc = -GM / (r * r);

    p[i].dr += acc * DT;
    p[i].r += p[i].dr * DT;
    p[i].t += p[i].dtau * finv * DT;

    if (p[i].r > 200.0f) p[i].color = 0.2f;
}

void handle_camera() {
    float speed = 10.0f / zoom;
    if (keys['w']) camY += speed;
    if (keys['s']) camY -= speed;
    if (keys['a']) camX -= speed;
    if (keys['d']) camX += speed;
    if (keys['q']) zoom *= 1.05f;
    if (keys['e']) zoom *= 0.95f;
}

void display() {
    handle_camera();
    update_inverted_geodesics<<<(NUM_PARTICLES + 255) / 256, 256>>>(d_particles);
    cudaMemcpy(h_particles, d_particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);

    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
    glTranslatef(WIDTH / 2.0f, HEIGHT / 2.0f, 0.0f);
    glScalef(zoom, zoom, 1.0f);
    glTranslatef(-camX, -camY, 0.0f);

    glBegin(GL_POINTS);
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        float r = h_particles[i].r;
        float t = h_particles[i].t;
        float c = h_particles[i].color;
        float x = r * 5.0f;
        float y = t * 2.0f;
        glColor3f(c, c, c);
        glVertex2f(x, y);
    }
    glEnd();

    glutSwapBuffers();
}

void idle() { glutPostRedisplay(); }

void keyDown(unsigned char key, int, int) { keys[key] = true; }
void keyUp(unsigned char key, int, int) { keys[key] = false; }

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        dragging = (state == GLUT_DOWN);
        lastMouseX = x;
        lastMouseY = y;
    }
}

void motion(int x, int y) {
    if (dragging) {
        float dx = (x - lastMouseX) / zoom;
        float dy = (y - lastMouseY) / zoom;
        camX -= dx;
        camY += dy;
        lastMouseX = x;
        lastMouseY = y;
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
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("Inverted Schwarzschild Geodesics - Full Camera");

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, WIDTH, 0, HEIGHT);
    glMatrixMode(GL_MODELVIEW);
    glPointSize(1.0f);
    glClearColor(0, 0, 0, 1);

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

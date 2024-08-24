// simple cuda kernel to add together vectors a and b, storing the result in c
// to run:
// nvcc add.cu && ./a.out

// for printing
#include <stdio.h>

// addition kernel
__global__ void add_vectors(float (*a)[], float (*b)[], float (*c)[], int n) {
    // get the current thread index
    // the kernel is run in parallel on a bunch of threads
    // from the perspective of a single thread
    int id = threadIdx.x;
    // basic sanity checks, make sure kernel is being called and values are correct
    printf("\nHello World from thread %d", id);
    printf("\nValue of a at thread: %f", (*a)[id]);

    // what's happening:
    // each thread adds one value
    // instead of [0.1, 0.2, 0.3, 0.4, 0.5] + [0.1, 0.2, 0.3, 0.4, 0.5] -> result [...],
    // thread 0 sees 0.1 + 0.1 = 0.2 -> c[0] = 0.2
    // thread 1 sees 0.2 + 0.2 = 0.4 -> c[1] = 0.4 and so on

    // check that the current thread index is within the length of the vectors
    if (id < n) {
        // if it is, write a[id] + b[id] into c[id]
        (*c)[id] = (*a)[id] + (*b)[id];
    }
    // log the output for debugging
    printf("\nOutput for thread: %f", (*c)[id]);

    // note that nothing is retuned, it is simply written to memory and later copied
}

int main() {
    printf("Program started");
    // define length of vectors
    // must be a const because compiler doesn't like variable length arrays
    const int n = 5;
    // size of vectors in memory (bytes)
    const int size = n * sizeof(float);
    // make some random vectors to test with, r_c is the output
    float r_a[n] = {0.1, 0.2, 0.3, 0.4, 0.5};
    float r_b[n] = {0.1, 0.2, 0.3, 0.4, 0.5};
    float r_c[n] = {0.0};
    // https://www.codeproject.com/Articles/7042/How-to-interpret-complex-C-C-declarations
    // make some empty pointer variables before malloc-ing on host and device, assigning, and memcpying
    float (*h_a)[n];
    float (*d_a)[n];

    float (*h_b)[n];
    float (*d_b)[n];

    float (*h_c)[n];
    float (*d_c)[n];
    // verify the size of memory we are allocating
    printf("\nSize: %d", size);
    printf("\nSize of r_a (bytes): %lu", sizeof(r_a));

    // https://stevengong.co/notes/CUDA-Memory-Allocation
    // allocate on device
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // allocate on host
    cudaMallocHost(&h_a, size);
    cudaMallocHost(&h_b, size);
    cudaMallocHost(&h_c, size);
    // assign variables (just pointers, will be read inside of kernel)
    h_a = &r_a;
    h_b = &r_b;
    h_c = &r_c;
    // make sure nothing has gone wrong after cuda malloc on the host
    printf("\nFirst element of r_a after cuda malloc host: %f", r_a[0]);
    printf("\nFirst element of h_a after cuda malloc host: %f", *h_a[0]);
    // copy h_a, h_b, and h_c to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, size, cudaMemcpyHostToDevice);
    // make sure everything made it there safely
    printf("\nFirst element of r_a after cuda memcpy: %f", r_a[0]);
    printf("\nFirst element of h_a after cuda memcpy: %f", *h_a[0]);
    // call the actual kernel
    // done with an extra thread to check if going out of boundaries works
    add_vectors<<<1,n+1>>>(d_a, d_b, d_c, n);
    // copy arrays back to the host so they can be read and printed from main()
    cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // print all items of c and make sure they are correct
    printf("\nAll entries of c:\n");
    for (int i=0; i<n; i++) {
        printf("%f", (*h_c)[i]);
        if (abs((r_a[i] + r_b[i]) - (*h_c)[i]) < 1e-4) {
            printf(" -- Correct");
        } else {
            printf(" -- Incorrect, should have been %f", r_a[i] + r_b[i]);
        }
        printf("\n");
    }

    // free memory (don't use regular free here or it will give invalid address)
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
__kernel void matrix_multiplication(__global const float* A, __global const float* B, __global float* C, const int A_n,
                                    const int A_m, const int B_n, const int B_m) {
  int row = get_global_id(0);
  int col = get_global_id(1);
  float sum = 0.0f;
  for (int k = 0; k < A_m; ++k) {
    sum += A[row * A_m + k] * B[k * B_m + col];
  }
  C[row * B_m + col] = sum;
}
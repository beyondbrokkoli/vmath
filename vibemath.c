#include <immintrin.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ========================================================================
// FAST AVX2 TRIGONOMETRY (Minimax Approximations)
// ========================================================================

// 1. Wraps any angle into the [-PI, PI] range so our polynomial works
static inline __m256 wrap_pi_avx(__m256 x) {
    __m256 inv_two_pi = _mm256_set1_ps(1.0f / (2.0f * M_PI));
    __m256 two_pi = _mm256_set1_ps(2.0f * M_PI);
    // q = round(x / 2PI)
    __m256 q = _mm256_round_ps(_mm256_mul_ps(x, inv_two_pi), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    // return x - q * 2PI
    return _mm256_fnmadd_ps(q, two_pi, x);
}

// 2. High-speed Sine approximation for 8 floats simultaneously
static inline __m256 fast_sin_avx(__m256 x) {
    x = wrap_pi_avx(x); // Keep it within bounds

    // Bhaskara I / Minimax base polynomial: sin(x) ~ (4/pi)*x - (4/pi^2)*x*|x|
    __m256 B = _mm256_set1_ps(4.0f / M_PI);
    __m256 C = _mm256_set1_ps(-4.0f / (M_PI * M_PI));

    // bitwise absolute value (clears the sign bit)
    __m256 x_abs = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
    __m256 y = _mm256_fmadd_ps(_mm256_mul_ps(C, x_abs), x, _mm256_mul_ps(B, x));

    // Extra precision refinement step
    __m256 P = _mm256_set1_ps(0.225f);
    __m256 y_abs = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), y);
    return _mm256_fmadd_ps(_mm256_fmadd_ps(y_abs, y, _mm256_sub_ps(_mm256_setzero_ps(), y)), P, y);
}

// 3. Cosine is just Sine shifted by PI/2
static inline __m256 fast_cos_avx(__m256 x) {
    __m256 half_pi = _mm256_set1_ps(M_PI / 2.0f);
    return fast_sin_avx(_mm256_add_ps(x, half_pi));
}

// ========================================================================
// 4D DEMOSCENE NOISE (Trigonometric fBM)
// ========================================================================
// Pure ALU noise generation. Zero memory lookups. Blasts 8 coordinates at once.
static inline __m256 fast_trig_noise_avx(__m256 nx, __m256 ny, __m256 nz, __m256 time) {
    // OCTAVE 1: Low Frequency, High Amplitude (The base tectonic shifts)
    __m256 v1 = fast_sin_avx(_mm256_add_ps(_mm256_mul_ps(nx, _mm256_set1_ps(3.1f)), time));
    __m256 v2 = fast_cos_avx(_mm256_add_ps(_mm256_mul_ps(ny, _mm256_set1_ps(2.8f)), time));
    __m256 v3 = fast_sin_avx(_mm256_add_ps(_mm256_mul_ps(nz, _mm256_set1_ps(3.4f)), time));
    __m256 out = _mm256_add_ps(v1, _mm256_add_ps(v2, v3)); // Range ~[-3.0, 3.0]

    // OCTAVE 2: High Frequency, Low Amplitude (The boiling surface details)
    __m256 time2 = _mm256_mul_ps(time, _mm256_set1_ps(1.8f)); // Moves faster
    __m256 v4 = fast_sin_avx(_mm256_add_ps(_mm256_mul_ps(nx, _mm256_set1_ps(7.2f)), time2));
    __m256 v5 = fast_cos_avx(_mm256_add_ps(_mm256_mul_ps(ny, _mm256_set1_ps(6.5f)), time2));
    __m256 v6 = fast_sin_avx(_mm256_add_ps(_mm256_mul_ps(nz, _mm256_set1_ps(8.1f)), time2));
    __m256 oct2 = _mm256_mul_ps(_mm256_add_ps(v4, _mm256_add_ps(v5, v6)), _mm256_set1_ps(0.35f));

    out = _mm256_add_ps(out, oct2);

    // Normalize back to roughly [-1.0, 1.0]
    return _mm256_mul_ps(out, _mm256_set1_ps(0.25f));
}

// ========================================================================
// CORE STRUCTS
// ========================================================================
typedef struct {
    float x, y, z;
    float yaw, pitch;
    float fov;
    float fwx, fwy, fwz;
    float rtx, rty, rtz;
    float upx, upy, upz;
} CameraState;

typedef struct {
    float *Obj_X, *Obj_Y, *Obj_Z, *Obj_Radius;
    float *Obj_FWX, *Obj_FWY, *Obj_FWZ;
    float *Obj_RTX, *Obj_RTY, *Obj_RTZ;
    float *Obj_UPX, *Obj_UPY, *Obj_UPZ;
    int *Obj_VertStart, *Obj_VertCount;
    int *Obj_TriStart, *Obj_TriCount;

    float *Vert_LX, *Vert_LY, *Vert_LZ;
    float *Vert_PX, *Vert_PY, *Vert_PZ; bool *Vert_Valid;

    int *Tri_V1, *Tri_V2, *Tri_V3;
    uint32_t *Tri_BakedColor, *Tri_ShadedColor; bool *Tri_Valid;

    float *Swarm_PX, *Swarm_PY, *Swarm_PZ;
    float *Swarm_VX, *Swarm_VY, *Swarm_VZ;
    float *Swarm_Seed;
    int Swarm_State;
    float Swarm_GravityBlend;
    float Swarm_MetalBlend;
    float Swarm_ParadoxBlend;
    bool Swarm_Explode1;
    bool Swarm_Explode2;
} RenderMemory;


// ========================================================================
// MEMORY & PROJECTION (The Core Pipeline)
// ========================================================================

EXPORT void vmath_clear_buffers(
    uint32_t* screen,
    float* zbuffer,
    uint32_t clear_color,
    float clear_z,
    int pixel_count
) {
    // 1. Pack 8 identical colors and 8 identical Z-values into AVX registers
    __m256i v_color = _mm256_set1_epi32(clear_color);
    __m256 v_z = _mm256_set1_ps(clear_z);

    int i = 0;
    // 2. The AVX Loop (Blindly overwrite 8 pixels at a time)
    for (; i <= pixel_count - 8; i += 8) {
        // Cast the screen pointer to a 256-bit integer pointer and fire!
        _mm256_storeu_si256((__m256i*)&screen[i], v_color);
        // Fire the Z-buffer floats!
        _mm256_storeu_ps(&zbuffer[i], v_z);
    }

    // 3. The Tail Loop (For any leftover pixels if screen isn't perfectly divisible by 8)
    for (; i < pixel_count; i++) {
        screen[i] = clear_color;
        zbuffer[i] = clear_z;
    }
}

EXPORT void vmath_project_vertices(
    int count,
    // Inputs (Local Coords)
    float* lx, float* ly, float* lz,
    // Outputs (Screen Coords & Validity)
    float* px, float* py, float* pz, bool* valid,

    // Object Matrix
    float ox, float oy, float oz,
    float rx, float ry, float rz, float ux, float uy, float uz, float fx, float fy, float fz,

    // Camera Matrix & Screen Info
    float cpx, float cpy, float cpz,
    float cfw_x, float cfw_y, float cfw_z,
    float crt_x, float crt_z,
    float cup_x, float cup_y, float cup_z,
    float cam_fov, float half_w, float half_h
) {
    // 1. Broadcast EVERYTHING outside the loop (Saves massive register pressure)
    __m256 v_ox = _mm256_set1_ps(ox); __m256 v_oy = _mm256_set1_ps(oy); __m256 v_oz = _mm256_set1_ps(oz);
    __m256 v_rx = _mm256_set1_ps(rx); __m256 v_ry = _mm256_set1_ps(ry); __m256 v_rz = _mm256_set1_ps(rz);
    __m256 v_ux = _mm256_set1_ps(ux); __m256 v_uy = _mm256_set1_ps(uy); __m256 v_uz = _mm256_set1_ps(uz);
    __m256 v_fx = _mm256_set1_ps(fx); __m256 v_fy = _mm256_set1_ps(fy); __m256 v_fz = _mm256_set1_ps(fz);

    __m256 v_cpx = _mm256_set1_ps(cpx); __m256 v_cpy = _mm256_set1_ps(cpy); __m256 v_cpz = _mm256_set1_ps(cpz);
    __m256 v_cfwx = _mm256_set1_ps(cfw_x); __m256 v_cfwy = _mm256_set1_ps(cfw_y); __m256 v_cfwz = _mm256_set1_ps(cfw_z);
    __m256 v_crtx = _mm256_set1_ps(crt_x); __m256 v_crtz = _mm256_set1_ps(crt_z);
    __m256 v_cupx = _mm256_set1_ps(cup_x); __m256 v_cupy = _mm256_set1_ps(cup_y); __m256 v_cupz = _mm256_set1_ps(cup_z);
    __m256 v_cam_fov = _mm256_set1_ps(cam_fov);
    __m256 v_half_w = _mm256_set1_ps(half_w); __m256 v_half_h = _mm256_set1_ps(half_h);
    __m256 v_two = _mm256_set1_ps(2.0f); // Needed for Newton-Raphson

    int i = 0;

    for (; i <= count - 8; i += 8) {
        __m256 v_lx = _mm256_loadu_ps(&lx[i]);
        __m256 v_ly = _mm256_loadu_ps(&ly[i]);
        __m256 v_lz = _mm256_loadu_ps(&lz[i]);

        __m256 v_wx = _mm256_fmadd_ps(v_lz, v_fx, _mm256_fmadd_ps(v_ly, v_ux, _mm256_fmadd_ps(v_lx, v_rx, v_ox)));
        __m256 v_wy = _mm256_fmadd_ps(v_lz, v_fy, _mm256_fmadd_ps(v_ly, v_uy, _mm256_fmadd_ps(v_lx, v_ry, v_oy)));
        __m256 v_wz = _mm256_fmadd_ps(v_lz, v_fz, _mm256_fmadd_ps(v_ly, v_uz, _mm256_fmadd_ps(v_lx, v_rz, v_oz)));

        __m256 v_vdx = _mm256_sub_ps(v_wx, v_cpx);
        __m256 v_vdy = _mm256_sub_ps(v_wy, v_cpy);
        __m256 v_vdz = _mm256_sub_ps(v_wz, v_cpz);

        __m256 v_cz = _mm256_fmadd_ps(v_vdz, v_cfwz, _mm256_fmadd_ps(v_vdy, v_cfwy, _mm256_mul_ps(v_vdx, v_cfwx)));

        __m256 v_mask = _mm256_cmp_ps(v_cz, _mm256_set1_ps(0.1f), _CMP_GE_OQ);
        int bitmask = _mm256_movemask_ps(v_mask);

        // --- NEWTON-RAPHSON FAST DIVISION ---
        // 1. Get fast hardware approximation of 1.0 / cz (~11 bits precision)
        __m256 v_rcp = _mm256_rcp_ps(v_cz);
        // 2. Refine precision using Newton-Raphson: rcp = rcp * (2.0 - cz * rcp) (~22 bits precision)
        // _mm256_fnmadd_ps does -(cz * rcp) + 2.0
        __m256 v_rcp_refined = _mm256_mul_ps(v_rcp, _mm256_fnmadd_ps(v_cz, v_rcp, v_two));
        // 3. Multiply fov by the refined reciprocal
        __m256 v_f = _mm256_mul_ps(v_cam_fov, v_rcp_refined);

        __m256 v_px = _mm256_add_ps(v_half_w, _mm256_mul_ps(v_f, _mm256_add_ps(_mm256_mul_ps(v_vdx, v_crtx), _mm256_mul_ps(v_vdz, v_crtz))));
        __m256 v_py = _mm256_add_ps(v_half_h, _mm256_mul_ps(v_f, _mm256_fmadd_ps(v_vdz, v_cupz, _mm256_fmadd_ps(v_vdy, v_cupy, _mm256_mul_ps(v_vdx, v_cupx)))));

        _mm256_storeu_ps(&px[i], v_px);
        _mm256_storeu_ps(&py[i], v_py);
        _mm256_storeu_ps(&pz[i], _mm256_mul_ps(v_cz, _mm256_set1_ps(1.004f)));

        valid[i+0] = (bitmask & (1 << 0)) != 0;
        valid[i+1] = (bitmask & (1 << 1)) != 0;
        valid[i+2] = (bitmask & (1 << 2)) != 0;
        valid[i+3] = (bitmask & (1 << 3)) != 0;
        valid[i+4] = (bitmask & (1 << 4)) != 0;
        valid[i+5] = (bitmask & (1 << 5)) != 0;
        valid[i+6] = (bitmask & (1 << 6)) != 0;
        valid[i+7] = (bitmask & (1 << 7)) != 0;
    }

    // Tail loop remains unchanged
    for (; i < count; i++) {
        float temp_wx = ox + lx[i]*rx + ly[i]*ux + lz[i]*fx;
        float temp_wy = oy + lx[i]*ry + ly[i]*uy + lz[i]*fy;
        float temp_wz = oz + lx[i]*rz + ly[i]*uz + lz[i]*fz;
        float vdx = temp_wx - cpx;
        float vdy = temp_wy - cpy;
        float vdz = temp_wz - cpz;
        float cz = vdx*cfw_x + vdy*cfw_y + vdz*cfw_z;

        if (cz < 0.1f) {
            valid[i] = false;
        } else {
            float f = cam_fov / cz; // Scalar division is fine here
            px[i] = half_w + (vdx*crt_x + vdz*crt_z) * f;
            py[i] = half_h + (vdx*cup_x + vdy*cup_y + vdz*cup_z) * f;
            pz[i] = cz * 1.004f;
            valid[i] = true;
        }
    }
}

EXPORT void vmath_process_triangles(
    int tCount,
    int* v1, int* v2, int* v3, bool* vert_valid,
    float* px, float* py, float* pz,
    float* lx, float* ly, float* lz,
    uint32_t* baked_color, uint32_t* shaded_color, bool* tri_valid,
    float rx, float ry, float rz,
    float ux, float uy, float uz,
    float fx, float fy, float fz,
    float sun_x, float sun_y, float sun_z
) {
    // --- PRE-LOAD UNIFORMS ---
    __m256 v_sun_x = _mm256_set1_ps(sun_x);
    __m256 v_sun_y = _mm256_set1_ps(sun_y);
    __m256 v_sun_z = _mm256_set1_ps(sun_z);

    __m256 v_rx = _mm256_set1_ps(rx), v_ry = _mm256_set1_ps(ry), v_rz = _mm256_set1_ps(rz);
    __m256 v_ux = _mm256_set1_ps(ux), v_uy = _mm256_set1_ps(uy), v_uz = _mm256_set1_ps(uz);
    __m256 v_fx = _mm256_set1_ps(fx), v_fy = _mm256_set1_ps(fy), v_fz = _mm256_set1_ps(fz);

    __m256 v_0_2 = _mm256_set1_ps(0.2f);
    __m256 v_1_0 = _mm256_set1_ps(1.0f);
    __m256i v_alpha_mask = _mm256_set1_epi32(0xFF000000);

    int i = 0;

    // ========================================================
    // 8-WIDE AVX2 GATHER PIPELINE
    // ========================================================
    for (; i <= tCount - 8; i += 8) {
        // 1. Gather 8 Triangle Indices
        __m256i vi1 = _mm256_loadu_si256((__m256i*)&v1[i]);
        __m256i vi2 = _mm256_loadu_si256((__m256i*)&v2[i]);
        __m256i vi3 = _mm256_loadu_si256((__m256i*)&v3[i]);

        // 2. Vertex Validity Check (Build 32-bit vector mask safely)
        int valids[8];
        for(int j=0; j<8; j++) {
            valids[j] = (vert_valid[v1[i+j]] && vert_valid[v2[i+j]] && vert_valid[v3[i+j]]) ? -1 : 0;
        }
        __m256i v_vert_valid = _mm256_loadu_si256((__m256i*)valids);

        // If NO vertices are valid in this batch of 8, skip entirely
        if (_mm256_testz_si256(v_vert_valid, v_vert_valid)) {
            for(int j=0; j<8; j++) tri_valid[i+j] = false;
            continue;
        }

        // 3. Gather 2D Screen Coords (The Magic Instruction)
        __m256 px1 = _mm256_i32gather_ps(px, vi1, 4);
        __m256 py1 = _mm256_i32gather_ps(py, vi1, 4);
        __m256 px2 = _mm256_i32gather_ps(px, vi2, 4);
        __m256 py2 = _mm256_i32gather_ps(py, vi2, 4);
        __m256 px3 = _mm256_i32gather_ps(px, vi3, 4);
        __m256 py3 = _mm256_i32gather_ps(py, vi3, 4);

        // 4. Backface Culling (Cross Product < 0)
        __m256 dx1 = _mm256_sub_ps(px2, px1);
        __m256 dy1 = _mm256_sub_ps(py2, py1);
        __m256 dx2 = _mm256_sub_ps(px3, px1);
        __m256 dy2 = _mm256_sub_ps(py3, py1);
        __m256 cross = _mm256_sub_ps(_mm256_mul_ps(dx1, dy2), _mm256_mul_ps(dy1, dx2));

        __m256 cross_mask = _mm256_cmp_ps(cross, _mm256_setzero_ps(), _CMP_LT_OQ);
        __m256i v_survive = _mm256_and_si256(v_vert_valid, _mm256_castps_si256(cross_mask));

        // Create an 8-bit mask of surviving triangles
        int survive_mask = _mm256_movemask_ps(_mm256_castsi256_ps(v_survive));

        // If ALL 8 triangles are facing backward, SKIP LIGHTING!
        if (survive_mask == 0) {
            for(int j=0; j<8; j++) tri_valid[i+j] = false;
            continue;
        }

        // 5. Gather Local 3D Coords for survivors
        __m256 lx1 = _mm256_i32gather_ps(lx, vi1, 4);
        __m256 ly1 = _mm256_i32gather_ps(ly, vi1, 4);
        __m256 lz1 = _mm256_i32gather_ps(lz, vi1, 4);
        __m256 lx2 = _mm256_i32gather_ps(lx, vi2, 4);
        __m256 ly2 = _mm256_i32gather_ps(ly, vi2, 4);
        __m256 lz2 = _mm256_i32gather_ps(lz, vi2, 4);
        __m256 lx3 = _mm256_i32gather_ps(lx, vi3, 4);
        __m256 ly3 = _mm256_i32gather_ps(ly, vi3, 4);
        __m256 lz3 = _mm256_i32gather_ps(lz, vi3, 4);

        // 6. Calculate Local Normal
        __m256 ax = _mm256_sub_ps(lx2, lx1);
        __m256 ay = _mm256_sub_ps(ly2, ly1);
        __m256 az = _mm256_sub_ps(lz2, lz1);
        __m256 bx = _mm256_sub_ps(lx3, lx1);
        __m256 by = _mm256_sub_ps(ly3, ly1);
        __m256 bz = _mm256_sub_ps(lz3, lz1);

        __m256 lnx = _mm256_sub_ps(_mm256_mul_ps(ay, bz), _mm256_mul_ps(az, by));
        __m256 lny = _mm256_sub_ps(_mm256_mul_ps(az, bx), _mm256_mul_ps(ax, bz));
        __m256 lnz = _mm256_sub_ps(_mm256_mul_ps(ax, by), _mm256_mul_ps(ay, bx));

        // 7. Transform to World Normal
        __m256 wnx = _mm256_fmadd_ps(lnz, v_fx, _mm256_fmadd_ps(lny, v_ux, _mm256_mul_ps(lnx, v_rx)));
        __m256 wny = _mm256_fmadd_ps(lnz, v_fy, _mm256_fmadd_ps(lny, v_uy, _mm256_mul_ps(lnx, v_ry)));
        __m256 wnz = _mm256_fmadd_ps(lnz, v_fz, _mm256_fmadd_ps(lny, v_uz, _mm256_mul_ps(lnx, v_rz)));

        // 8. FAST INVERSE SQUARE ROOT
        __m256 len_sq = _mm256_fmadd_ps(wnz, wnz, _mm256_fmadd_ps(wny, wny, _mm256_mul_ps(wnx, wnx)));
        len_sq = _mm256_max_ps(len_sq, _mm256_set1_ps(0.000001f)); // Prevent Inf
        __m256 inv_len = _mm256_rsqrt_ps(len_sq);

        wnx = _mm256_mul_ps(wnx, inv_len);
        wny = _mm256_mul_ps(wny, inv_len);
        wnz = _mm256_mul_ps(wnz, inv_len);

        // 9. Lambertian Lighting (Dot Product)
        __m256 dot = _mm256_fmadd_ps(wnz, v_sun_z, _mm256_fmadd_ps(wny, v_sun_y, _mm256_mul_ps(wnx, v_sun_x)));
        __m256 light = _mm256_max_ps(v_0_2, _mm256_min_ps(dot, v_1_0)); // Clamp 0.2 -> 1.0

        // 10. Color Decompression & Multiplication
        __m256i orig_col = _mm256_loadu_si256((__m256i*)&baked_color[i]);

        __m256i r_i = _mm256_and_si256(orig_col, _mm256_set1_epi32(0xFF));
        __m256i g_i = _mm256_and_si256(_mm256_srli_epi32(orig_col, 8), _mm256_set1_epi32(0xFF));
        __m256i b_i = _mm256_and_si256(_mm256_srli_epi32(orig_col, 16), _mm256_set1_epi32(0xFF));

        __m256 r_f = _mm256_mul_ps(_mm256_cvtepi32_ps(r_i), light);
        __m256 g_f = _mm256_mul_ps(_mm256_cvtepi32_ps(g_i), light);
        __m256 b_f = _mm256_mul_ps(_mm256_cvtepi32_ps(b_i), light);

        r_i = _mm256_cvtps_epi32(r_f);
        g_i = _mm256_cvtps_epi32(g_f);
        b_i = _mm256_cvtps_epi32(b_f);

        // Repack into uint32_t ARGB format
        __m256i final_col = _mm256_or_si256(v_alpha_mask,
                            _mm256_or_si256(_mm256_slli_epi32(b_i, 16),
                            _mm256_or_si256(_mm256_slli_epi32(g_i, 8), r_i)));

        // 11. Masked Store Output
        // ONLY write colors for triangles that survived!
        _mm256_maskstore_epi32((int*)&shaded_color[i], v_survive, final_col);

        // Output boolean valid flags
        for(int j=0; j<8; j++) {
            tri_valid[i+j] = (survive_mask & (1 << j)) != 0;
        }
    }

    // ========================================================
    // SCALAR TAIL LOOP (For Safety)
    // ========================================================
    for (; i < tCount; i++) {
        int i1 = v1[i], i2 = v2[i], i3 = v3[i];

        if (!vert_valid[i1] || !vert_valid[i2] || !vert_valid[i3]) {
            tri_valid[i] = false;
            continue;
        }

        float px1 = px[i1], py1 = py[i1];
        float px2 = px[i2], py2 = py[i2];
        float px3 = px[i3], py3 = py[i3];

        float cross = (px2 - px1) * (py3 - py1) - (py2 - py1) * (px3 - px1);
        if (cross >= 0) { tri_valid[i] = false; continue; }

        uint32_t orig_col = baked_color[i];
        float ax = lx[i2] - lx[i1], ay = ly[i2] - ly[i1], az = lz[i2] - lz[i1];
        float bx = lx[i3] - lx[i1], by = ly[i3] - ly[i1], bz = lz[i3] - lz[i1];

        float lnx = ay * bz - az * by, lny = az * bx - ax * bz, lnz = ax * by - ay * bx;
        float wnx = lnx * rx + lny * ux + lnz * fx;
        float wny = lnx * ry + lny * uy + lnz * fy;
        float wnz = lnx * rz + lny * uz + lnz * fz;

        float inv_len = 1.0f / sqrtf(wnx*wnx + wny*wny + wnz*wnz + 0.000001f);
        wnx *= inv_len; wny *= inv_len; wnz *= inv_len;

        float dot = wnx * sun_x + wny * sun_y + wnz * sun_z;
        float light = dot < 0.2f ? 0.2f : (dot > 1.0f ? 1.0f : dot);

        uint32_t b = (uint32_t)(((orig_col >> 16) & 0xFF) * light);
        uint32_t g = (uint32_t)(((orig_col >> 8) & 0xFF) * light);
        uint32_t r = (uint32_t)((orig_col & 0xFF) * light);

        shaded_color[i] = 0xFF000000 | (b << 16) | (g << 8) | r;
        tri_valid[i] = true;
    }
}

EXPORT void vmath_rasterize_triangles(
    int tCount,
    int* v1, int* v2, int* v3, bool* tri_valid,
    float* px, float* py, float* pz,
    uint32_t* shaded_color,
    uint32_t* screen_buffer, float* z_buffer,
    int canvas_w, int canvas_h
) {
    for (int i = 0; i < tCount; i++) {
        if (!tri_valid[i]) continue;

        int i1 = v1[i], i2 = v2[i], i3 = v3[i];
        float x1 = px[i1], y1 = py[i1], z1 = pz[i1];
        float x2 = px[i2], y2 = py[i2], z2 = pz[i2];
        float x3 = px[i3], y3 = py[i3], z3 = pz[i3];

        // Broadcast color to 8-wide integer register
        __m256i v_color = _mm256_set1_epi32((int)shaded_color[i]);

        if (y1 > y2) { float t=x1; x1=x2; x2=t;  t=y1; y1=y2; y2=t;  t=z1; z1=z2; z2=t; }
        if (y1 > y3) { float t=x1; x1=x3; x3=t;  t=y1; y1=y3; y3=t;  t=z1; z1=z3; z3=t; }
        if (y2 > y3) { float t=x2; x2=x3; x3=t;  t=y2; y2=y3; y3=t;  t=z2; z2=z3; z3=t; }

        float total_height = y3 - y1;
        if (total_height <= 0.0f) continue;

        float inv_total = 1.0f / total_height;
        int y_start = (int)fmaxf(0.0f, ceilf(y1));
        int y_end   = (int)fminf((float)(canvas_h - 1), floorf(y3));

        // ==========================================
        // UPPER TRIANGLE
        // ==========================================
        float dy_upper = y2 - y1;
        if (dy_upper > 0.0f) {
            float inv_upper = 1.0f / dy_upper;
            int limit_y = (int)fminf((float)y_end, floorf(y2));

            for (int y = y_start; y <= limit_y; y++) {
                float t_total = (y - y1) * inv_total;
                float t_half  = (y - y1) * inv_upper;
                float ax = x1 + (x3 - x1) * t_total, az = z1 + (z3 - z1) * t_total;
                float bx = x1 + (x2 - x1) * t_half,  bz = z1 + (z2 - z1) * t_half;

                if (ax > bx) { float t=ax; ax=bx; bx=t;  t=az; az=bz; bz=t; }

                float row_width = bx - ax;
                if (row_width > 0.0f) {
                    float z_step = (bz - az) / row_width;
                    int start_x = (int)fmaxf(0.0f, ceilf(ax));
                    int end_x   = (int)fminf((float)(canvas_w - 1), floorf(bx));
                    float current_z = az + z_step * (start_x - ax);

                    int off = y * canvas_w;
                    int x = start_x;

                    // --- THE AVX2 HORIZONTAL LOOP ---
                    __m256 v_z_step8 = _mm256_set1_ps(z_step * 8.0f);
                    __m256 v_current_z = _mm256_set_ps(
                        current_z + z_step*7.0f, current_z + z_step*6.0f,
                        current_z + z_step*5.0f, current_z + z_step*4.0f,
                        current_z + z_step*3.0f, current_z + z_step*2.0f,
                        current_z + z_step*1.0f, current_z
                    );

                    for (; x <= end_x - 7; x += 8) {
                        __m256 v_old_z = _mm256_loadu_ps(&z_buffer[off + x]);
                        __m256 v_cmp = _mm256_cmp_ps(v_current_z, v_old_z, _CMP_LT_OQ);
                        __m256i v_mask = _mm256_castps_si256(v_cmp);

                        _mm256_maskstore_ps(&z_buffer[off + x], v_mask, v_current_z);
                        _mm256_maskstore_epi32((int*)&screen_buffer[off + x], v_mask, v_color);

                        v_current_z = _mm256_add_ps(v_current_z, v_z_step8);
                    }

                    // --- SCALAR TAIL LOOP ---
                    current_z = az + z_step * (x - ax); // Recalculate scalar Z exactly
                    for (; x <= end_x; x++) {
                        if (current_z < z_buffer[off + x]) {
                            z_buffer[off + x] = current_z;
                            screen_buffer[off + x] = (uint32_t)shaded_color[i];
                        }
                        current_z += z_step;
                    }
                }
            }
        }

        // ==========================================
        // LOWER TRIANGLE
        // ==========================================
        float dy_lower = y3 - y2;
        if (dy_lower > 0.0f) {
            float inv_lower = 1.0f / dy_lower;
            int start_y = (int)fmaxf((float)y_start, ceilf(y2));

            for (int y = start_y; y <= y_end; y++) {
                float t_total = (y - y1) * inv_total;
                float t_half  = (y - y2) * inv_lower;
                float ax = x1 + (x3 - x1) * t_total, az = z1 + (z3 - z1) * t_total;
                float bx = x2 + (x3 - x2) * t_half,  bz = z2 + (z3 - z2) * t_half;

                if (ax > bx) { float t=ax; ax=bx; bx=t;  t=az; az=bz; bz=t; }

                float row_width = bx - ax;
                if (row_width > 0.0f) {
                    float z_step = (bz - az) / row_width;
                    int start_x = (int)fmaxf(0.0f, ceilf(ax));
                    int end_x   = (int)fminf((float)(canvas_w - 1), floorf(bx));
                    float current_z = az + z_step * (start_x - ax);

                    int off = y * canvas_w;
                    int x = start_x;

                    // --- THE AVX2 HORIZONTAL LOOP ---
                    __m256 v_z_step8 = _mm256_set1_ps(z_step * 8.0f);
                    __m256 v_current_z = _mm256_set_ps(
                        current_z + z_step*7.0f, current_z + z_step*6.0f,
                        current_z + z_step*5.0f, current_z + z_step*4.0f,
                        current_z + z_step*3.0f, current_z + z_step*2.0f,
                        current_z + z_step*1.0f, current_z
                    );

                    for (; x <= end_x - 7; x += 8) {
                        __m256 v_old_z = _mm256_loadu_ps(&z_buffer[off + x]);
                        __m256 v_cmp = _mm256_cmp_ps(v_current_z, v_old_z, _CMP_LT_OQ);
                        __m256i v_mask = _mm256_castps_si256(v_cmp);

                        _mm256_maskstore_ps(&z_buffer[off + x], v_mask, v_current_z);
                        _mm256_maskstore_epi32((int*)&screen_buffer[off + x], v_mask, v_color);

                        v_current_z = _mm256_add_ps(v_current_z, v_z_step8);
                    }

                    // --- SCALAR TAIL LOOP ---
                    current_z = az + z_step * (x - ax);
                    for (; x <= end_x; x++) {
                        if (current_z < z_buffer[off + x]) {
                            z_buffer[off + x] = current_z;
                            screen_buffer[off + x] = (uint32_t)shaded_color[i];
                        }
                        current_z += z_step;
                    }
                }
            }
        }
    }
}

// ========================================================================
// SWARM PHYSICS (The Particle Baseline)
// ========================================================================

EXPORT void vmath_swarm_generate_quads(
    int count, float* px, float* py, float* pz,
    float* lx, float* ly, float* lz, float size,
    CameraState* cam, float HALF_W, float HALF_H
) {
    __m256 v_cpx = _mm256_set1_ps(cam->x), v_cpy = _mm256_set1_ps(cam->y), v_cpz = _mm256_set1_ps(cam->z);
    __m256 v_cfwx = _mm256_set1_ps(cam->fwx), v_cfwy = _mm256_set1_ps(cam->fwy), v_cfwz = _mm256_set1_ps(cam->fwz);
    __m256 v_crtx = _mm256_set1_ps(cam->rtx), v_crty = _mm256_set1_ps(cam->rty), v_crtz = _mm256_set1_ps(cam->rtz);
    __m256 v_cupx = _mm256_set1_ps(cam->upx), v_cupy = _mm256_set1_ps(cam->upy), v_cupz = _mm256_set1_ps(cam->upz);
    __m256 v_fov = _mm256_set1_ps(cam->fov);
    __m256 v_half_w = _mm256_set1_ps(HALF_W), v_half_h = _mm256_set1_ps(HALF_H);
    __m256 v_size = _mm256_set1_ps(size);
    __m256 v_0_1 = _mm256_set1_ps(0.1f);
    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    
    // The "Ghost" Coordinate: Exactly 1000 units behind the camera lens
    float dead_x = cam->x - cam->fwx * 1000.0f;
    float dead_y = cam->y - cam->fwy * 1000.0f;
    float dead_z = cam->z - cam->fwz * 1000.0f;

    int v_idx = 0;
    for (int i = 0; i <= count - 8; i += 8) {
        __m256 v_px = _mm256_loadu_ps(&px[i]);
        __m256 v_py = _mm256_loadu_ps(&py[i]);
        __m256 v_pz = _mm256_loadu_ps(&pz[i]);

        // 1. Vector from Camera to Particle
        __m256 dx = _mm256_sub_ps(v_px, v_cpx);
        __m256 dy = _mm256_sub_ps(v_py, v_cpy);
        __m256 dz = _mm256_sub_ps(v_pz, v_cpz);

        // 2. Project into Camera Space
        __m256 cz = _mm256_fmadd_ps(dz, v_cfwz, _mm256_fmadd_ps(dy, v_cfwy, _mm256_mul_ps(dx, v_cfwx)));
        __m256 depth = _mm256_max_ps(v_0_1, cz); // Prevent division by zero

        __m256 cx = _mm256_fmadd_ps(dz, v_crtz, _mm256_fmadd_ps(dy, v_crty, _mm256_mul_ps(dx, v_crtx)));
        __m256 cy = _mm256_fmadd_ps(dz, v_cupz, _mm256_fmadd_ps(dy, v_cupy, _mm256_mul_ps(dx, v_cupx)));

        // 3. Calculate Frustum Bounds at this depth
        __m256 frustum_w = _mm256_add_ps(_mm256_div_ps(_mm256_mul_ps(v_half_w, depth), v_fov), v_size);
        __m256 frustum_h = _mm256_add_ps(_mm256_div_ps(_mm256_mul_ps(v_half_h, depth), v_fov), v_size);

        __m256 abs_cx = _mm256_and_ps(cx, sign_mask);
        __m256 abs_cy = _mm256_and_ps(cy, sign_mask);

        // 4. Visibility Masks
        __m256 mask_z = _mm256_cmp_ps(_mm256_add_ps(cz, v_size), v_0_1, _CMP_GE_OQ);
        __m256 mask_x = _mm256_cmp_ps(abs_cx, frustum_w, _CMP_LE_OQ);
        __m256 mask_y = _mm256_cmp_ps(abs_cy, frustum_h, _CMP_LE_OQ);

        // Combine masks and extract as an 8-bit integer
        __m256 v_visible = _mm256_and_ps(mask_z, _mm256_and_ps(mask_x, mask_y));
        int visible_mask = _mm256_movemask_ps(v_visible);

        // FAST PATH: If all 8 particles are off-screen, dump 32 ghost vertices instantly
        if (visible_mask == 0) {
            for (int j = 0; j < 32; j++) {
                lx[v_idx] = dead_x; ly[v_idx] = dead_y; lz[v_idx] = dead_z; v_idx++;
            }
            continue;
        }

        // MIXED PATH: Evaluate which ones survive
        for (int j = 0; j < 8; j++) {
            if (visible_mask & (1 << j)) {
                // SURVIVOR: Build the actual tetrahedron
                float p_x = px[i+j], p_y = py[i+j], p_z = pz[i+j];
                lx[v_idx] = p_x; ly[v_idx] = p_y + size; lz[v_idx] = p_z; v_idx++;
                lx[v_idx] = p_x - size; ly[v_idx] = p_y - size; lz[v_idx] = p_z + size; v_idx++;
                lx[v_idx] = p_x + size; ly[v_idx] = p_y - size; lz[v_idx] = p_z + size; v_idx++;
                lx[v_idx] = p_x; ly[v_idx] = p_y - size; lz[v_idx] = p_z - size; v_idx++;
            } else {
                // CASUALTY: Teleport behind camera
                lx[v_idx] = dead_x; ly[v_idx] = dead_y; lz[v_idx] = dead_z; v_idx++;
                lx[v_idx] = dead_x; ly[v_idx] = dead_y; lz[v_idx] = dead_z; v_idx++;
                lx[v_idx] = dead_x; ly[v_idx] = dead_y; lz[v_idx] = dead_z; v_idx++;
                lx[v_idx] = dead_x; ly[v_idx] = dead_y; lz[v_idx] = dead_z; v_idx++;
            }
        }
    }
}
EXPORT void vmath_swarm_update_velocities(
    int count, float* px, float* py, float* pz,
    float* vx, float* vy, float* vz,
    float minX, float maxX, float minY, float maxY, float minZ, float maxZ,
    float dt, float gravity
) {
    __m256 v_minX = _mm256_set1_ps(minX), v_maxX = _mm256_set1_ps(maxX);
    __m256 v_minY = _mm256_set1_ps(minY), v_maxY = _mm256_set1_ps(maxY);
    __m256 v_minZ = _mm256_set1_ps(minZ), v_maxZ = _mm256_set1_ps(maxZ);
    __m256 v_dt = _mm256_set1_ps(dt);
    __m256 v_grav = _mm256_set1_ps(gravity * dt);
    __m256 v_drag = _mm256_set1_ps(0.995f);
    __m256 v_rest = _mm256_set1_ps(0.8f);
    __m256 v_neg_rest = _mm256_set1_ps(-0.8f);
    __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)); // For absolute value

    for (int i = 0; i <= count - 8; i += 8) {
        __m256 v_vx = _mm256_loadu_ps(&vx[i]);
        __m256 v_vy = _mm256_loadu_ps(&vy[i]);
        __m256 v_vz = _mm256_loadu_ps(&vz[i]);
        __m256 v_px = _mm256_loadu_ps(&px[i]);
        __m256 v_py = _mm256_loadu_ps(&py[i]);
        __m256 v_pz = _mm256_loadu_ps(&pz[i]);

        // Apply Gravity & Drag
        v_vy = _mm256_sub_ps(v_vy, v_grav);
        v_vx = _mm256_mul_ps(v_vx, v_drag);
        v_vy = _mm256_mul_ps(v_vy, v_drag);
        v_vz = _mm256_mul_ps(v_vz, v_drag);

        // Integrate Position
        v_px = _mm256_fmadd_ps(v_vx, v_dt, v_px);
        v_py = _mm256_fmadd_ps(v_vy, v_dt, v_py);
        v_pz = _mm256_fmadd_ps(v_vz, v_dt, v_pz);

        // --- X BOUNCE ---
        __m256 mask_minX = _mm256_cmp_ps(v_px, v_minX, _CMP_LT_OQ);
        __m256 mask_maxX = _mm256_cmp_ps(v_px, v_maxX, _CMP_GT_OQ);
        __m256 abs_vx = _mm256_and_ps(v_vx, sign_mask);
        v_px = _mm256_blendv_ps(v_px, v_minX, mask_minX);
        v_vx = _mm256_blendv_ps(v_vx, _mm256_mul_ps(abs_vx, v_rest), mask_minX);
        v_px = _mm256_blendv_ps(v_px, v_maxX, mask_maxX);
        v_vx = _mm256_blendv_ps(v_vx, _mm256_mul_ps(abs_vx, v_neg_rest), mask_maxX);

        // --- Y BOUNCE ---
        __m256 mask_minY = _mm256_cmp_ps(v_py, v_minY, _CMP_LT_OQ);
        __m256 mask_maxY = _mm256_cmp_ps(v_py, v_maxY, _CMP_GT_OQ);
        __m256 abs_vy = _mm256_and_ps(v_vy, sign_mask);
        v_py = _mm256_blendv_ps(v_py, v_minY, mask_minY);
        v_vy = _mm256_blendv_ps(v_vy, _mm256_mul_ps(abs_vy, v_rest), mask_minY);
        v_py = _mm256_blendv_ps(v_py, v_maxY, mask_maxY);
        v_vy = _mm256_blendv_ps(v_vy, _mm256_mul_ps(abs_vy, v_neg_rest), mask_maxY);

        // --- Z BOUNCE ---
        __m256 mask_minZ = _mm256_cmp_ps(v_pz, v_minZ, _CMP_LT_OQ);
        __m256 mask_maxZ = _mm256_cmp_ps(v_pz, v_maxZ, _CMP_GT_OQ);
        __m256 abs_vz = _mm256_and_ps(v_vz, sign_mask);
        v_pz = _mm256_blendv_ps(v_pz, v_minZ, mask_minZ);
        v_vz = _mm256_blendv_ps(v_vz, _mm256_mul_ps(abs_vz, v_rest), mask_minZ);
        v_pz = _mm256_blendv_ps(v_pz, v_maxZ, mask_maxZ);
        v_vz = _mm256_blendv_ps(v_vz, _mm256_mul_ps(abs_vz, v_neg_rest), mask_maxZ);

        _mm256_storeu_ps(&px[i], v_px); _mm256_storeu_ps(&py[i], v_py); _mm256_storeu_ps(&pz[i], v_pz);
        _mm256_storeu_ps(&vx[i], v_vx); _mm256_storeu_ps(&vy[i], v_vy); _mm256_storeu_ps(&vz[i], v_vz);
    }
}

EXPORT void vmath_swarm_apply_explosion(
    int count, float* px, float* py, float* pz,
    float* vx, float* vy, float* vz,
    float ex, float ey, float ez, float force, float radius
) {
    __m256 v_ex = _mm256_set1_ps(ex), v_ey = _mm256_set1_ps(ey), v_ez = _mm256_set1_ps(ez);
    __m256 v_r2 = _mm256_set1_ps(radius * radius);
    __m256 v_1 = _mm256_set1_ps(1.0f);
    __m256 v_force = _mm256_set1_ps(force);
    __m256 v_inv_radius = _mm256_set1_ps(1.0f / radius);

    for (int i = 0; i <= count - 8; i += 8) {
        __m256 dx = _mm256_sub_ps(_mm256_loadu_ps(&px[i]), v_ex);
        __m256 dy = _mm256_sub_ps(_mm256_loadu_ps(&py[i]), v_ey);
        __m256 dz = _mm256_sub_ps(_mm256_loadu_ps(&pz[i]), v_ez);

        __m256 dist2 = _mm256_fmadd_ps(dz, dz, _mm256_fmadd_ps(dy, dy, _mm256_mul_ps(dx, dx)));

        // Mask: 1.0f < dist2 < r2
        __m256 mask = _mm256_and_ps(_mm256_cmp_ps(dist2, v_r2, _CMP_LT_OQ), _mm256_cmp_ps(dist2, v_1, _CMP_GT_OQ));

        if (!_mm256_testz_ps(mask, mask)) {
            __m256 inv_dist = _mm256_rsqrt_ps(dist2); // Fast hardware inverse square root
            __m256 dist = _mm256_mul_ps(dist2, inv_dist);

            // f = force * (1.0f - dist * inv_radius)
            __m256 f = _mm256_mul_ps(v_force, _mm256_sub_ps(v_1, _mm256_mul_ps(dist, v_inv_radius)));
            __m256 f_inv_dist = _mm256_mul_ps(f, inv_dist); // (f / dist)

            __m256 v_vx = _mm256_loadu_ps(&vx[i]);
            __m256 v_vy = _mm256_loadu_ps(&vy[i]);
            __m256 v_vz = _mm256_loadu_ps(&vz[i]);

            v_vx = _mm256_blendv_ps(v_vx, _mm256_fmadd_ps(dx, f_inv_dist, v_vx), mask);
            v_vy = _mm256_blendv_ps(v_vy, _mm256_fmadd_ps(dy, f_inv_dist, v_vy), mask);
            v_vz = _mm256_blendv_ps(v_vz, _mm256_fmadd_ps(dz, f_inv_dist, v_vz), mask);

            _mm256_storeu_ps(&vx[i], v_vx);
            _mm256_storeu_ps(&vy[i], v_vy);
            _mm256_storeu_ps(&vz[i], v_vz);
        }
    }
}

// Boilerplate Spring Physics Macro to keep the shape functions perfectly clean
#define APPLY_SPRING_PHYSICS() \
    __m256 v_px = _mm256_loadu_ps(&px[i]), v_py = _mm256_loadu_ps(&py[i]), v_pz = _mm256_loadu_ps(&pz[i]); \
    __m256 v_vx = _mm256_loadu_ps(&vx[i]), v_vy = _mm256_loadu_ps(&vy[i]), v_vz = _mm256_loadu_ps(&vz[i]); \
    v_vx = _mm256_mul_ps(_mm256_fmadd_ps(_mm256_sub_ps(v_tx, v_px), v_k, v_vx), v_damp); \
    v_vy = _mm256_mul_ps(_mm256_fmadd_ps(_mm256_sub_ps(v_ty, v_py), v_k, v_vy), v_damp); \
    v_vz = _mm256_mul_ps(_mm256_fmadd_ps(_mm256_sub_ps(v_tz, v_pz), v_k, v_vz), v_damp); \
    _mm256_storeu_ps(&px[i], _mm256_fmadd_ps(v_vx, v_dt, v_px)); \
    _mm256_storeu_ps(&py[i], _mm256_fmadd_ps(v_vy, v_dt, v_py)); \
    _mm256_storeu_ps(&pz[i], _mm256_fmadd_ps(v_vz, v_dt, v_pz)); \
    _mm256_storeu_ps(&vx[i], v_vx); _mm256_storeu_ps(&vy[i], v_vy); _mm256_storeu_ps(&vz[i], v_vz);


EXPORT void vmath_swarm_bundle(
    int count, float* px, float* py, float* pz, float* vx, float* vy, float* vz, float* seed,
    float cx, float cy, float cz, float time, float dt
) {
    __m256 v_cx = _mm256_set1_ps(cx), v_cy = _mm256_set1_ps(cy), v_cz = _mm256_set1_ps(cz);
    __m256 v_r = _mm256_set1_ps(2000.0f + 400.0f * sinf(time * 6.0f));
    __m256 v_golden = _mm256_set1_ps(2.39996323f);
    __m256 v_1 = _mm256_set1_ps(1.0f), v_2 = _mm256_set1_ps(2.0f);
    __m256 v_dt = _mm256_set1_ps(dt), v_k = _mm256_set1_ps(4.0f * dt), v_damp = _mm256_set1_ps(0.92f);

    for (int i = 0; i <= count - 8; i += 8) {
        __m256 v_s = _mm256_loadu_ps(&seed[i]);
        __m256 v_i = _mm256_set_ps(i+7, i+6, i+5, i+4, i+3, i+2, i+1, i);

        __m256 v_phi = _mm256_mul_ps(v_i, v_golden);

        // Math Hack: No acos needed! cos(theta) = 1-2s. sin(theta) = 2*sqrt(s*(1-s))
        __m256 v_cos_theta = _mm256_fnmadd_ps(v_2, v_s, v_1);
        __m256 v_sin_theta = _mm256_mul_ps(v_2, _mm256_sqrt_ps(_mm256_mul_ps(v_s, _mm256_sub_ps(v_1, v_s))));

        __m256 v_tx = _mm256_fmadd_ps(v_r, _mm256_mul_ps(v_sin_theta, fast_cos_avx(v_phi)), v_cx);
        __m256 v_ty = _mm256_fmadd_ps(v_r, v_cos_theta, v_cy);
        __m256 v_tz = _mm256_fmadd_ps(v_r, _mm256_mul_ps(v_sin_theta, fast_sin_avx(v_phi)), v_cz);

        APPLY_SPRING_PHYSICS();
    }
}

EXPORT void vmath_swarm_galaxy(
    int count, float* px, float* py, float* pz, float* vx, float* vy, float* vz, float* seed, 
    float cx, float cy, float cz, float time, float dt
) {
    __m256 v_cx = _mm256_set1_ps(cx), v_cy = _mm256_set1_ps(cy), v_cz = _mm256_set1_ps(cz);
    __m256 v_time_ang = _mm256_set1_ps(time * 1.5f), v_time_z = _mm256_set1_ps(time * 3.0f);
    __m256 v_dt = _mm256_set1_ps(dt), v_k = _mm256_set1_ps(4.0f * dt), v_damp = _mm256_set1_ps(0.92f);

    for (int i = 0; i <= count - 8; i += 8) {
        __m256 v_s = _mm256_loadu_ps(&seed[i]);
        __m256 v_angle = _mm256_fmadd_ps(v_s, _mm256_set1_ps(3.14159f * 30.0f), v_time_ang);
        __m256 v_r = _mm256_fmadd_ps(v_s, _mm256_set1_ps(14000.0f), _mm256_set1_ps(1000.0f));

        __m256 v_tx = _mm256_fmadd_ps(v_r, fast_cos_avx(v_angle), v_cx);
        __m256 v_ty = _mm256_fmadd_ps(_mm256_set1_ps(800.0f), fast_sin_avx(_mm256_fnmadd_ps(v_time_z, _mm256_set1_ps(1.0f), _mm256_mul_ps(v_s, _mm256_set1_ps(40.0f)))), v_cy);
        __m256 v_tz = _mm256_fmadd_ps(v_r, fast_sin_avx(v_angle), v_cz);

        APPLY_SPRING_PHYSICS();
    }
}

EXPORT void vmath_swarm_tornado(
    int count, float* px, float* py, float* pz, float* vx, float* vy, float* vz, float* seed,
    float cx, float cy, float cz, float time, float dt
) {
    __m256 v_cx = _mm256_set1_ps(cx), v_cy = _mm256_set1_ps(cy), v_cz = _mm256_set1_ps(cz);
    __m256 v_time_ang = _mm256_set1_ps(time * 4.0f);
    __m256 v_dt = _mm256_set1_ps(dt), v_k = _mm256_set1_ps(4.0f * dt), v_damp = _mm256_set1_ps(0.92f);

    for (int i = 0; i <= count - 8; i += 8) {
        __m256 v_s = _mm256_loadu_ps(&seed[i]);
        __m256 v_height = _mm256_fnmadd_ps(_mm256_set1_ps(-24000.0f), v_s, _mm256_set1_ps(-12000.0f));
        __m256 v_angle = _mm256_fnmadd_ps(v_time_ang, _mm256_set1_ps(1.0f), _mm256_mul_ps(v_s, _mm256_set1_ps(3.14159f * 30.0f)));
        __m256 v_r = _mm256_fmadd_ps(v_s, _mm256_set1_ps(4000.0f), _mm256_set1_ps(2000.0f));

        __m256 v_tx = _mm256_fmadd_ps(v_r, fast_cos_avx(v_angle), v_cx);
        __m256 v_ty = _mm256_add_ps(v_cy, v_height);
        __m256 v_tz = _mm256_fmadd_ps(v_r, fast_sin_avx(v_angle), v_cz);

        APPLY_SPRING_PHYSICS();
    }
}

EXPORT void vmath_swarm_gyroscope(
    int count, float* px, float* py, float* pz, float* vx, float* vy, float* vz, float* seed,
    float cx, float cy, float cz, float time, float dt
) {
    __m256 v_cx = _mm256_set1_ps(cx), v_cy = _mm256_set1_ps(cy), v_cz = _mm256_set1_ps(cz);
    __m256 v_r = _mm256_set1_ps(7000.0f);
    __m256 v_time_ang = _mm256_set1_ps(time * 2.5f);
    __m256 v_dt = _mm256_set1_ps(dt), v_k = _mm256_set1_ps(4.0f * dt), v_damp = _mm256_set1_ps(0.92f);

    for (int i = 0; i <= count - 8; i += 8) {
        __m256 v_s = _mm256_loadu_ps(&seed[i]);
        __m256 v_angle = _mm256_fmadd_ps(v_s, _mm256_set1_ps(3.14159f * 2.0f), v_time_ang);

        __m256 v_cos = fast_cos_avx(v_angle);
        __m256 v_sin = fast_sin_avx(v_angle);

        // Calculate all 3 ring positions simultaneously!
        __m256 r0_x = _mm256_fmadd_ps(v_r, v_cos, v_cx), r0_y = _mm256_fmadd_ps(v_r, v_sin, v_cy), r0_z = v_cz;
        __m256 r1_x = r0_x, r1_y = v_cy, r1_z = _mm256_fmadd_ps(v_r, v_sin, v_cz);
        __m256 r2_x = v_cx, r2_y = _mm256_fmadd_ps(v_r, v_cos, v_cy), r2_z = r1_z;

        // Masking logic based on (i % 3)
        int rings[8] = { (i)%3, (i+1)%3, (i+2)%3, (i+3)%3, (i+4)%3, (i+5)%3, (i+6)%3, (i+7)%3 };
        __m256i v_ring = _mm256_loadu_si256((__m256i*)rings);

        __m256 m0 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(v_ring, _mm256_setzero_si256()));
        __m256 m1 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(v_ring, _mm256_set1_epi32(1)));

        __m256 v_tx = _mm256_blendv_ps(r2_x, _mm256_blendv_ps(r1_x, r0_x, m0), _mm256_or_ps(m0, m1));
        __m256 v_ty = _mm256_blendv_ps(r2_y, _mm256_blendv_ps(r1_y, r0_y, m0), _mm256_or_ps(m0, m1));
        __m256 v_tz = _mm256_blendv_ps(r2_z, _mm256_blendv_ps(r1_z, r0_z, m0), _mm256_or_ps(m0, m1));

        APPLY_SPRING_PHYSICS();
    }
}

EXPORT void vmath_swarm_metal(
    int count,
    float* px, float* py, float* pz,
    float* vx, float* vy, float* vz,
    float* seed, // 0.0 to 1.0 particle ID
    float cx, float cy, float cz,
    float time, float dt,
    float noise_blend // 0.0 = Perfect Sphere, 1.0 = Boiling Metal
) {
    __m256 v_cx = _mm256_set1_ps(cx), v_cy = _mm256_set1_ps(cy), v_cz = _mm256_set1_ps(cz);
    __m256 v_time = _mm256_set1_ps(time);
    __m256 v_blend = _mm256_set1_ps(noise_blend);
    __m256 v_radius = _mm256_set1_ps(4000.0f);
    __m256 v_max_disp = _mm256_set1_ps(3000.0f); // Max noise distortion

    __m256 v_dt = _mm256_set1_ps(dt);
    __m256 v_k = _mm256_set1_ps(4.0f * dt); // Spring stiffness
    __m256 v_damp = _mm256_set1_ps(0.92f);  // Friction

    int i = 0;
    // BLAST 8 PARTICLES PER CYCLE
    for (; i <= count - 8; i += 8) {
        __m256 v_s = _mm256_loadu_ps(&seed[i]);

        // 1. FAST SPHERICAL MAPPING (Fibonacci-style distribution without acos)
        // Z goes from 1.0 to -1.0 based on seed
        __m256 v_sz = _mm256_fnmadd_ps(v_s, _mm256_set1_ps(2.0f), _mm256_set1_ps(1.0f));
        // Radius at this Z: r_xy = sqrt(1.0 - z*z)
        __m256 v_rxy = _mm256_sqrt_ps(_mm256_fnmadd_ps(v_sz, v_sz, _mm256_set1_ps(1.0f)));
        // Phi rotates wildly based on seed
        __m256 v_phi = _mm256_mul_ps(v_s, _mm256_set1_ps(10000.0f));

        __m256 v_sx = _mm256_mul_ps(v_rxy, fast_cos_avx(v_phi));
        __m256 v_sy = _mm256_mul_ps(v_rxy, fast_sin_avx(v_phi));

        // 2. EVALUATE 4D NOISE AT THE NORMALS
        __m256 v_noise = fast_trig_noise_avx(v_sx, v_sy, v_sz, v_time);

        // 3. APPLY DISPLACEMENT (Using FMA to blend seamlessly!)
        // displacement = noise * noise_blend * max_disp
        __m256 v_disp = _mm256_mul_ps(v_noise, _mm256_mul_ps(v_blend, v_max_disp));

        // Target Pos = Center + Normal * (Radius + Displacement)
        __m256 v_final_r = _mm256_add_ps(v_radius, v_disp);
        __m256 v_tx = _mm256_fmadd_ps(v_sx, v_final_r, v_cx);
        __m256 v_ty = _mm256_fmadd_ps(v_sy, v_final_r, v_cy);
        __m256 v_tz = _mm256_fmadd_ps(v_sz, v_final_r, v_cz);

        // 4. SPRING PHYSICS (Pull current pos toward Target Pos)
        __m256 v_px = _mm256_loadu_ps(&px[i]);
        __m256 v_py = _mm256_loadu_ps(&py[i]);
        __m256 v_pz = _mm256_loadu_ps(&pz[i]);

        __m256 v_vx = _mm256_loadu_ps(&vx[i]);
        __m256 v_vy = _mm256_loadu_ps(&vy[i]);
        __m256 v_vz = _mm256_loadu_ps(&vz[i]);

        // v += (target - p) * k * dt; v *= damp;
        v_vx = _mm256_mul_ps(_mm256_fmadd_ps(_mm256_sub_ps(v_tx, v_px), v_k, v_vx), v_damp);
        v_vy = _mm256_mul_ps(_mm256_fmadd_ps(_mm256_sub_ps(v_ty, v_py), v_k, v_vy), v_damp);
        v_vz = _mm256_mul_ps(_mm256_fmadd_ps(_mm256_sub_ps(v_tz, v_pz), v_k, v_vz), v_damp);

        // p += v * dt;
        v_px = _mm256_fmadd_ps(v_vx, v_dt, v_px);
        v_py = _mm256_fmadd_ps(v_vy, v_dt, v_py);
        v_pz = _mm256_fmadd_ps(v_vz, v_dt, v_pz);

        _mm256_storeu_ps(&px[i], v_px);
        _mm256_storeu_ps(&py[i], v_py);
        _mm256_storeu_ps(&pz[i], v_pz);
        _mm256_storeu_ps(&vx[i], v_vx);
        _mm256_storeu_ps(&vy[i], v_vy);
        _mm256_storeu_ps(&vz[i], v_vz);
    }

    // (Scalar tail loop for remainder goes here, though PCOUNT = 10000 is perfectly mod 8)
}

EXPORT void vmath_swarm_smales(
    int count, float* px, float* py, float* pz,
    float* vx, float* vy, float* vz, float* seed,
    float cx, float cy, float cz,
    float time, float dt, float blend
) {
    __m256 v_cx = _mm256_set1_ps(cx), v_cy = _mm256_set1_ps(cy), v_cz = _mm256_set1_ps(cz);
    __m256 v_base_radius = _mm256_set1_ps(4000.0f);

    // THE DOD BLENDING MATH (Calculated once outside the loop!)
    // If blend=0: eversion=1.0, bulge=0.0
    // If blend=1: eversion=cos(t), bulge=sin(t)
    float t_scaled = time * 1.5f;
    float eversion_scalar = 1.0f + blend * (cosf(t_scaled) - 1.0f);
    float bulge_scalar = blend * sinf(t_scaled);

    __m256 v_eversion = _mm256_set1_ps(eversion_scalar);
    __m256 v_bulge = _mm256_set1_ps(bulge_scalar);

    __m256 v_1_2 = _mm256_set1_ps(1.2f);
    __m256 v_0_5 = _mm256_set1_ps(0.5f);
    __m256 v_4_0 = _mm256_set1_ps(4.0f);
    __m256 v_2_0 = _mm256_set1_ps(2.0f);
    __m256 v_3_0 = _mm256_set1_ps(3.0f);
    __m256 v_pi = _mm256_set1_ps(M_PI);
    __m256 v_phi_mul = _mm256_set1_ps(M_PI * 2.0f * 100.0f); // Wrap phi around 100 times

    __m256 v_dt = _mm256_set1_ps(dt);
    __m256 v_k = _mm256_set1_ps(4.0f * dt);
    __m256 v_damp = _mm256_set1_ps(0.92f);

    int i = 0;
    for (; i <= count - 8; i += 8) {
        __m256 v_s = _mm256_loadu_ps(&seed[i]);

        // 1. Map seed to Theta [0, PI] and Phi [0, 2PI * 100]
        __m256 v_theta = _mm256_mul_ps(v_s, v_pi);
        __m256 v_phi = _mm256_mul_ps(v_s, v_phi_mul);

        __m256 v_ny = fast_cos_avx(v_theta);
        __m256 v_sin_theta = fast_sin_avx(v_theta);

        __m256 v_nx = _mm256_mul_ps(v_sin_theta, fast_cos_avx(v_phi));
        __m256 v_nz = _mm256_mul_ps(v_sin_theta, fast_sin_avx(v_phi));

        // 2. PARADOX MATH
        __m256 v_waves = fast_cos_avx(_mm256_mul_ps(v_phi, v_4_0));
        __m256 v_twist = fast_sin_avx(_mm256_mul_ps(v_theta, v_2_0));

        __m256 v_r_corr = _mm256_mul_ps(v_base_radius,
                          _mm256_mul_ps(v_bulge,
                          _mm256_mul_ps(v_waves,
                          _mm256_mul_ps(v_twist, v_1_2))));

        __m256 v_r_main = _mm256_mul_ps(v_base_radius, v_eversion);

        // 3. APPLY DISPLACEMENT
        __m256 v_tx = _mm256_fmadd_ps(v_nx, _mm256_add_ps(v_r_main, v_r_corr), v_cx);
        __m256 v_tz = _mm256_fmadd_ps(v_nz, _mm256_add_ps(v_r_main, v_r_corr), v_cz);

        __m256 v_ty_offset = _mm256_mul_ps(fast_cos_avx(_mm256_mul_ps(v_theta, v_3_0)),
                             _mm256_mul_ps(v_base_radius,
                             _mm256_mul_ps(v_bulge, v_0_5)));

        __m256 v_ty = _mm256_add_ps(v_cy, _mm256_fmadd_ps(v_ny, v_r_main, v_ty_offset));

        // 4. SPRING PHYSICS
        __m256 v_px = _mm256_loadu_ps(&px[i]);
        __m256 v_py = _mm256_loadu_ps(&py[i]);
        __m256 v_pz = _mm256_loadu_ps(&pz[i]);

        __m256 v_vx = _mm256_loadu_ps(&vx[i]);
        __m256 v_vy = _mm256_loadu_ps(&vy[i]);
        __m256 v_vz = _mm256_loadu_ps(&vz[i]);

        v_vx = _mm256_mul_ps(_mm256_fmadd_ps(_mm256_sub_ps(v_tx, v_px), v_k, v_vx), v_damp);
        v_vy = _mm256_mul_ps(_mm256_fmadd_ps(_mm256_sub_ps(v_ty, v_py), v_k, v_vy), v_damp);
        v_vz = _mm256_mul_ps(_mm256_fmadd_ps(_mm256_sub_ps(v_tz, v_pz), v_k, v_vz), v_damp);

        v_px = _mm256_fmadd_ps(v_vx, v_dt, v_px);
        v_py = _mm256_fmadd_ps(v_vy, v_dt, v_py);
        v_pz = _mm256_fmadd_ps(v_vz, v_dt, v_pz);

        _mm256_storeu_ps(&px[i], v_px);
        _mm256_storeu_ps(&py[i], v_py);
        _mm256_storeu_ps(&pz[i], v_pz);
        _mm256_storeu_ps(&vx[i], v_vx);
        _mm256_storeu_ps(&vy[i], v_vy);
        _mm256_storeu_ps(&vz[i], v_vz);
    }
}
// ========================================================================
// RENDER BATCH HELPERS
// ========================================================================

EXPORT void vmath_render_batch(
    int start_id, int end_id,
    CameraState* cam,
    float HALF_W, float HALF_H,
    float sun_x, float sun_y, float sun_z,
    RenderMemory* mem,
    uint32_t* ScreenPtr, float* ZBuffer, int CANVAS_W, int CANVAS_H
) {
    // 1. Unpack Camera State locally (Very fast)
    float cpx = cam->x, cpy = cam->y, cpz = cam->z;
    float cfw_x = cam->fwx, cfw_y = cam->fwy, cfw_z = cam->fwz;
    float crt_x = cam->rtx, crt_z = cam->rtz;
    float cup_x = cam->upx, cup_y = cam->upy, cup_z = cam->upz;
    float cam_fov = cam->fov;

    for (int id = start_id; id <= end_id; id++) {
        float r = mem->Obj_Radius[id];
        float ox = mem->Obj_X[id], oy = mem->Obj_Y[id], oz = mem->Obj_Z[id];

        // 1. Coarse Z-Cull
        float cz_center = (ox - cpx)*cfw_x + (oy - cpy)*cfw_y + (oz - cpz)*cfw_z;
        if (cz_center + r < 0.1f) continue;

        // 2. Fetch object matrices & slice info
        float rx = mem->Obj_RTX[id], ry = mem->Obj_RTY[id], rz = mem->Obj_RTZ[id];
        float ux = mem->Obj_UPX[id], uy = mem->Obj_UPY[id], uz = mem->Obj_UPZ[id];
        float fx = mem->Obj_FWX[id], fy = mem->Obj_FWY[id], fz = mem->Obj_FWZ[id];
        int vStart = mem->Obj_VertStart[id], vCount = mem->Obj_VertCount[id];
        int tStart = mem->Obj_TriStart[id], tCount = mem->Obj_TriCount[id];

        // 3. Project Vertices
        vmath_project_vertices(
            vCount,
            mem->Vert_LX + vStart, mem->Vert_LY + vStart, mem->Vert_LZ + vStart,
            mem->Vert_PX + vStart, mem->Vert_PY + vStart, mem->Vert_PZ + vStart, mem->Vert_Valid + vStart,
            ox, oy, oz, rx, ry, rz, ux, uy, uz, fx, fy, fz,
            cpx, cpy, cpz, cfw_x, cfw_y, cfw_z, crt_x, crt_z, cup_x, cup_y, cup_z,
            cam_fov, HALF_W, HALF_H
        );

        // 4. Assemble & Light Triangles
        vmath_process_triangles(
            tCount,
            mem->Tri_V1 + tStart, mem->Tri_V2 + tStart, mem->Tri_V3 + tStart, mem->Vert_Valid,
            mem->Vert_PX, mem->Vert_PY, mem->Vert_PZ, mem->Vert_LX, mem->Vert_LY, mem->Vert_LZ,
            mem->Tri_BakedColor + tStart, mem->Tri_ShadedColor + tStart, mem->Tri_Valid + tStart,
            rx, ry, rz, ux, uy, uz, fx, fy, fz,
            sun_x, sun_y, sun_z
        );

        // 5. Rasterize
        vmath_rasterize_triangles(
            tCount,
            mem->Tri_V1 + tStart, mem->Tri_V2 + tStart, mem->Tri_V3 + tStart, mem->Tri_Valid + tStart,
            mem->Vert_PX, mem->Vert_PY, mem->Vert_PZ, mem->Tri_ShadedColor + tStart,
            ScreenPtr, ZBuffer, CANVAS_W, CANVAS_H
        );
    }
}

// ========================================================================
// TESTBED
// ========================================================================

// A dead-simple scalar sphere. No noise, no SIMD, purely a stable target for our Transition Weaving tests.
EXPORT void vmath_generate_basic_sphere(float* lx, float* ly, float* lz, int latitudes, int longitudes, float radius) {
    int idx = 0;
    for (int i = 0; i <= latitudes; i++) {
        float theta = ((float)i / latitudes) * (float)M_PI;
        float sin_theta = sinf(theta);
        float cos_theta = cosf(theta);

        for (int j = 0; j <= longitudes; j++) {
            float phi = ((float)j / longitudes) * (float)M_PI * 2.0f;

            lx[idx] = sin_theta * cosf(phi) * radius;
            ly[idx] = cos_theta * radius;
            lz[idx] = sin_theta * sinf(phi) * radius;
            idx++;
        }
    }
}

// THE COMMAND QUEUE DISPATCHER (100% Branchless)
EXPORT void vmath_execute_queue(
    int* queue, int command_count,
    CameraState* cam, RenderMemory* mem,
    uint32_t* ScreenPtr, float* ZBuffer,
    int CANVAS_W, int CANVAS_H,
    float time, float dt
) {
    float HALF_W = CANVAS_W * 0.5f;
    float HALF_H = CANVAS_H * 0.5f;
    float sun_x = 0.577f, sun_y = -0.577f, sun_z = 0.577f;

    for (int i = 0; i < command_count; i++) {
        int opcode = queue[i];

        switch (opcode) {
            case 1: // CMD_CLEAR
                vmath_clear_buffers(ScreenPtr, ZBuffer, 0xFF000000, 99999.0f, CANVAS_W * CANVAS_H);
                break;

            case 2: // SWARM_APPLY_BASE_PHYSICS
                vmath_swarm_update_velocities(10000, mem->Swarm_PX, mem->Swarm_PY, mem->Swarm_PZ, mem->Swarm_VX, mem->Swarm_VY, mem->Swarm_VZ, -15000, 15000, -4000, 15000, -15000, 15000, dt, -8000.0f * mem->Swarm_GravityBlend);
                break;

            case 3: // SWARM_BUNDLE (State 1)
                vmath_swarm_bundle(10000, mem->Swarm_PX, mem->Swarm_PY, mem->Swarm_PZ, mem->Swarm_VX, mem->Swarm_VY, mem->Swarm_VZ, mem->Swarm_Seed, 0, 5000, 0, time, dt);
                break;

            case 4: // SWARM_GALAXY (State 2)
                vmath_swarm_galaxy(10000, mem->Swarm_PX, mem->Swarm_PY, mem->Swarm_PZ, mem->Swarm_VX, mem->Swarm_VY, mem->Swarm_VZ, mem->Swarm_Seed, 0, 5000, 0, time, dt);
                break;

            case 5: // SWARM_TORNADO (State 3)
                vmath_swarm_tornado(10000, mem->Swarm_PX, mem->Swarm_PY, mem->Swarm_PZ, mem->Swarm_VX, mem->Swarm_VY, mem->Swarm_VZ, mem->Swarm_Seed, 0, 5000, 0, time, dt);
                break;

            case 6: // SWARM_GYROSCOPE (State 4)
                vmath_swarm_gyroscope(10000, mem->Swarm_PX, mem->Swarm_PY, mem->Swarm_PZ, mem->Swarm_VX, mem->Swarm_VY, mem->Swarm_VZ, mem->Swarm_Seed, 0, 5000, 0, time, dt);
                break;

            case 7: // SWARM_METAL (State 5)
                vmath_swarm_metal(10000, mem->Swarm_PX, mem->Swarm_PY, mem->Swarm_PZ, mem->Swarm_VX, mem->Swarm_VY, mem->Swarm_VZ, mem->Swarm_Seed, 0, 5000, 0, time, dt, mem->Swarm_MetalBlend);
                break;

            case 8: // SWARM_PARADOX (State 6)
                vmath_swarm_smales(10000, mem->Swarm_PX, mem->Swarm_PY, mem->Swarm_PZ, mem->Swarm_VX, mem->Swarm_VY, mem->Swarm_VZ, mem->Swarm_Seed, 0, 5000, 0, time, dt, mem->Swarm_ParadoxBlend);
                break;

            case 9: // SWARM_GEN_QUADS
                vmath_swarm_generate_quads(10000, mem->Swarm_PX, mem->Swarm_PY, mem->Swarm_PZ, mem->Vert_LX + mem->Obj_VertStart[0], mem->Vert_LY + mem->Obj_VertStart[0], mem->Vert_LZ + mem->Obj_VertStart[0], 120.0f, cam, HALF_W, HALF_H);
                break;

            case 10: // SPHERE_TICK
                vmath_generate_basic_sphere(mem->Vert_LX + mem->Obj_VertStart[1], mem->Vert_LY + mem->Obj_VertStart[1], mem->Vert_LZ + mem->Obj_VertStart[1], 100, 100, 3500.0f);
                break;

            case 11: { // RENDER_CULL
                int id = queue[++i];
                vmath_render_batch(id, id, cam, HALF_W, HALF_H, sun_x, sun_y, sun_z, mem, ScreenPtr, ZBuffer, CANVAS_W, CANVAS_H);
                break;
            }

            case 12: // SWARM_EXPLOSION_PUSH
                vmath_swarm_apply_explosion(10000, mem->Swarm_PX, mem->Swarm_PY, mem->Swarm_PZ, mem->Swarm_VX, mem->Swarm_VY, mem->Swarm_VZ, 0, 5000, 0, 5000000.0f * dt, 15000.0f);
                break;

            case 13: // SWARM_EXPLOSION_PULL
                vmath_swarm_apply_explosion(10000, mem->Swarm_PX, mem->Swarm_PY, mem->Swarm_PZ, mem->Swarm_VX, mem->Swarm_VY, mem->Swarm_VZ, 0, 5000, 0, -4000000.0f * dt, 20000.0f);
                break;
        }
    }
}

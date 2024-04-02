#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>

#define MAX_VERTS 2e6 // 2 million
#define MAX_FACES 1e6 // 1 million

#define HEAD(i) (i.x)
#define INFACE(i) (i.y)
#define NEXT(i) (i.z)
#define TWIN(i) (i.w)

struct GlobalConstants {
    size_t num_verts;
    size_t num_faces;
    float3* vertices;
    uint3* faces;

    size_t out_num_verts;
    
    int4* halfedges; 
    int3* halfedges_HTI; // head and next tail and original index
    uint* hnt_first_vertex;
};

struct float3_equal {
    __host__ __device__
    bool operator()(const float3& lhs, const float3& rhs) {
        return abs(lhs.x - rhs.x) < 1e-6 && abs(lhs.y - rhs.y) < 1e-6 && abs(lhs.z - rhs.z) < 1e-6;
    }
};


struct float3_less {
    __host__ __device__
    bool operator()(const float3& lhs, const float3& rhs) {
        if (lhs.x != rhs.x) return lhs.x < rhs.x;
        if (lhs.y != rhs.y) return lhs.y < rhs.y;
        return lhs.z < rhs.z;
    }
};

struct int2_less {
    __host__ __device__
    bool operator()(const int2& lhs, const int2& rhs) {
        return lhs.x < rhs.x;
    }
};

struct comparehead{
    __host__ __device__
    bool operator()(const int3& lhs, const int3& rhs) {
        return lhs.x < rhs.x;
    }
};

struct comparesame{
    __host__ __device__
    bool operator()(const int3& lhs, const int3& rhs) {
        return lhs.x == rhs.x;
    }
};




__host__ void InitCUDA();

__host__ void ExitCUDA();

__global__ void updateFaces(uint3* faces, int num_faces, uint* vertex_map);

__host__ void meshcopyHtoD(const size_t num_verts, const size_t num_faces, const float3* const vertices, const uint3* const faces);

__host__ void meshcopyDtoH(thrust::host_vector<float3>& out_verts, thrust::host_vector<int4>& half_edges);

__host__ size_t deduplicateVertices(GlobalConstants &mesh, thrust::device_vector<float3>& unique_vertices, thrust::device_vector<uint>& new_vertex_map);

__global__ void construct_halfedges(int4* halfedges, uint3* faces, int num_faces, int3* HTI);

__global__ void fill_halfedges_twin(size_t num_verts, size_t num_halfedges, int4* halfedges, int3* sorted_HTI, uint* frist_vertex_idx);

__host__ void generate_halfedges(GlobalConstants &mesh);

__host__ size_t deduplicateAndGenerateHalfedges();
#include "kernel.cuh"

float3* d_vertices;
uint3* d_faces;

thrust::device_vector<float3> unique_vertices;
thrust::device_vector<int4> d_half_edges;

size_t out_num_verts;


__host__ __device__
bool operator==(const int2& a, const int2& b)
{
    return a.x == b.x;
}


__constant__ GlobalConstants targetMesh;

__host__ void InitCUDA()
{
  int deviceCount = 0;
  std::string name;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  printf("---------------------------------------------------------\n");
  printf("Initializing CUDA\n");
  printf("Found %d CUDA devices\n", deviceCount);

  for (int i=0; i<deviceCount; i++) {
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, i);
    printf("Device %d: %s\n", i, deviceProps.name);
    printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
    printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
    printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
  }
  printf("---------------------------------------------------------\n");
  
  setenv("CUDA_VISIBLE_DEVICES", "1", 1);
  cudaError_t cudaStatus = cudaSetDevice(1);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice failed!");
    return;
  }
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 1);
  printf("Connencting to device 1\n");
  printf("   Device ID: %d\n", 1);
  printf("   Device name: %s\n", prop.name);
  printf("   Global memory: %.1f MB\n", prop.totalGlobalMem);
  printf("   CUDA Capability: %d.%d\n", prop.major, prop.minor);
  printf("---------------------------------------------------------\n");

  cudaMalloc((void**)&d_vertices, MAX_VERTS * sizeof(float3));
  cudaMalloc((void**)&d_faces, MAX_FACES * sizeof(int3));

  cudaMemset(d_vertices, 0, MAX_VERTS * sizeof(float3));
  cudaMemset(d_faces, 0, MAX_FACES * sizeof(int3));

}

__host__ void ExitCUDA()
{
  cudaDeviceSynchronize();

  cudaFree(d_vertices);
  cudaFree(d_faces);


  cudaDeviceReset();
  cudaDeviceSynchronize();
}

__global__ void updateFaces(uint3* faces, int num_faces, uint* vertex_map)
{
  uint t_id = blockIdx.x * blockDim.x + threadIdx.x;
  if(t_id < num_faces)
  {
    uint3 face = faces[t_id];
    face.x = vertex_map[face.x];
    face.y = vertex_map[face.y];
    face.z = vertex_map[face.z];
    faces[t_id] = face;
  }
}


__host__ void meshcopyHtoD(const size_t num_verts, const size_t num_faces, const float3* const vertices, const uint3* const faces)
{
  if(num_verts > MAX_VERTS || num_faces > MAX_FACES)
  {
    printf("Error: too many vertices or faces | current limit: 2 million verts, 1 million faces \n");
    return;
  }

  cudaMemcpy(d_vertices, vertices, num_verts * sizeof(float3), cudaMemcpyHostToDevice);
  cudaMemcpy(d_faces, faces, num_faces * sizeof(uint3), cudaMemcpyHostToDevice);

  GlobalConstants temp;
  temp.num_verts = num_verts;
  temp.num_faces = num_faces;
  temp.vertices = d_vertices;
  temp.faces = d_faces;
  temp.halfedges = nullptr;
  cudaMemcpyToSymbol(targetMesh, &temp, sizeof(GlobalConstants));

  cudaDeviceSynchronize();
}

__host__ void meshcopyDtoH(thrust::host_vector<float3>& out_verts, thrust::host_vector<int4>& half_edges)
{
  GlobalConstants temp;
  cudaMemcpyFromSymbol(&temp, targetMesh, sizeof(GlobalConstants));
  
  thrust::copy(unique_vertices.begin(), unique_vertices.end(), out_verts.begin());
  
  thrust::copy(d_half_edges.begin(), d_half_edges.end(), half_edges.begin());
  
  cudaDeviceSynchronize();
}

/**
 * @brief The function to deduplicate vertices using thrust library
 * 
 * @param mesh input mesh data
 * @param unique_vertices output unique vertices
 * @param new_vertex_map output old to new vertex map
 * @return size_t the number of unique vertices 
 */
__host__ size_t deduplicateVertices(GlobalConstants &mesh, thrust::device_vector<float3>& unique_vertices, thrust::device_vector<uint>& new_vertex_map)
{
  printf("[Deduplication] Input Vertices: %d\n", mesh.num_verts);

  thrust::device_vector<float3> device_vertices = thrust::device_vector<float3>(d_vertices, d_vertices + mesh.num_verts);
  unique_vertices = device_vertices;
  thrust::device_vector<uint> vertex_map(mesh.num_verts);

  thrust::sequence(vertex_map.begin(), vertex_map.end());
  thrust::stable_sort_by_key(unique_vertices.begin(), unique_vertices.end(), vertex_map.begin(),float3_less());

  auto end = thrust::unique_by_key(unique_vertices.begin(), unique_vertices.end(), vertex_map.begin(), float3_equal());
  unique_vertices.erase(end.first, unique_vertices.end());

  thrust::lower_bound(unique_vertices.begin(), unique_vertices.end(), device_vertices.begin(), device_vertices.end(), new_vertex_map.begin(), float3_less());
  out_num_verts = unique_vertices.size();

  // allocate unique vertices to d_vertices
  cudaMemset(d_vertices, 0, MAX_VERTS * sizeof(float3));
  cudaMemcpy(d_vertices, thrust::raw_pointer_cast(unique_vertices.data()), out_num_verts * sizeof(float3), cudaMemcpyDeviceToDevice);

  printf("[Deduplication] Output Vertices: %d\n", out_num_verts);
  return out_num_verts;
}


/**
 * @brief The function to construct halfedges
 * 
 * @param halfedges output device pointer to halfedges
 * @param faces input device pointer to faces
 * @param num_faces input number of faces
 * @param HTI output device pointer to halfedge's head and tail and original index
 */
__global__ void construct_halfedges(int4* halfedges, uint3* faces, int num_faces, int3* HTI)
{
  uint t_id = blockIdx.x * blockDim.x + threadIdx.x;
  if(t_id < num_faces)
  { 
    uint he0_id = 3*t_id;
    uint he1_id = 3*t_id+1;
    uint he2_id = 3*t_id+2;
    int4 he0 = halfedges[he0_id];
    int4 he1 = halfedges[he1_id];
    int4 he2 = halfedges[he2_id];
    int3 tempHTI;

    HEAD(he0) = faces[t_id].x;
    INFACE(he0) = t_id;
    NEXT(he0) = he1_id;
    TWIN(he0) = faces[t_id].y;
    halfedges[he0_id] = he0;

    HEAD(he1) = faces[t_id].y;
    INFACE(he1) = t_id;
    NEXT(he1) = he2_id;
    TWIN(he1) = faces[t_id].z;
    halfedges[he1_id] = he1;

    HEAD(he2) = faces[t_id].z;
    INFACE(he2) = t_id;
    NEXT(he2) = he0_id;
    TWIN(he2) = faces[t_id].x;
    halfedges[he2_id] = he2;

    tempHTI.x = HEAD(he0);
    tempHTI.y = faces[t_id].y;
    tempHTI.z = he0_id;
    HTI[he0_id] = tempHTI;

    tempHTI.x = HEAD(he1);
    tempHTI.y = faces[t_id].z;
    tempHTI.z = he1_id;
    HTI[he1_id] = tempHTI;

    tempHTI.x = HEAD(he2);
    tempHTI.y = faces[t_id].x;
    tempHTI.z = he2_id;
    HTI[he2_id] = tempHTI;
  }
}

/**
 * @brief The function to fill twin field of halfedges
 * 
 * @param num_halfedges The number of halfedges
 * @param halfedges The halfedges array
 * @param sorted_HTI The sorted array of head and tail and original index
 * @param frist_vertex_idx The array of first vertex index
 */
__global__ void fill_halfedges_twin(size_t num_verts, size_t num_halfedges, int4* halfedges, int3* sorted_HTI, uint* frist_vertex_idx)
{
  uint t_id = blockIdx.x * blockDim.x + threadIdx.x;
  if(t_id < num_halfedges)
  {
    int4 he = halfedges[t_id];
    int this_head = HEAD(he);
    int this_tail = TWIN(he);

    // find twin halfedge which is head and tail are reversed
    int start = frist_vertex_idx[this_tail];
    TWIN(halfedges[t_id]) = -1;
    for(int i=0; i<12; i++)
    {
      if(start+i >= num_halfedges) break;

      int3 temp = sorted_HTI[start+i];
      if(temp.y == this_head && temp.x == this_tail)
      {
        TWIN(halfedges[t_id]) = temp.z;
        break;
      }
    }
  }
}

__host__ void generate_halfedges(GlobalConstants &mesh)
{
  size_t num_faces = mesh.num_faces;
  size_t num_verts = mesh.out_num_verts;
  uint3* faces = mesh.faces;
  int3* HTI = mesh.halfedges_HTI;
  int4* halfedges = mesh.halfedges;

  dim3 threads_per_block(1024);
  dim3 num_blocks((num_faces + threads_per_block.x - 1) / threads_per_block.x);

  // 1. Construct halfedges
  construct_halfedges<<<num_blocks, threads_per_block>>>(halfedges, faces, num_faces, HTI);
  cudaDeviceSynchronize();
  
  // 2. Sort halfedges HTI, generate first vertex index array

  thrust::device_vector<int3> HTI_vec(HTI, HTI + 3 * num_faces);
  thrust::device_vector<uint> first_vertex(num_faces*3);

  thrust::sort(HTI_vec.begin(), HTI_vec.end(), comparehead());
  thrust::sequence(first_vertex.begin(), first_vertex.end());

  thrust::device_vector<int3> sorted_HTI_vec = HTI_vec;
  auto end = thrust::unique_by_key(sorted_HTI_vec.begin(), sorted_HTI_vec.end(), first_vertex.begin(), comparesame());
  first_vertex.erase(end.second, first_vertex.end());

  HTI = thrust::raw_pointer_cast(HTI_vec.data());
  mesh.hnt_first_vertex = thrust::raw_pointer_cast(first_vertex.data());

  num_blocks = dim3((3 * num_faces + threads_per_block.x - 1) / threads_per_block.x);


  // 3. Fill Twin Field
  fill_halfedges_twin<<<num_blocks, threads_per_block>>>(num_verts, 3 * num_faces, halfedges, HTI, mesh.hnt_first_vertex);
  cudaDeviceSynchronize();

  d_half_edges = thrust::device_vector<int4>(halfedges, halfedges + 3 * num_faces);
}


__host__ size_t deduplicateAndGenerateHalfedges()
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  GlobalConstants mesh;
  cudaMemcpyFromSymbol(&mesh, targetMesh, sizeof(GlobalConstants));

  // Vertex Deduplication
  // thrust::device_vector<float3> unique_vertices;
  thrust::device_vector<uint> new_vertex_map(mesh.num_verts);



  cudaEventRecord(start);
  out_num_verts = deduplicateVertices(mesh, unique_vertices, new_vertex_map);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("[Deduplication] Vertex Deduplication Time : %f ms\n", milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  mesh.out_num_verts = out_num_verts;
  // mesh.out_num_verts = mesh.num_verts;
  dim3 threads_per_block(512);
  dim3 num_blocks((mesh.num_faces + threads_per_block.x - 1) / threads_per_block.x);

  // Face Update
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  updateFaces<<<num_blocks, threads_per_block>>>(mesh.faces, mesh.num_faces, thrust::raw_pointer_cast(new_vertex_map.data()));

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("[Deduplication] Triangle Updating Time    : %f ms\n", milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaMalloc((void**)&mesh.halfedges, 3 * mesh.num_faces * sizeof(int4));
  cudaMalloc((void**)&mesh.halfedges_HTI, 3 * mesh.num_faces * sizeof(int3));
  cudaMalloc((void**)&mesh.hnt_first_vertex, mesh.out_num_verts * sizeof(uint));
  
  cudaMemset(mesh.halfedges, 0, 3 * mesh.num_faces * sizeof(int4));
  cudaMemset(mesh.halfedges_HTI, 0, 3 * mesh.num_faces * sizeof(int3));
  cudaMemset(mesh.hnt_first_vertex, 0, mesh.out_num_verts * sizeof(uint));

  cudaMemcpyToSymbol(targetMesh, &mesh, sizeof(GlobalConstants));
  
  // Halfedge Generation
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  generate_halfedges(mesh);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("[Half-edges] Generation Time              : %f ms\n", milliseconds);


  return out_num_verts;
}
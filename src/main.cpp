// 240306 - do 
// Test code for parallel vertex deduplication

#include <string>
#include <iostream>
#include <chrono>
#include <limits.h>
#include <unistd.h>
#include <signal.h>
#include "kernel.cuh"
#include "../include/host_deduplication.hpp"


bool validation = false;

struct mesh_t
{
  int num_vertices;
  int num_triangles;
  float3* vertices;
  uint3* triangles;
};


void ctrlc_handler(int s)
{
  // std::cout << "\n[signal] value : " << s << ", key : Ctrl + C" <<std::endl;
  ExitCUDA();
  std::cout << "\nProgram Terminated." << std::endl;
  exit(1);
}

/**
 * @brief Read mesh data from file and store it in a mesh_t struct and vectors for validation
 * 
 * @param filename mesh file name to read
 * @param vertices vector to store vertices, for validation
 * @param faces vector to store faces, for validation
 * @return mesh_t 
 */
mesh_t read_mesh(const std::string& filename, std::vector<float3> &vertices, std::vector<uint3> &faces)
{
  std::cout<< "Reading mesh data from file: " << filename << std::endl;
  auto start = std::chrono::system_clock::now();
  FILE* fp = fopen(filename.c_str(), "r");
  
  if (fp == NULL)
  {
    printf("Error: file not found\n");
    exit(-1);
  }
  
  int num_vertices, num_triangles;
  fscanf(fp, "%d %d\n", &num_vertices, &num_triangles);
  
  mesh_t mesh;
  mesh.num_vertices = num_vertices;
  mesh.num_triangles = num_triangles;
  std::cout << "[Input Data] Vertices: " << num_vertices << ", Triangles: " << num_triangles << std::endl;

  mesh.vertices = new float3[num_vertices];
  mesh.triangles = new uint3[num_triangles];


  vertices.reserve(num_vertices);
  faces.reserve(num_triangles);

  for(int i = 0; i < num_vertices; i++)
  {
    float x, y, z;
    fscanf(fp, "v %f %f %f\n", &x, &y, &z);

    float3 vertex {x, y, z};
    mesh.vertices[i] = vertex;
    vertices.push_back(vertex);
  }
  for(int i = 0; i < num_triangles; i++)
  {
    uint v0, v1, v2;
    fscanf(fp, "f %u %u %u\n", &v0, &v1, &v2);

    uint3 face {v0, v1, v2};
    mesh.triangles[i] = face;
    faces.push_back(face);
  }

  fclose(fp);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;

  std::cout<< "Done. << " << elapsed_seconds.count()*1000 << "ms"<< std::endl;
  return mesh;
}


void write_mesh(const std::string& filename, mesh_t input_mesh, thrust::host_vector<float3>& out_verts, thrust::host_vector<int4>& half_edges)
{
  std::cout<< "Writing output data to file: " << filename << std::endl;
  FILE* fp = fopen(filename.c_str(), "w");
  if (fp == NULL)
  {
    printf("Error: file not found\n");
    exit(-1);
  }

  size_t out_num_verts = out_verts.size();
  
  fprintf(fp, "%d %d\n", out_num_verts, input_mesh.num_triangles*3);
  for(int i = 0; i < out_num_verts; i++)
  {
    fprintf(fp, "%f %f %f\n", out_verts[i].x, out_verts[i].y, out_verts[i].z);
  }
  
  // write half edges
  for(int i = 0; i < input_mesh.num_triangles*3; i++)
  {
    fprintf(fp, "%d %d %d %d\n", half_edges[i].x, half_edges[i].y, half_edges[i].z, half_edges[i].w);
  }


  fclose(fp);
  std::cout<< "Done." << std::endl;
  std::cout<< "[ RESULT ]" << std::endl;
  std::cout<< "- INPUT  | Vertices: " << input_mesh.num_vertices << ", Triangles: " << input_mesh.num_triangles << std::endl;
  std::cout<< "- OUTPUT | Vertices: " << out_num_verts << ", Half-edges: " << input_mesh.num_triangles*3 << std::endl;
}


int main()
{
  signal(SIGINT, ctrlc_handler);

  std::vector<float3> vertices;
  std::vector<uint3> faces;

  InitCUDA();

  std::string filename = "/home/do/Desktop/do_code/mesh_deduplicator_cuda/data/sample_mesh_data2.txt";
  mesh_t mesh = read_mesh(filename, vertices, faces);

  // auto start = std::chrono::system_clock::now();
  // Copy mesh data from Host to device
  meshcopyHtoD(mesh.num_vertices,mesh.num_triangles,mesh.vertices,mesh.triangles);
  // auto chk1 = std::chrono::system_clock::now();


  // Deduplicate vertices
  size_t out_num_vertices = deduplicateAndGenerateHalfedges();
  printf("out_num_vertices: %d\n", out_num_vertices);
  thrust::host_vector<float3> out_vertices(out_num_vertices);
  thrust::host_vector<int4> half_edges(mesh.num_triangles*3);

  // auto chk2 = std::chrono::system_clock::now();
  meshcopyDtoH(out_vertices,half_edges);
  // auto end = std::chrono::system_clock::now();


  // std::chrono::duration<double> elapsed_seconds1 = chk1-start;
  // std::chrono::duration<double> elapsed_seconds2 = chk2-chk1;
  // std::chrono::duration<double> elapsed_seconds3 = end-chk2;
  // std::chrono::duration<double> elapsed_seconds = end-start;
  // std::cout<< "Deduplication and Half-edges Generation Time: " << elapsed_seconds1.count()*1000 << "ms" << std::endl;
  // std::cout<< "Copy HtoD Time: " << elapsed_seconds2.count()*1000 << "ms" << std::endl;
  // std::cout<< "Copy DtoH Time: " << elapsed_seconds3.count()*1000 << "ms" << std::endl;
  // std::cout<< "Total Time: " << elapsed_seconds.count()*1000 << "ms" << std::endl;


  // write_mesh("/home/do/Desktop/do_code/mesh_deduplicator_cuda/data/dragon_he_output.txt", mesh, out_vertices, half_edges);
  write_mesh("/home/do/Desktop/do_code/mesh_deduplicator_cuda/data/output.txt", mesh, out_vertices, half_edges);
  if(validation) host_deduplication(vertices, faces);

  std::cout<< "Press Ctrl + C to quit." << std::endl;

  while(1)
  {
    sleep(1);
  }
  return 0;
}
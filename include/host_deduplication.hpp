#pragma once

#include <vector>
#include <unordered_map>
#include <tuple>


/// for removing duplicated vertex
template <typename TT>
struct hash_tuple {
    size_t operator()(TT const& tt) const { return std::hash<TT>()(tt); }
};

namespace {

template <class T>
inline void hash_combine(std::size_t& hash_seed, T const& v) {
    hash_seed ^= std::hash<T>()(v) + 0x9e3779b9 + (hash_seed << 6) +
                 (hash_seed >> 2);
}

template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
struct HashValueImpl {
    static void apply(size_t& hash_seed, Tuple const& tuple) {
        HashValueImpl<Tuple, Index - 1>::apply(hash_seed, tuple);
        hash_combine(hash_seed, std::get<Index>(tuple));
    }
};

template <class Tuple>
struct HashValueImpl<Tuple, 0> {
    static void apply(size_t& hash_seed, Tuple const& tuple) {
        hash_combine(hash_seed, std::get<0>(tuple));
    }
};

}  // unnamed namespace
template <typename... TT>
struct hash_tuple<std::tuple<TT...>> {
    size_t operator()(std::tuple<TT...> const& tt) const {
        size_t hash_seed = 0;
        HashValueImpl<std::tuple<TT...>>::apply(hash_seed, tt);
        return hash_seed;
    }
};

void RemoveDuplicatedVertices(std::vector<float3> &vertices, std::vector<uint3> &triangles)
{
  typedef std::tuple<double, double, double> Coordinate3;
  std::unordered_map<Coordinate3, size_t, hash_tuple<Coordinate3>>vertex_to_old_idx;
  std::vector<int> old_idx_to_new_idx(vertices.size());
  size_t old_vertex_num = vertices.size();
  size_t k = 0;                                  // new index
  for (size_t i = 0; i < old_vertex_num; i++) {  // old index
    Coordinate3 coord = std::make_tuple(vertices[i].x, vertices[i].y,
                                        vertices[i].z);
    if (vertex_to_old_idx.find(coord) == vertex_to_old_idx.end()) {
      vertex_to_old_idx[coord] = i;
      vertices[k] = vertices[i];
      old_idx_to_new_idx[i] = (int)k;
      k++;
    } else {
      old_idx_to_new_idx[i] = old_idx_to_new_idx[vertex_to_old_idx[coord]];
    }
  }
  vertices.resize(k);
  if (k < old_vertex_num) {
    for (auto &triangle : triangles) {
      triangle.x = old_idx_to_new_idx[triangle.x];
      triangle.y = old_idx_to_new_idx[triangle.y];
      triangle.z = old_idx_to_new_idx[triangle.z];
    }
  }
}

void RemoveUnreferencedVertices(std::vector<float3> &vertices, std::vector<uint3> &triangles)
{
  std::vector<bool> vertex_has_reference(vertices.size(), false);
  for (const auto &triangle : triangles) {
    vertex_has_reference[triangle.x] = true;
    vertex_has_reference[triangle.y] = true;
    vertex_has_reference[triangle.z] = true;
  }
  std::vector<int> old_idx_to_new_idx(vertices.size());
  size_t old_vertex_num = vertices.size();
  size_t k = 0;                                  // new index
  for (size_t i = 0; i < old_vertex_num; i++) {  // old index
    if (vertex_has_reference[i]) {
      vertices[k] = vertices[i];
      old_idx_to_new_idx[i] = (int)k;
      k++;
    } else {
      old_idx_to_new_idx[i] = -1;
    }
  }
  vertices.resize(k);
  if (k < old_vertex_num) {
    for (auto &triangle : triangles) {
      triangle.x = old_idx_to_new_idx[triangle.x];
      triangle.y = old_idx_to_new_idx[triangle.y];
      triangle.z = old_idx_to_new_idx[triangle.z];
    }
  }
}

/**
 * @brief The validation function for the vertex deduplication on host
 * 
 * @param vertices vector of vertices
 * @param faces vector of faces
 */
void host_deduplication(std::vector<float3> &vertices, std::vector<uint3> &faces)
{
  std::cout<< "Deduplicating Vertices..." << std::endl;
  auto start = std::chrono::system_clock::now();
  RemoveDuplicatedVertices(vertices, faces);
  RemoveUnreferencedVertices(vertices, faces);
  std::cout << "[Host Output] Vertices: " << vertices.size() << ", Triangles: " << faces.size() << std::endl;
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout<< "Done. << " << elapsed_seconds.count()*1000 << "ms"<< std::endl;

  FILE* fp = fopen("host_deduplication_output.txt", "w");
  fprintf(fp, "%d %d\n", vertices.size(), faces.size());
  for(int i = 0; i < vertices.size(); i++)
  {
    fprintf(fp, "v %f %f %f\n", vertices[i].x, vertices[i].y, vertices[i].z);
  }
  for(int i = 0; i < faces.size(); i++)
  {
    fprintf(fp, "f %d %d %d\n", faces[i].x, faces[i].y, faces[i].z);
  }
  fclose(fp);
  std::cout<< "Output written to host_deduplication_output.txt" << std::endl;
}


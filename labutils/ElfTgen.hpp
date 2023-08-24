#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace elf
{
	namespace tgen
	{
		using real_t = float;
		using index_t = std::uint32_t;
		const real_t eps = real_t(1e-8);

		//-------------------------------------------------------------------------

		struct TangentCaculation
		{
			std::vector<glm::vec3> ctangents3d;
			std::vector<glm::vec3> cbitangents3d;
			std::vector<glm::vec3> tangents3d;
			std::vector<glm::vec3> bitangents3d;

			TangentCaculation() = default;
		};

		inline void computeCornerTSpace(
			TangentCaculation& ret,
			const std::vector<uint32_t>& indices,
			const std::vector<glm::vec3>& positions3D,
			const std::vector<glm::vec2>& uvs2D)
		{
			const auto& triangleCounts = indices.size();

			// std::vector<glm::vec3> cTangents3D;
			ret.ctangents3d.resize(triangleCounts);
			// std::vector<glm::vec3> cBitangents3D;
			ret.cbitangents3d.resize(triangleCounts);

			std::array<glm::vec3, 3> edge3D;
			std::array<glm::vec2, 3> edgeUV;

			for (std::size_t i = 0; i < triangleCounts; i += 3)
			{
				const glm::vec3 vertexIndicesPos = glm::vec3(indices[i], indices[i + 1], indices[i + 2]);

				// compute derivatives of positions and UVs along the edges
				for (std::size_t j = 0; j < 3; ++j)
				{
					const std::size_t next = (j + 1) % 3;

					const size_t curr_idx = indices[i + j];
					const size_t next_idx = indices[i + next];

					edge3D[j] = positions3D[next_idx] - positions3D[curr_idx];
					edgeUV[j] = uvs2D[next_idx] - uvs2D[curr_idx];
				}

				// compute per-corner tangent and bitangent (not normalized),
				// using the derivatives of the UVs
				// http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-13-normal-mapping/
				for (std::size_t j = 0; j < 3; ++j)
				{
					const std::size_t prev = (j + 2) % 3;

					const glm::vec3 dPos0 = edge3D[j];
					const glm::vec3 dPos1Neg = edge3D[prev];
					const glm::vec2 dUV0 = edgeUV[j];
					const glm::vec2 dUV1Neg = edgeUV[prev];

					real_t denom = (dUV0[0] * -dUV1Neg[1] - dUV0[1] * -dUV1Neg[0]);
					real_t r = (real_t)std::fabs(denom) > eps ? (real_t)1.0 / (real_t)denom : (real_t)0.0;

					glm::vec3 tmp0 = dPos0 * (-dUV1Neg[1] * r);
					glm::vec3 tmp1 = dPos1Neg * (-dUV0[1] * r);

					ret.ctangents3d[i + j] = tmp0 - tmp1;

					tmp0 = dPos1Neg * (-dUV0[0] * r);
					tmp1 = dPos0 * (-dUV1Neg[0] * r);

					ret.cbitangents3d[i + j] = tmp0 - tmp1;
				}
			}
		}

		//-------------------------------------------------------------------------
		inline void computeVertexTSpace(TangentCaculation& ret,
		                                const std::vector<uint32_t>& triIndicesUV,
		                                std::size_t numUVVertices)
		{
			// std::vector<glm::vec3> vTangents3D;
			ret.tangents3d.resize(numUVVertices);

			// std::vector<glm::vec3> vBitangents3D;
			ret.bitangents3d.resize(numUVVertices);

			// average tangent vectors for each "wedge" (UV vertex)
			// this assumes that we do not use different vertex positions
			// for the same UV coordinate (example: mirrored parts)

			for (std::size_t i = 0; i < triIndicesUV.size(); ++i)
			{
				const auto& uvIdx = triIndicesUV[i];

				ret.tangents3d[uvIdx] += ret.ctangents3d[i];
				ret.bitangents3d[uvIdx] += ret.cbitangents3d[i];
			}

			// normalize results
			for (uint32_t i = 0; i < numUVVertices; ++i)
			{
				if (glm::length(ret.tangents3d[i]) > eps)
				{
					glm::normalize(ret.tangents3d[i]);
				}
				if (glm::length(ret.bitangents3d[i]) > eps)
				{
					glm::normalize(ret.bitangents3d[i]);
				}
			}
		}

		//-------------------------------------------------------------------------

		inline void orthogonalizeTSpace(TangentCaculation& ret, const std::vector<glm::vec3>& normals3D)
		{
			// Gram-Schmidt
			for (uint32_t i = 0; i < normals3D.size(); ++i)
			{
				real_t d = glm::dot(normals3D[i], ret.tangents3d[i]);

				glm::vec3 correction = normals3D[i] * d;
				ret.tangents3d[i] = glm::normalize(ret.tangents3d[i] - correction);
				ret.bitangents3d[i] = glm::cross(normals3D[i], ret.tangents3d[i]);
			}

			// Mirror
			for (uint32_t i = 0; i < normals3D.size(); ++i)
			{
				glm::vec3 cross = glm::cross(normals3D[i], ret.tangents3d[i]);
				real_t sign = (real_t)glm::dot(cross, ret.bitangents3d[i]) > (real_t)0.0 ? (real_t)1.0 : (real_t)-1.0;
				if (sign < 0)
				{
					ret.tangents3d[i] *= -1.0f;
				}
			}
		}

		inline std::vector<glm::vec3> CalculateTangents(
			const std::vector<uint32_t>& indices,
			const std::vector<glm::vec3>& vertices,
			const std::vector<glm::vec3>& normals,
			const std::vector<glm::vec2>& uvs)
		{
			TangentCaculation ret;
			computeCornerTSpace(ret, indices, vertices, uvs);
			computeVertexTSpace(ret, indices, vertices.size());
			orthogonalizeTSpace(ret, normals);

			return ret.tangents3d;
		}

		inline std::vector<glm::quat> CalculateTBNQuats(
			const std::vector<uint32_t>& indices,
			const std::vector<glm::vec3>& vertices,
			const std::vector<glm::vec3>& normals,
			const std::vector<glm::vec2>& uvs)
		{
			TangentCaculation ret;
			computeCornerTSpace(ret, indices, vertices, uvs);
			computeVertexTSpace(ret, indices, vertices.size());
			orthogonalizeTSpace(ret, normals);

			std::vector<glm::quat> tbnQuaternions{};
			for (size_t n = 0; n < normals.size(); ++n)
			{
				glm::mat3 tbnMatrix(ret.tangents3d[n], ret.bitangents3d[n], normals[n]);
				tbnQuaternions.emplace_back(glm::normalize(glm::quat_cast(tbnMatrix)));
			}

			return tbnQuaternions;
		}
	} //namespace tgen
}

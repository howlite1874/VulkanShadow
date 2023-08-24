#include "vertex_data.hpp"

#include <limits>

#include <cstring> // for std::memcpy()

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/to_string.hpp"
#include "../labutils/vkbuffer.hpp"

#include "glm/glm.hpp"

namespace lut = labutils;


objMesh create_mesh(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator, BakedMeshData aMesh)
{

	objMesh tmpMesh;

	//position
	lut::Buffer VertexPosGPU = create_buffer(
		aAllocator,
		aMesh.positions.size() * sizeof(glm::vec3),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		//VMA_MEMORY_USAGE_CPU_TO_GPU:This indicates that VMA should try to use device local memory for
		//the on - GPU buffer whenever possible
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	//normal
	lut::Buffer VertexNormGPU =create_buffer(
		aAllocator,
		aMesh.normals.size() * sizeof(glm::vec3),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		//VMA_MEMORY_USAGE_CPU_TO_GPU:This indicates that VMA should try to use device local memory for
		//the on - GPU buffer whenever possible
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	//texcoords
	lut::Buffer VertexTexGPU = create_buffer(
		aAllocator,
		aMesh.texcoords.size() * sizeof(glm::vec2),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);


	//tangents
	lut::Buffer VertexTanGPU = create_buffer(
		aAllocator,
		aMesh.tangents.size() * sizeof(glm::vec3),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	//indices
	lut::Buffer VertexIndGPU = create_buffer(
		aAllocator,
		aMesh.indices.size() * sizeof(std::uint32_t),
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	//-----------------------------------------------------------------------------------------------------------
	//position
	lut::Buffer posStaging = create_buffer(
		aAllocator,
		aMesh.positions.size() * sizeof(glm::vec3),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);

	//normal
	lut::Buffer normStaging =create_buffer(
		aAllocator,
		aMesh.normals.size() * sizeof(glm::vec3),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);

	//texcoords
	lut::Buffer texStaging = create_buffer(
		aAllocator,
		aMesh.texcoords.size() * sizeof(glm::vec2),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);

	//tangents
	lut::Buffer tanStaging = create_buffer(
		aAllocator,
		aMesh.tangents.size() * sizeof(glm::vec3),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);

	//indices
	lut::Buffer indStaging =create_buffer(
		aAllocator,
		aMesh.indices.size() * sizeof(std::uint32_t),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);

	//-------------------------------------------------------------------------------------------------------
	void* posPtr = nullptr;
	if (const auto res = vmaMapMemory(aAllocator.allocator, posStaging.allocation, &posPtr);
		VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n"
			"vmaMapMemory() returned %s", lut::to_string(res).c_str()
		);
	}
	std::memcpy(posPtr, aMesh.positions.data(), aMesh.positions.size() * sizeof(glm::vec3));
	vmaUnmapMemory(aAllocator.allocator, posStaging.allocation);


	void* normPtr = nullptr;
	if (const auto res = vmaMapMemory(aAllocator.allocator, normStaging.allocation, &normPtr);
		VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n"
			"vmaMapMemory() returned %s", lut::to_string(res).c_str()
		);
	}
	std::memcpy(normPtr, aMesh.normals.data(), aMesh.normals.size() * sizeof(glm::vec3));
	vmaUnmapMemory(aAllocator.allocator, normStaging.allocation);

	void* texPtr = nullptr;
	if (const auto res = vmaMapMemory(aAllocator.allocator, texStaging.allocation, &texPtr);
		VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n"
			"vmaMapMemory() returned %s", lut::to_string(res).c_str()
		);
	}
	std::memcpy(texPtr, aMesh.texcoords.data(), aMesh.texcoords.size() * sizeof(glm::vec2));
	vmaUnmapMemory(aAllocator.allocator, texStaging.allocation);

	void* tanPtr = nullptr;
	if (const auto res = vmaMapMemory(aAllocator.allocator, tanStaging.allocation, &tanPtr);
		VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n"
			"vmaMapMemory() returned %s", lut::to_string(res).c_str()
		);
	}
	std::memcpy(tanPtr, aMesh.tangents.data(), aMesh.tangents.size() * sizeof(glm::vec3));
	vmaUnmapMemory(aAllocator.allocator, tanStaging.allocation);

	void* indPtr = nullptr;
	if (const auto res = vmaMapMemory(aAllocator.allocator, indStaging.allocation, &indPtr);
		VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n"
			"vmaMapMemory() returned %s", lut::to_string(res).c_str()
		);
	}
	std::memcpy(indPtr, aMesh.indices.data(), aMesh.indices.size() * sizeof(std::uint32_t));
	vmaUnmapMemory(aAllocator.allocator, indStaging.allocation);

	//-----------------------------------------------------------------------------------------------------------
	//prepare for issuing the transfer commands that copy data from the staging buffers to
	//the final on - GPU buffers
	lut::Fence uploadComplete = create_fence(aContext);

	// Queue data uploads from staging buffers to the final buffers 
	// This uses a separate command pool for simplicity. 
	lut::CommandPool uploadPool = lut::create_command_pool(aContext);
	VkCommandBuffer uploadCmd = lut::alloc_command_buffer(aContext, uploadPool.handle);

	//record copy commands into command buffer
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = 0;
	beginInfo.pInheritanceInfo = nullptr;

	if (const auto res = vkBeginCommandBuffer(uploadCmd, &beginInfo);
		VK_SUCCESS != res)
	{
		throw lut::Error("Beginning command buffer recording\n"
			"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str()
		);
	}

	//position
	VkBufferCopy pcopy{};
	pcopy.size = aMesh.positions.size() * sizeof(glm::vec3);

	vkCmdCopyBuffer(uploadCmd, posStaging.buffer, VertexPosGPU.buffer, 1, &pcopy);

	lut::buffer_barrier(uploadCmd,
		VertexPosGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	//normal
	VkBufferCopy ncopy{};
	ncopy.size = aMesh.normals.size() * sizeof(glm::vec3);

	vkCmdCopyBuffer(uploadCmd, normStaging.buffer, VertexNormGPU.buffer, 1, &ncopy);

	lut::buffer_barrier(uploadCmd,
		VertexNormGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	//texcoords
	VkBufferCopy tcopy{};
	tcopy.size = aMesh.texcoords.size() * sizeof(glm::vec2);

	vkCmdCopyBuffer(uploadCmd, texStaging.buffer, VertexTexGPU.buffer, 1, &tcopy);

	lut::buffer_barrier(uploadCmd,
		VertexTexGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	//tangents
	VkBufferCopy tancopy{};
	tancopy.size = aMesh.tangents.size() * sizeof(glm::vec3);

	vkCmdCopyBuffer(uploadCmd, tanStaging.buffer, VertexTanGPU.buffer, 1, &tancopy);

	lut::buffer_barrier(uploadCmd,
		VertexTanGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	//indices
	VkBufferCopy icopy{};
	icopy.size = aMesh.indices.size() * sizeof(std::uint32_t);

	vkCmdCopyBuffer(uploadCmd, indStaging.buffer, VertexIndGPU.buffer, 1, &icopy);

	lut::buffer_barrier(uploadCmd,
		VertexIndGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);


	if (const auto res = vkEndCommandBuffer(uploadCmd);
		VK_SUCCESS != res)
	{
		throw lut::Error("Ending command buffer recording\n"
			"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str()
		);
	}

	//submit transfer commands
	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &uploadCmd;

	if (const auto res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadComplete.handle);
		VK_SUCCESS != res)
	{
		throw lut::Error("Submitting commands\n"
			"vkQueueSubmit() returned %s", lut::to_string(res).c_str()
		);
	}

	// Wait for commands to finish before we destroy the temporary resources 
	// required for the transfers (staging buffers, command pool, ...)
	//
	// The code doesn¡¯t destory the resources implicitly ¨C the resources are 
	// destroyed by the destructors of the labutils wrappers for the various 
	// objects once we leave the function¡¯s scope.
	if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle,
		VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
	{
		throw lut::Error("Waiting for upload to complete\n"
			"vkWaitForFences() returned %s", lut::to_string(res).c_str()
		);
	}

	tmpMesh.positions = std::move(VertexPosGPU);
	tmpMesh.normals = std::move(VertexNormGPU);
	tmpMesh.texcoords = std::move(VertexTexGPU);
	tmpMesh.tangents= std::move(VertexTanGPU);
	tmpMesh.indices = std::move(VertexIndGPU);
	return tmpMesh;
}

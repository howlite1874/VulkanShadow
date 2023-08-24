#pragma once

#include <cstdint>
#include <vector> 

#include "../labutils/vulkan_context.hpp"

#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp"
#include "../cw4/baked_model.hpp"

struct objMesh
{
	labutils::Buffer positions;
	labutils::Buffer texcoords;
	labutils::Buffer normals;
	labutils::Buffer tangents;
	labutils::Buffer indices;
	labutils::Buffer packedTBN;

};


objMesh create_mesh(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator, BakedMeshData aModel);


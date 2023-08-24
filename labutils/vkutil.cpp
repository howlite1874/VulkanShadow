#include "vkutil.hpp"

#include <vector>

#include <cstdio>
#include <cassert>

#include "error.hpp"
#include "to_string.hpp"

namespace labutils
{
	ShaderModule load_shader_module( VulkanContext const& aContext, char const* aSpirvPath )
	{
		//todo:doing
		assert(aSpirvPath);
		if(std::FILE* fin = std::fopen(aSpirvPath,"rb"))
		{
			/*	fseek and ftell to determine the size of a file opened in binary mode using fin file pointer overall,
			these three lines of code are used to determine the size of a file by moving the file position indicator
			to the end of the file, getting the current position using ftell, and then moving the file position indicator
			back to the beginning of the file. This information about the file size can be useful in various file operations.*/
			std::fseek(fin, 0, SEEK_END);
			auto const bytes = std::size_t(std::ftell(fin));
			std::fseek(fin, 0, SEEK_SET);

			//SPIR-V consists of a number of 32-bit = 4 byte words
			assert(0 == bytes % 4);
			auto const words = bytes / 4;

			std::vector<std::uint32_t> code(words);

			std::size_t offset = 0;
			while(offset != words)
			{
				//code.data() + offset : starting address of the buffer where the data will be read
				//sizeof(std::uint32_t) : specifies the size of each element to be read in bytes
				//words - offset : specifies the maximum number of elements to be read
				//fin :  is the file pointer to the binary file being read from
				auto const read = std::fread(code.data() + offset, sizeof(std::uint32_t), words - offset, fin);
				if(0 == read)
				{
					std::fclose(fin);

					throw Error("Error reading ¡¯%s¡¯: ferror = %d, feof = %d", aSpirvPath, std::ferror(fin), std::feof(fin));
				}
				offset += read;
			}
			std::fclose(fin);

			VkShaderModuleCreateInfo moduleInfo{
				VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
				nullptr,
				0,
				bytes,
				code.data()
			};

			VkShaderModule smod = VK_NULL_HANDLE;
			if(auto const res=vkCreateShaderModule(aContext.device,&moduleInfo,nullptr,&smod);
				VK_SUCCESS!=res)
			{
				throw Error("Unable to create shader module from %s\n"
					"vkCreateShaderModule() returned %s", aSpirvPath, to_string(res).c_str()
				);
			}
			return ShaderModule(aContext.device, smod);
		}
		throw Error("Cannont open ¡¯%s¡¯ for reading", aSpirvPath);
	}


	CommandPool create_command_pool( VulkanContext const& aContext, VkCommandPoolCreateFlags aFlags )
	{
		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.queueFamilyIndex = aContext.graphicsFamilyIndex;
		poolInfo.flags = aFlags;

		VkCommandPool cpool = VK_NULL_HANDLE;
		if(const auto res=vkCreateCommandPool(aContext.device,&poolInfo,nullptr,&cpool);
			VK_SUCCESS!=res)
		{
			throw Error("Unable to create command pool\n"
				"vkCreateCommandPool() returned %s", to_string(res).c_str()
			);
		}

		return CommandPool(aContext.device, cpool);
	}

	//We use vkAllocateCommandBuffers to allocate a command buffer from the provided command pool.
	VkCommandBuffer alloc_command_buffer( VulkanContext const& aContext, VkCommandPool aCmdPool )
	{
		//TODO:HAVE DONE
		VkCommandBufferAllocateInfo cbufInfo{};
		cbufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cbufInfo.commandPool = aCmdPool;
		cbufInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cbufInfo.commandBufferCount = 1;

		VkCommandBuffer cbuff = VK_NULL_HANDLE;
		if(auto const res=vkAllocateCommandBuffers(aContext.device,&cbufInfo,&cbuff);
			VK_SUCCESS!=res)
		{
			throw Error("Unable to allocate command buffer\n" 
				"vkAllocateCommandBuffers() returned %s", to_string(res).c_str() 
			);
		}
		//We are not required to free command buffers individually, as they will be freed
		//automatically when the parent command pool is destroyed so we don't need to wrap it
		return cbuff;
	}

	//we need a VkFence in order to be able to wait until all commands have finished executing
	Fence create_fence( VulkanContext const& aContext, VkFenceCreateFlags aFlags )
	{
		//TODO:HAVE DONE
		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = aFlags;

		VkFence fence = VK_NULL_HANDLE;
		if(auto const res=vkCreateFence(aContext.device,&fenceInfo,nullptr,&fence);
			VK_SUCCESS!=res)
		{
			throw Error("Unable to create fence\n" 
				"vkCreateFence() returned %s", to_string(res).c_str() 
			);
		}

		return Fence(aContext.device, fence);
	}

	Semaphore create_semaphore( VulkanContext const& aContext )
	{
		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkSemaphore semaphore = VK_NULL_HANDLE;
		if(const auto res = vkCreateSemaphore(aContext.device,&semaphoreInfo,nullptr,&semaphore);VK_SUCCESS!=res)
		{
			throw Error("Unable to create semaphore\n" 
				"vkCreateSemaphore() returned %s", to_string(res).c_str() 
				);
		}

		return Semaphore(aContext.device, semaphore);
	}

	void buffer_barrier(
		VkCommandBuffer aCmdBuff,
		VkBuffer aBuffer,
		VkAccessFlags aSrcAccessMask,
		VkAccessFlags aDstAccessMask,
		VkPipelineStageFlags aSrcStageMask,
		VkPipelineStageFlags aDstStageMask,
		VkDeviceSize aSize,
		VkDeviceSize aOffset,
		uint32_t aSrcQueueFamilyIndex,
		uint32_t ADstQueueFamilyIndex
	)
	{
		VkBufferMemoryBarrier bbarrier{};
		bbarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		bbarrier.srcAccessMask = aSrcAccessMask;
		bbarrier.dstAccessMask = aDstAccessMask;
		bbarrier.buffer = aBuffer;
		bbarrier.size = aSize;
		bbarrier.offset = aOffset;
		bbarrier.srcQueueFamilyIndex = aSrcQueueFamilyIndex;
		bbarrier.dstQueueFamilyIndex = ADstQueueFamilyIndex;

		vkCmdPipelineBarrier(
			aCmdBuff,
			aSrcStageMask, aDstStageMask,
			0,
			0, nullptr,
			1, &bbarrier,
			0, nullptr
		);
	}
	void image_barrier(
		VkCommandBuffer aCmdBuff,
		VkImage aImage,
		VkAccessFlags aSrcAccessMask,
		VkAccessFlags aDstAccessMask,
		VkImageLayout aSrcLayout,
		VkImageLayout aDstLayout,
		VkPipelineStageFlags aSrcStageMask,
		VkPipelineStageFlags aDstStageMask,
		VkImageSubresourceRange aRange,
		std::uint32_t aSrcQueueFamilyIndex,
		std::uint32_t aDstQueueFamilyIndex
	)
	{
		VkImageMemoryBarrier ibarrier{};
		ibarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		ibarrier.image = aImage;
		ibarrier.srcAccessMask = aSrcAccessMask;
		ibarrier.dstAccessMask = aDstAccessMask;
		ibarrier.srcQueueFamilyIndex = aSrcQueueFamilyIndex;
		ibarrier.dstQueueFamilyIndex = aDstQueueFamilyIndex;
		ibarrier.oldLayout = aSrcLayout;
		ibarrier.newLayout = aDstLayout;
		ibarrier.subresourceRange = aRange;

		vkCmdPipelineBarrier(aCmdBuff, aSrcStageMask, aDstStageMask, 0, 0, nullptr, 0, nullptr, 1, &ibarrier);
	}

	DescriptorPool create_descriptor_pool(VulkanContext const& aContext, std::uint32_t aMaxDescriptors, std::uint32_t aMaxSets)
	{
		VkDescriptorPoolSize const pools[] = {
			{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,aMaxDescriptors},
			{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,aMaxDescriptors},
			{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,aMaxDescriptors},
		};

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.maxSets = aMaxSets;
		poolInfo.poolSizeCount = sizeof(pools) / sizeof(pools[0]);
		poolInfo.pPoolSizes = pools;
		poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT; 


		VkDescriptorPool pool = VK_NULL_HANDLE;
		if(const auto& res = vkCreateDescriptorPool(aContext.device,&poolInfo,nullptr,&pool);VK_SUCCESS!=res)
		{
			throw Error("Unable to create descriptor pool\n"
				"vkCreateDescriptorPool() returned %s", to_string(res).c_str()
			);
		}


		return DescriptorPool(aContext.device,pool);
	}

	VkDescriptorSet alloc_desc_set(VulkanContext const& aContext, VkDescriptorPool aPool, VkDescriptorSetLayout aSetLayout)
	{
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = aPool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &aSetLayout;

		VkDescriptorSet dset = VK_NULL_HANDLE;
		if (auto const res = vkAllocateDescriptorSets(aContext.device, &allocInfo, &dset); VK_SUCCESS != res)
		{
			throw Error("Unable to allocate descriptor set\n"
				"vkAllocateDescriptorSets() returned %s", to_string(res).c_str()
			);
		}

		return dset;
	}

	Sampler create_default_sampler(VulkanContext const& aContext)
	{
		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.anisotropyEnable = VK_FALSE;
		samplerInfo.minLod = 0.f;
		samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
		//samplerInfo.maxLod = static_cast<float>(mipLevels);
		samplerInfo.mipLodBias = 0.f;


		VkSampler sampler = VK_NULL_HANDLE;
		if(const auto res = vkCreateSampler(aContext.device,&samplerInfo,nullptr,&sampler);VK_SUCCESS!=res)
		{
			throw Error("Unable to create sampler\n"
				"vkCreateSampler() returned %s", to_string(res).c_str()
			);
		}

		return Sampler(aContext.device, sampler);
		
	}

	Sampler create_cubemap_sampler(VulkanContext const& aContext)
	{
		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		samplerInfo.anisotropyEnable = VK_TRUE;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.maxAnisotropy = 1.0f;
		samplerInfo.minLod = 0.f;
		samplerInfo.maxLod = 1.0f;
		samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;

		VkSampler sampler = VK_NULL_HANDLE;
		if (const auto res = vkCreateSampler(aContext.device, &samplerInfo, nullptr, &sampler); VK_SUCCESS != res)
		{
			throw Error("Unable to create sampler\n"
				"vkCreateSampler() returned %s", to_string(res).c_str()
			);
		}

		return Sampler(aContext.device, sampler);

	}

	ImageView create_image_view_texture2d(VulkanContext const& aContext, VkImage aImage, VkFormat aFormat)
	{
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = aImage;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = aFormat;
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_COLOR_BIT,
			0,VK_REMAINING_MIP_LEVELS,
			0, 1
		};

		VkImageView view = VK_NULL_HANDLE;
		if (const auto& res = vkCreateImageView(aContext.device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
		{
			throw Error("Unable to create image view\n"
				"vkCreateImageView() returned %s", to_string(res).c_str()
			);
		}

		return ImageView(aContext.device, view);
	}

	std::array<ImageView, 6> create_cubemap_image_view(VulkanContext const& aContext,VkImage img, VkFormat aFormat)
	{
		std::array<ImageView, 6> cubemapView;
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = VK_NULL_HANDLE;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
		viewInfo.format = aFormat;
		viewInfo.components = { VK_COMPONENT_SWIZZLE_R };
		viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		viewInfo.subresourceRange.layerCount = 6;
		viewInfo.image = img;
		VkImageView viewContainer = VK_NULL_HANDLE;
		if (const auto& res = vkCreateImageView(aContext.device, &viewInfo, nullptr, &viewContainer); VK_SUCCESS != res)
		{
			throw Error("Unable to create image view\n"
				"vkCreateImageView() returned %s", to_string(res).c_str()
			);
		}

		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.subresourceRange.layerCount = 1;
		viewInfo.image = img;

		for (uint32_t i = 0; i < 6; i++) {
			VkImageView view = VK_NULL_HANDLE;
			if (const auto& res = vkCreateImageView(aContext.device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
			{
				throw Error("Unable to create image view\n"
					"vkCreateImageView() returned %s", to_string(res).c_str()
				);
			}
			cubemapView[i] = ImageView(aContext.device, view);
		}

		return cubemapView;
	}

	void flushCommandBuffer(VulkanContext const& aContext,VkCommandBuffer commandBuffer, VkQueue queue, VkCommandPool pool, bool free)
	{
		if (commandBuffer == VK_NULL_HANDLE)
		{
			return;
		}

		if (const auto& res = vkEndCommandBuffer(commandBuffer); VK_SUCCESS != res)
		{
			throw Error("Unable to end command\n"
				"vkEndCommandBuffer() returned %s", to_string(res).c_str()
			);
		}

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		// Create fence to ensure that the command buffer has finished executing
		Fence fence= create_fence(aContext, 0);
		
		// Submit to the queue
		if (const auto& res = vkQueueSubmit(queue, 1, &submitInfo, fence.handle); VK_SUCCESS != res)
		{
			throw Error("Unable to submit queue\n"
				"vkQueueSubmit() returned %s", to_string(res).c_str()
			);
		}

		// Wait for the fence to signal that command buffer has finished executing
		if (const auto& res = (vkWaitForFences(aContext.device, 1, &fence.handle, VK_TRUE, 100000000000)); VK_SUCCESS != res)
		{
			throw Error("Unable to wait fences\n"
				"vkWaitForFences() returned %s", to_string(res).c_str()
			);
		}

		vkDestroyFence(aContext.device, fence.handle, nullptr);
		if (free)
		{
			vkFreeCommandBuffers(aContext.device, pool, 1, &commandBuffer);
		}
	}
}

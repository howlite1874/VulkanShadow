#pragma once

#include <volk/volk.h>
#include <vk_mem_alloc.h>

#include <utility>

#include <cassert>

#include "allocator.hpp"

namespace labutils
{
	class Image
	{
		public:
			Image() noexcept, ~Image();

			explicit Image( VmaAllocator, VkImage = VK_NULL_HANDLE, VmaAllocation = VK_NULL_HANDLE ) noexcept;

			Image( Image const& ) = delete;
			Image& operator= (Image const&) = delete;

			Image( Image&& ) noexcept;
			Image& operator = (Image&&) noexcept;

		public:
			VkImage image = VK_NULL_HANDLE;
			VmaAllocation allocation = VK_NULL_HANDLE;

		private:
			VmaAllocator mAllocator = VK_NULL_HANDLE;
	};

	Image default_normal_texture(VulkanContext const& aContext, VkCommandPool aCmdPool, Allocator const& aAllocator);

	Image load_image_texture2d( char const* aPath, VulkanContext const&, VkCommandPool, Allocator const& );

	Image create_image_texture2d( Allocator const&, std::uint32_t aWidth, std::uint32_t aHeight, VkFormat, VkImageUsageFlags = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT );

	Image create_depth_texture2d(Allocator const& aAllocator, std::uint32_t aWidth, std::uint32_t aHeight, VkFormat aFormat, VkImageUsageFlags aUsage);

	std::uint32_t compute_mip_level_count( std::uint32_t aWidth, std::uint32_t aHeight );

	void setImageLayout(VkCommandBuffer cmdbuffer, VkImage image, VkImageLayout oldImageLayout, VkImageLayout newImageLayout, VkImageSubresourceRange subresourceRange, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask);
}

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/allocator.h"

#include <array>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/hal/memory/slab_provider.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::vulkan {
namespace {

constexpr VkDeviceSize kMiB = 1024 * 1024;
constexpr VkMemoryPropertyFlags kPrivate = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
constexpr VkMemoryPropertyFlags kBar = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
constexpr VkMemoryPropertyFlags kHost = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                        VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

// Exercise the real allocator and pools against synthetic memory topologies,
// recording the VkMemoryAllocateInfo rather than allocating GPU memory.
class AllocatorTest : public ::testing::Test {
 protected:
  struct Buffer {
    VkDeviceSize size;
  };
  struct Memory {
    uint32_t heap_index;
    VkDeviceSize size;
  };

  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
    snapshot_.properties2.properties.limits.nonCoherentAtomSize = 256;
    snapshot_.properties11.maxMemoryAllocationSize = 4ull * 1024 * kMiB;
    syms_.vkCreateBuffer = CreateBuffer;
    syms_.vkDestroyBuffer = DestroyBuffer;
    syms_.vkGetBufferMemoryRequirements = GetBufferMemoryRequirements;
    syms_.vkAllocateMemory = AllocateMemory;
    syms_.vkFreeMemory = FreeMemory;
    syms_.vkBindBufferMemory = BindBufferMemory;
  }

  void TearDown() override {
    for (auto* buffer : buffers_) iree_hal_buffer_release(buffer);
    iree_hal_allocator_release(allocator_);
    EXPECT_EQ((std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS>{}), heap_usage_);
    iree_async_proactor_release(proactor_);
  }

  void AddMemoryType(VkMemoryPropertyFlags flags, VkDeviceSize heap_size) {
    auto& properties = snapshot_.memory_properties2.memoryProperties;
    const uint32_t index = properties.memoryTypeCount++;
    properties.memoryTypes[index].propertyFlags = flags;
    properties.memoryTypes[index].heapIndex = properties.memoryHeapCount;
    properties.memoryHeaps[properties.memoryHeapCount++].size = heap_size;
  }

  iree_status_t CreateAllocator() {
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_allocator_create(
        reinterpret_cast<iree_hal_device_t*>(this), &syms_,
        reinterpret_cast<VkDevice>(this), &snapshot_,
        IREE_HAL_VULKAN_FEATURE_NONE, IREE_HAL_VULKAN_DEVICE_EXTENSION_NONE,
        IREE_HAL_QUEUE_AFFINITY_ANY,
        /*sparse_binding_queue=*/nullptr, proactor_, iree_allocator_system(),
        &allocator_));
    allocation_types_.clear();
    return iree_ok_status();
  }

  static iree_hal_buffer_params_t DeviceParams() {
    iree_hal_buffer_params_t params = {};
    params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
    iree_hal_buffer_params_canonicalize(&params);
    return params;
  }

  iree_status_t Allocate(iree_hal_buffer_params_t params,
                         iree_device_size_t size, bool direct,
                         iree_hal_buffer_t** out_buffer) {
    iree_status_t status =
        direct ? iree_hal_vulkan_allocator_allocate_direct_buffer(
                     allocator_, &params, size, out_buffer)
               : iree_hal_allocator_allocate_buffer(allocator_, params, size,
                                                    out_buffer);
    if (iree_status_is_ok(status)) buffers_.push_back(*out_buffer);
    return status;
  }

  iree_status_t SelectQueuePool(iree_hal_buffer_params_t params,
                                iree_hal_memory_type_t* out_type) {
    iree_device_size_t size = 4096;
    iree_hal_vulkan_queue_alloca_plan_t plan = {};
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_allocator_select_queue_alloca_plan(
        allocator_, /*requested_pool=*/nullptr, &params, &size, &plan));
    if (plan.strategy != IREE_HAL_VULKAN_QUEUE_ALLOCA_STRATEGY_POOL) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "expected a pool allocation");
    }
    iree_hal_pool_capabilities_t capabilities;
    iree_hal_pool_query_capabilities(plan.pool, &capabilities);
    *out_type = capabilities.memory_type;
    return iree_ok_status();
  }

  static VKAPI_ATTR VkResult VKAPI_CALL
  CreateBuffer(VkDevice device, const VkBufferCreateInfo* info,
               const VkAllocationCallbacks* callbacks, VkBuffer* out_buffer) {
    *out_buffer = reinterpret_cast<VkBuffer>(new Buffer{info->size});
    return VK_SUCCESS;
  }

  static VKAPI_ATTR void VKAPI_CALL
  DestroyBuffer(VkDevice device, VkBuffer buffer,
                const VkAllocationCallbacks* callbacks) {
    delete reinterpret_cast<Buffer*>(buffer);
  }

  static VKAPI_ATTR void VKAPI_CALL GetBufferMemoryRequirements(
      VkDevice device, VkBuffer buffer, VkMemoryRequirements* requirements) {
    auto* self = reinterpret_cast<AllocatorTest*>(device);
    requirements->size = reinterpret_cast<Buffer*>(buffer)->size;
    requirements->alignment = 256;
    requirements->memoryTypeBits = self->allowed_memory_type_bits_;
  }

  static VKAPI_ATTR VkResult VKAPI_CALL AllocateMemory(
      VkDevice device, const VkMemoryAllocateInfo* info,
      const VkAllocationCallbacks* callbacks, VkDeviceMemory* out_memory) {
    auto* self = reinterpret_cast<AllocatorTest*>(device);
    const auto& properties =
        self->snapshot_.memory_properties2.memoryProperties;
    const uint32_t heap =
        properties.memoryTypes[info->memoryTypeIndex].heapIndex;
    if (info->allocationSize >
        properties.memoryHeaps[heap].size - self->heap_usage_[heap]) {
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    self->heap_usage_[heap] += info->allocationSize;
    self->allocation_types_.push_back(info->memoryTypeIndex);
    *out_memory = reinterpret_cast<VkDeviceMemory>(
        new Memory{heap, info->allocationSize});
    return VK_SUCCESS;
  }

  static VKAPI_ATTR void VKAPI_CALL
  FreeMemory(VkDevice device, VkDeviceMemory memory,
             const VkAllocationCallbacks* callbacks) {
    auto* self = reinterpret_cast<AllocatorTest*>(device);
    auto* allocation = reinterpret_cast<Memory*>(memory);
    self->heap_usage_[allocation->heap_index] -= allocation->size;
    delete allocation;
  }

  static VKAPI_ATTR VkResult VKAPI_CALL BindBufferMemory(VkDevice device,
                                                         VkBuffer buffer,
                                                         VkDeviceMemory memory,
                                                         VkDeviceSize offset) {
    return VK_SUCCESS;
  }

  iree_async_proactor_t* proactor_ = nullptr;
  iree_hal_allocator_t* allocator_ = nullptr;
  iree_hal_vulkan_device_syms_t syms_ = {};
  iree_hal_vulkan_physical_device_snapshot_t snapshot_ = {};
  uint32_t allowed_memory_type_bits_ = UINT32_MAX;
  std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS> heap_usage_ = {};
  std::vector<uint32_t> allocation_types_;
  std::vector<iree_hal_buffer_t*> buffers_;
};

TEST_F(AllocatorTest, UnmappedDeviceBuffersDoNotExhaustSmallBarHeap) {
  // Deliberately put BAR memory first: preference must not depend on
  // enumeration.
  AddMemoryType(kBar, 214 * kMiB);
  AddMemoryType(kPrivate, 4ull * 1024 * kMiB);
  AddMemoryType(kHost, 8ull * 1024 * kMiB);
  IREE_ASSERT_OK(CreateAllocator());

  for (int i = 0; i < 32; ++i) {
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(Allocate(DeviceParams(), 16 * kMiB, false, &buffer));
    EXPECT_EQ(IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
              iree_hal_buffer_memory_type(buffer));
  }
  ASSERT_FALSE(allocation_types_.empty());
  for (uint32_t index : allocation_types_) EXPECT_EQ(1u, index);
}

TEST_F(AllocatorTest, DirectDeviceBufferPrefersPrivateMemory) {
  AddMemoryType(kBar, 214 * kMiB);
  AddMemoryType(kPrivate, 4ull * 1024 * kMiB);
  IREE_ASSERT_OK(CreateAllocator());
  auto params = DeviceParams();
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(Allocate(params, 4096, true, &buffer));
  ASSERT_EQ(1u, allocation_types_.size());
  EXPECT_EQ(1u, allocation_types_.front());
}

TEST_F(AllocatorTest, QueueDevicePoolPrefersPrivateMemory) {
  AddMemoryType(kBar, 214 * kMiB);
  AddMemoryType(kPrivate, 4ull * 1024 * kMiB);
  IREE_ASSERT_OK(CreateAllocator());
  iree_hal_memory_type_t type = 0;
  IREE_ASSERT_OK(SelectQueuePool(DeviceParams(), &type));
  EXPECT_EQ(IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL, type);
}

TEST_F(AllocatorTest, TransferOnlyDeviceBuffersPreferPrivateMemory) {
  AddMemoryType(kBar, 214 * kMiB);
  AddMemoryType(kPrivate, 4ull * 1024 * kMiB);
  IREE_ASSERT_OK(CreateAllocator());
  auto params = DeviceParams();
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  for (bool direct : {false, true}) {
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(Allocate(params, 4096, direct, &buffer));
    EXPECT_EQ(IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
              iree_hal_buffer_memory_type(buffer));
  }
  iree_hal_memory_type_t type = 0;
  IREE_ASSERT_OK(SelectQueuePool(params, &type));
  EXPECT_EQ(IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL, type);
}

TEST_F(AllocatorTest, QueueSlabProviderPrefersPrivateMemory) {
  AddMemoryType(kBar, 214 * kMiB);
  AddMemoryType(kPrivate, 4ull * 1024 * kMiB);
  IREE_ASSERT_OK(CreateAllocator());
  iree_hal_queue_pool_backend_t backend = {};
  IREE_ASSERT_OK(iree_hal_vulkan_allocator_query_queue_pool_backend(
      allocator_, IREE_HAL_QUEUE_AFFINITY_ANY, &backend));
  iree_hal_memory_type_t type = 0;
  iree_hal_buffer_usage_t usage = 0;
  iree_hal_slab_provider_query_properties(backend.slab_provider, &type, &usage);
  EXPECT_EQ(IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL, type);
}

TEST_F(AllocatorTest, RequiredMappingStillUsesHostVisibleMemory) {
  AddMemoryType(kBar, 214 * kMiB);
  AddMemoryType(kPrivate, 4ull * 1024 * kMiB);
  IREE_ASSERT_OK(CreateAllocator());
  auto params = DeviceParams();
  params.usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(Allocate(params, 4096, false, &buffer));
  EXPECT_TRUE(iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                                IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  iree_hal_memory_type_t type = 0;
  IREE_ASSERT_OK(SelectQueuePool(params, &type));
  EXPECT_TRUE(iree_all_bits_set(type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  iree_hal_buffer_t* direct = nullptr;
  IREE_ASSERT_OK(Allocate(params, 4096, true, &direct));
  EXPECT_TRUE(iree_all_bits_set(iree_hal_buffer_memory_type(direct),
                                IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
}

TEST_F(AllocatorTest, UnifiedMemoryRemainsUsable) {
  AddMemoryType(kBar | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 4ull * 1024 * kMiB);
  IREE_ASSERT_OK(CreateAllocator());
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(Allocate(DeviceParams(), 4096, false, &buffer));
  EXPECT_TRUE(iree_all_bits_set(
      iree_hal_buffer_memory_type(buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  auto params = DeviceParams();
  iree_hal_buffer_t* direct = nullptr;
  IREE_ASSERT_OK(Allocate(params, 4096, true, &direct));
  iree_hal_memory_type_t type = 0;
  IREE_ASSERT_OK(SelectQueuePool(params, &type));
  EXPECT_TRUE(iree_all_bits_set(type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
}

TEST_F(AllocatorTest, HostStagingDoesNotRequirePrivateMemory) {
  AddMemoryType(kBar, 214 * kMiB);
  AddMemoryType(kPrivate, 4ull * 1024 * kMiB);
  AddMemoryType(kHost, 8ull * 1024 * kMiB);
  IREE_ASSERT_OK(CreateAllocator());
  auto params = DeviceParams();
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(Allocate(params, 4096, false, &buffer));
  EXPECT_TRUE(iree_all_bits_set(
      iree_hal_buffer_memory_type(buffer),
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  EXPECT_GT(heap_usage_[2], 0u);
}

TEST_F(AllocatorTest, DirectAllocationHonorsVulkanMemoryTypeBits) {
  AddMemoryType(kBar, 214 * kMiB);
  AddMemoryType(kPrivate, 4ull * 1024 * kMiB);
  IREE_ASSERT_OK(CreateAllocator());
  allowed_memory_type_bits_ = 1u;
  auto params = DeviceParams();
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(Allocate(params, 4096, true, &buffer));
  ASSERT_EQ(1u, allocation_types_.size());
  EXPECT_EQ(0u, allocation_types_.front());
}

}  // namespace
}  // namespace iree::hal::vulkan

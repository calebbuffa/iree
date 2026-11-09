// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/queue.h"

#include <cstdint>

#include "iree/async/proactor_platform.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::vulkan {
namespace {

#if !IREE_HAL_VULKAN_LIBVULKAN_STATIC

class QueueCompletionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    current_ = this;
    iree_slim_mutex_initialize(&queue_.submission_mutex);
    iree_slim_mutex_initialize(&native_mutex_);
    queue_.queue_handle_mutex = &native_mutex_;
    queue_.host_allocator = iree_allocator_system();
    queue_.next_epoch_value = 1;
    queue_.axis = iree_async_axis_make_queue(1, 0, 0, 0);
    queue_.syms.vkGetSemaphoreCounterValue = QueryEpoch;
    queue_.syms.vkQueueSubmit2 = Submit;
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &queue_.frontier_tracker));
    IREE_ASSERT_OK(iree_async_frontier_tracker_register_axis(
        queue_.frontier_tracker, queue_.axis, nullptr));
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
    IREE_ASSERT_OK(iree_async_semaphore_create(
        proactor_, 0, IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY,
        iree_allocator_system(), &signal_));
  }

  void TearDown() override {
    // Drain any submissions left by a failed assertion without a real device.
    reported_result_ = VK_ERROR_DEVICE_LOST;
    iree_hal_vulkan_queue_drain_completions(&queue_);
    iree_status_free(reinterpret_cast<iree_status_t>(
        iree_atomic_load(&queue_.failure_status, iree_memory_order_acquire)));
    iree_async_semaphore_release(signal_);
    iree_async_proactor_release(proactor_);
    iree_async_frontier_tracker_release(queue_.frontier_tracker);
    iree_slim_mutex_deinitialize(&native_mutex_);
    iree_slim_mutex_deinitialize(&queue_.submission_mutex);
    current_ = nullptr;
  }

  iree_status_t SubmitBarrier(uint64_t value) {
    iree_hal_semaphore_t* semaphore = iree_hal_semaphore_cast(signal_);
    iree_hal_semaphore_list_t signals = {1, &semaphore, &value};
    return iree_hal_vulkan_queue_submit_barrier(
        &queue_, iree_hal_semaphore_list_empty(), signals);
  }

  iree_status_code_t FailureCode() {
    return iree_status_code(reinterpret_cast<iree_status_t>(
        iree_atomic_load(&queue_.failure_status, iree_memory_order_acquire)));
  }

  void ExpectSignal(iree_status_code_t code, uint64_t value) {
    EXPECT_EQ(code, iree_async_semaphore_query_status(signal_));
    EXPECT_EQ(value, iree_async_semaphore_query(signal_));
    if (code != IREE_STATUS_OK) {
      iree_hal_semaphore_t* semaphore = iree_hal_semaphore_cast(signal_);
      uint64_t wait_value = 1;
      iree_hal_semaphore_list_t signals = {1, &semaphore, &wait_value};
      iree_status_t status = iree_hal_semaphore_list_wait(
          signals, iree_immediate_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
      EXPECT_EQ(code, iree_status_code(status));
      iree_status_free(status);
    }
  }

  static VKAPI_ATTR VkResult VKAPI_CALL QueryEpoch(VkDevice device,
                                                   VkSemaphore semaphore,
                                                   uint64_t* value) {
    ++current_->query_count_;
    if (current_->submit_during_query_) {
      current_->submit_during_query_ = false;
      IREE_EXPECT_OK(current_->SubmitBarrier(2));
    }
    if (current_->fail_during_query_) {
      current_->fail_during_query_ = false;
      iree_slim_mutex_lock(&current_->queue_.submission_mutex);
      iree_atomic_store(&current_->queue_.failure_status,
                        reinterpret_cast<intptr_t>(
                            iree_status_from_code(IREE_STATUS_ABORTED)),
                        iree_memory_order_release);
      iree_slim_mutex_unlock(&current_->queue_.submission_mutex);
    }
    *value = current_->reported_epoch_;
    return current_->reported_result_;
  }

  static VKAPI_ATTR VkResult VKAPI_CALL Submit(VkQueue queue, uint32_t count,
                                               const VkSubmitInfo2* infos,
                                               VkFence fence) {
    EXPECT_EQ(1u, count);
    EXPECT_EQ(1u, infos[0].signalSemaphoreInfoCount);
    ++current_->submit_count_;
    return VK_SUCCESS;
  }

  static QueueCompletionTest* current_;
  iree_hal_vulkan_queue_t queue_ = {};
  iree_slim_mutex_t native_mutex_;
  iree_async_proactor_t* proactor_ = nullptr;
  iree_async_semaphore_t* signal_ = nullptr;
  uint64_t reported_epoch_ = 0;
  VkResult reported_result_ = VK_SUCCESS;
  int query_count_ = 0;
  int submit_count_ = 0;
  bool submit_during_query_ = false;
  bool fail_during_query_ = false;
};

QueueCompletionTest* QueueCompletionTest::current_ = nullptr;

TEST_F(QueueCompletionTest, RetiresOnlyCompletedSubmissions) {
  IREE_ASSERT_OK(SubmitBarrier(1));
  IREE_ASSERT_OK(SubmitBarrier(2));
  EXPECT_EQ(2, submit_count_);
  EXPECT_EQ(0u, iree_hal_vulkan_queue_drain_completions(&queue_));
  ExpectSignal(IREE_STATUS_OK, 0);
  reported_epoch_ = 1;
  EXPECT_EQ(1u, iree_hal_vulkan_queue_drain_completions(&queue_));
  ExpectSignal(IREE_STATUS_OK, 1);
  EXPECT_TRUE(iree_async_frontier_tracker_query_epoch(queue_.frontier_tracker,
                                                      queue_.axis, 1));
  EXPECT_FALSE(iree_async_frontier_tracker_query_epoch(queue_.frontier_tracker,
                                                       queue_.axis, 2));
  reported_epoch_ = 2;
  EXPECT_EQ(1u, iree_hal_vulkan_queue_drain_completions(&queue_));
  ExpectSignal(IREE_STATUS_OK, 2);
  EXPECT_EQ(IREE_STATUS_OK, FailureCode());
  EXPECT_EQ(nullptr, queue_.pending_head);
  EXPECT_EQ(0u, iree_hal_vulkan_queue_drain_completions(&queue_));
}

TEST_F(QueueCompletionTest, RejectsNextUnsubmittedEpoch) {
  IREE_ASSERT_OK(SubmitBarrier(1));
  reported_epoch_ = queue_.next_epoch_value;
  EXPECT_EQ(0u, iree_hal_vulkan_queue_drain_completions(&queue_));
  EXPECT_EQ(IREE_STATUS_DATA_LOSS, FailureCode());
  ExpectSignal(IREE_STATUS_DATA_LOSS, 0);
  EXPECT_FALSE(iree_async_frontier_tracker_query_epoch(queue_.frontier_tracker,
                                                       queue_.axis, 1));
  EXPECT_EQ(nullptr, queue_.pending_head);
  EXPECT_EQ(nullptr, queue_.pending_tail);
}

TEST_F(QueueCompletionTest, RejectsMaxCounterAndKeepsFailureSticky) {
  IREE_ASSERT_OK(SubmitBarrier(1));
  IREE_ASSERT_OK(SubmitBarrier(2));
  reported_epoch_ = UINT64_MAX;
  EXPECT_EQ(0u, iree_hal_vulkan_queue_drain_completions(&queue_));
  EXPECT_EQ(IREE_STATUS_DATA_LOSS, FailureCode());
  ExpectSignal(IREE_STATUS_DATA_LOSS, 0);
  EXPECT_FALSE(iree_async_frontier_tracker_query_epoch(queue_.frontier_tracker,
                                                       queue_.axis, 2));
  EXPECT_EQ(nullptr, queue_.pending_head);
  reported_epoch_ = 2;
  EXPECT_EQ(0u, iree_hal_vulkan_queue_drain_completions(&queue_));
  EXPECT_EQ(1, query_count_);
  iree_status_t status = SubmitBarrier(3);
  EXPECT_EQ(IREE_STATUS_DATA_LOSS, iree_status_code(status));
  iree_status_free(status);
  EXPECT_EQ(2, submit_count_);
  EXPECT_EQ(3u, queue_.next_epoch_value);
}

TEST_F(QueueCompletionTest, PropagatesNativeDeviceLoss) {
  IREE_ASSERT_OK(SubmitBarrier(1));
  reported_result_ = VK_ERROR_DEVICE_LOST;
  EXPECT_EQ(0u, iree_hal_vulkan_queue_drain_completions(&queue_));
  EXPECT_EQ(IREE_STATUS_DATA_LOSS, FailureCode());
  ExpectSignal(IREE_STATUS_DATA_LOSS, 0);
  EXPECT_EQ(nullptr, queue_.pending_head);
}

TEST_F(QueueCompletionTest, AllowsSubmissionDuringCounterQuery) {
  IREE_ASSERT_OK(SubmitBarrier(1));
  submit_during_query_ = true;
  reported_epoch_ = 2;
  EXPECT_EQ(2u, iree_hal_vulkan_queue_drain_completions(&queue_));
  ExpectSignal(IREE_STATUS_OK, 2);
  EXPECT_EQ(IREE_STATUS_OK, FailureCode());
}

TEST_F(QueueCompletionTest, PreservesFailurePublishedDuringCounterQuery) {
  IREE_ASSERT_OK(SubmitBarrier(1));
  fail_during_query_ = true;
  reported_epoch_ = 1;
  EXPECT_EQ(0u, iree_hal_vulkan_queue_drain_completions(&queue_));
  EXPECT_EQ(IREE_STATUS_ABORTED, FailureCode());
  ExpectSignal(IREE_STATUS_ABORTED, 0);
  EXPECT_FALSE(iree_async_frontier_tracker_query_epoch(queue_.frontier_tracker,
                                                       queue_.axis, 1));
}

TEST_F(QueueCompletionTest, RejectsImpossibleEpochOnIdleQueue) {
  reported_epoch_ = 1;
  EXPECT_EQ(0u, iree_hal_vulkan_queue_drain_completions(&queue_));
  EXPECT_EQ(IREE_STATUS_DATA_LOSS, FailureCode());
}

#endif  // !IREE_HAL_VULKAN_LIBVULKAN_STATIC

}  // namespace
}  // namespace iree::hal::vulkan

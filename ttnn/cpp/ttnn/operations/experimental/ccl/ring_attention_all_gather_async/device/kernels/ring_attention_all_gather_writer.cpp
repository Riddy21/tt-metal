// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>

// FABRIC_2D vs 1D dispatch is handled portably via ccl_routing_utils::fabric_set_line_unicast_route
// (templated on packet-header type). The kernel pulls dst_chip_id/dst_mesh_id from runtime
// args and packs them into a line_unicast_route_info_t once; under 1D the helper reads
// route_info.distance_in_hops, under 2D it reads route_info.dst_chip_id + dst_mesh_id.

// UDM compat shim for ccl_routing_utils::fabric_set_line_unicast_route — see moe_utils.hpp's
// helper of the same name. Slice a UDMHybridMeshPacketHeader pointer down to HybridMesh* so the
// is_same_v<HybridMesh> arm in fabric_set_line_unicast_route catches it. No-op for HybridMesh
// and LowLatency.
template <typename HdrT>
FORCE_INLINE auto* pkt_hdr_for_route_helper(volatile HdrT* hdr) {
    if constexpr (std::is_same_v<HdrT, tt::tt_fabric::UDMHybridMeshPacketHeader>) {
        return reinterpret_cast<volatile tt::tt_fabric::HybridMeshPacketHeader*>(hdr);
    } else {
        return hdr;
    }
}

using address_t = uint32_t;
using ttnn::ccl::Topology;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t num_packet_headers_storable = get_compile_time_arg_val(2);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(3);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(4);
constexpr uint32_t output_page_size = get_compile_time_arg_val(5);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(6);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(7);
constexpr bool dynamic_alternate = get_compile_time_arg_val(8);
constexpr bool fuse_op = get_compile_time_arg_val(9);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(10));
constexpr uint32_t contig_pages_advanced = get_compile_time_arg_val(11);
constexpr uint32_t num_inputs = get_compile_time_arg_val(12);
constexpr bool direction = get_compile_time_arg_val(13);  // 1 is forward, 0 is backward

void kernel_main() {
    constexpr uint32_t page_size_base_idx = 14;
    constexpr auto outputs_args = make_tensor_accessor_args_tuple<num_inputs, page_size_base_idx + num_inputs>();

    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    uint32_t arg_idx = 0;
    uint32_t gather_dim = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);

    std::array<uint32_t, num_inputs> input_tensor_Wt;
    std::array<uint32_t, num_inputs> input_tensor_Ht;
    std::array<uint32_t, num_inputs> output_tensor_Wt;
    std::array<uint32_t, num_inputs> output_tensor_Ht;
    std::array<uint32_t, num_inputs> input_batch_head_count;
    std::array<uint32_t, num_inputs> input_tile_id_start;
    std::array<uint32_t, num_inputs> input_tile_id_end;

    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        input_tensor_Wt[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_tensor_Ht[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        output_tensor_Wt[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        output_tensor_Ht[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_batch_head_count[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_tile_id_start[input_idx] = get_arg_val<uint32_t>(arg_idx++);
        input_tile_id_end[input_idx] = get_arg_val<uint32_t>(arg_idx++);
    }

    auto outputs_tuple = make_tensor_accessor_tuple(outputs_args, arg_idx);
    arg_idx += num_inputs;
    auto output_addrgens = make_abstract_tensor_accessor_wrappers(outputs_tuple);
    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_for_fab);
    // FABRIC_2D: destination chip+mesh for this writer's fabric direction. Pushed by
    // the program factory unconditionally (both under 1D and 2D) — under 1D the kernel
    // ignores these. See program_factory writer_{forward,backward}_extra_args.
    const uint16_t fabric_dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_for_fab++));
    const uint16_t fabric_dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_for_fab++));

    // Single route_info populated once. Under 1D the helper consumes distance_in_hops (always
    // 1 hop to the immediate neighbor); under 2D it consumes dst_chip_id + dst_mesh_id.
    ccl_routing_utils::line_unicast_route_info_t route_info{};
    route_info.dst_mesh_id = fabric_dst_mesh_id;
    if constexpr (
        std::is_same_v<PACKET_HEADER_TYPE, tt::tt_fabric::HybridMeshPacketHeader> ||
        std::is_same_v<PACKET_HEADER_TYPE, tt::tt_fabric::UDMHybridMeshPacketHeader>) {
        route_info.dst_chip_id = fabric_dst_chip_id;
    } else {
        route_info.distance_in_hops = 1;  // ring-AG only sends to the immediate neighbor
    }
    /* Args for overlapped all gather */
    OpSignaler op_signaler_sender;

    if constexpr (fuse_op) {
        arg_idx = arg_for_fab;
        op_signaler_sender = OpSignaler(arg_idx);
    }

    // packet header cb
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_seminc = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);

    // pre-populate packet headers — only meaningful when this writer actually sends in
    // its direction. On terminal chips num_targets_in_direction (==
    // direction==1 ? num_targets_backward_direction : num_targets_forward_direction) is 0
    // and the program factory pushes fabric_dst_{chip,mesh}_id=0; pre-populating with those
    // would corrupt the header with (chip 0, mesh 0). Guard so the route is set only when
    // there is at least one downstream target.
    volatile PACKET_HEADER_TYPE* pkt_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr);
    constexpr uint32_t num_targets_in_direction =
        direction == 1 ? num_targets_backward_direction : num_targets_forward_direction;
    if constexpr (num_targets_in_direction > 0) {
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_for_route_helper(pkt_hdr), route_info);
    }

    fabric_connection.open();

    tt::tt_fabric::WorkerToFabricEdmSender* fabric_direction_connection =
        fabric_connection.is_logically_connected() ? (direction == 1 ? &fabric_connection.get_backward_connection()
                                                                     : &fabric_connection.get_forward_connection())
                                                   : nullptr;
    // num_targets_in_direction was declared above (next to the pkt_hdr pre-population guard).

    uint32_t slice_writes = 0;

    uint32_t row_offset = 0;
    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        /**
         * Write out the local slice to forward and backward devices
         * Note that it is not copied to local output buffer. This is because
         * the fused op (RingJointAttention) reads from the input buffer directly
         * when accessing the local slice. This is a performance optimization
         * to remove startup latency from the fused op.
         */

        uint32_t tile_id_start = my_chip_id * input_tensor_Wt[input_idx];
        uint32_t pages_read_in_row = input_tile_id_start[input_idx] % input_tensor_Wt[input_idx];
        uint32_t row_offset =
            (input_tile_id_start[input_idx] / input_tensor_Wt[input_idx]) * output_tensor_Wt[input_idx];
        uint32_t tiles_read = input_tile_id_start[input_idx];
        uint32_t tiles_to_read = input_tile_id_end[input_idx];
        if (gather_dim == 3) {
            tile_id_start = my_chip_id * input_tensor_Wt[input_idx];
        } else {
            tile_id_start = my_chip_id * input_tensor_Ht[input_idx] * input_tensor_Wt[input_idx];
        }

        for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count[input_idx]; bh_idx++) {
            while (tiles_read < tiles_to_read) {
                uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
                cb_wait_front(cb_output_id, packet_size_in_pages);
                const size_t l1_read_addr_base = get_read_ptr(cb_output_id);
                size_t l1_read_addr = l1_read_addr_base;

                // for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                uint32_t tile_id = tile_id_start + row_offset + pages_read_in_row;

                pages_read_in_row++;
                if (pages_read_in_row >= input_tensor_Wt[input_idx]) {
                    row_offset += output_tensor_Wt[input_idx];
                    pages_read_in_row = 0;
                }

                if (num_pages_to_read == 2) {
                    uint32_t second_tile_id = tile_id_start + row_offset + pages_read_in_row;

                    if constexpr (num_targets_in_direction) {
                        scatter_fabric_write_unidir(
                            tile_id,
                            second_tile_id,
                            output_addrgens[input_idx],
                            pkt_hdr,
                            *fabric_direction_connection,
                            l1_read_addr,
                            output_page_size);
                    }

                    pages_read_in_row++;
                    if (pages_read_in_row >= input_tensor_Wt[input_idx]) {
                        row_offset += output_tensor_Wt[input_idx];
                        pages_read_in_row = 0;
                    }
                } else {
                    ASSERT(num_pages_to_read == 1);

                    if constexpr (num_targets_in_direction) {
                        // Has valid targets to send to
                        fabric_write_unidir(
                            tile_id,
                            output_addrgens[input_idx],
                            pkt_hdr,
                            *fabric_direction_connection,
                            l1_read_addr,
                            output_page_size);
                    }
                }

                tiles_read += num_pages_to_read;
                cb_pop_front(cb_output_id, packet_size_in_pages);
            }
            tile_id_start += output_tensor_Wt[input_idx] * output_tensor_Ht[input_idx];
            tiles_read = input_tile_id_start[input_idx];
            tiles_to_read = input_tile_id_end[input_idx];
            pages_read_in_row = input_tile_id_start[input_idx] % input_tensor_Wt[input_idx];
            row_offset = (input_tile_id_start[input_idx] / input_tensor_Wt[input_idx]) * output_tensor_Wt[input_idx];
        }
    }

    noc_async_write_barrier();
    // increment locally
    if constexpr (fuse_op && direction == 1) {
        /**
         * Synchronize and signal that the local tensor slice is available
         *
         * While the fused op will not wait on this "local write done" increment,
         * the fused op signaler will account for it in future waits.
         */
        op_signaler_sender.synchronize_workers_and_signal_op(my_chip_id);
    }

    // 2. unicast output ready semaphore
    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
    auto* pkt_hdr_sem_inc = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);
    const auto sem_inc_cmd_header =
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, static_cast<uint32_t>(1)};

    // Write the unicast packet. num_hops=1 is correct under both topologies: 1D ring-AG always
    // targets the immediate neighbor; 2D ignores num_hops (HybridMesh::to_chip_unicast is a no-op
    // and route_info carries the 2D destination).
    if constexpr (num_targets_in_direction) {
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_for_route_helper(pkt_hdr_sem_inc), route_info);
        tt::tt_fabric::linear::experimental::fabric_unicast_noc_unicast_atomic_inc(
            fabric_direction_connection, pkt_hdr_sem_inc, sem_inc_cmd_header, /*num_hops=*/1);
    }

    uint32_t writes_expected = 0;
    if constexpr (topology == Topology::Linear) {
        if constexpr (direction == 1 && num_targets_backward_direction) {
            writes_expected = num_targets_forward_direction;
        } else if constexpr (direction == 0 && num_targets_forward_direction) {
            writes_expected = num_targets_backward_direction;
        }
    } else if constexpr (topology == Topology::Ring) {
        if constexpr (direction == 1) {
            writes_expected = num_targets_backward_direction - 1;
        } else {
            writes_expected = num_targets_forward_direction - 1;
        }
    }

    while (slice_writes < writes_expected) {
        // Direction == backward
        // Did I get something from my left to send to my right?
        // In the linear case, I expect num_targets_backward_direction slices from the left, and check if I have a
        // neighbor to the right
        // In the ring case, I expect to write to the right num_forward_target times
        // Direction == forward
        // Did I get something from my right to send to my left?
        // In the linear case, I expect num_targets_forward_direction slices from the right, and check if I have a
        // neighbor to the left
        // In the ring case, I expect to write to the left num_backward_target times

        int slice_chip_id;
        uint32_t actual_slice_chip_id;
        if constexpr (direction == 1) {
            slice_chip_id = my_chip_id + slice_writes + 1;
            actual_slice_chip_id = (slice_chip_id >= (int)ring_size) ? slice_chip_id - ring_size : slice_chip_id;
        } else {
            slice_chip_id = my_chip_id - slice_writes - 1;
            actual_slice_chip_id = (slice_chip_id < 0) ? ring_size + slice_chip_id : slice_chip_id;
        }
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            uint32_t tiles_read = input_tile_id_start[input_idx];
            uint32_t tiles_to_read = input_tile_id_end[input_idx];
            uint32_t tile_id_start = actual_slice_chip_id * input_tensor_Wt[input_idx];
            uint32_t row_offset =
                (input_tile_id_start[input_idx] / input_tensor_Wt[input_idx]) * output_tensor_Wt[input_idx];
            uint32_t pages_read_in_row = (input_tile_id_start[input_idx] % input_tensor_Wt[input_idx]);
            uint32_t slice_Wt = input_tensor_Wt[input_idx];
            uint32_t stride_Wt = output_tensor_Wt[input_idx];

            if (gather_dim == 3) {
                tile_id_start = actual_slice_chip_id * input_tensor_Wt[input_idx];
            } else {
                tile_id_start = actual_slice_chip_id * input_tensor_Ht[input_idx] * input_tensor_Wt[input_idx];
            }
            for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count[input_idx]; bh_idx++) {
                while (tiles_read < tiles_to_read) {
                    uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
                    cb_wait_front(cb_output_id, packet_size_in_pages);
                    size_t l1_read_addr = get_read_ptr(cb_output_id);
                    uint32_t first_tile_id = tile_id_start + row_offset + pages_read_in_row;
                    pages_read_in_row++;
                    if (pages_read_in_row >= slice_Wt) {
                        row_offset += stride_Wt;
                        pages_read_in_row = 0;
                    }

                    if (num_pages_to_read == 2) {
                        uint32_t second_tile_id = tile_id_start + row_offset + pages_read_in_row;
                        pages_read_in_row++;
                        if (pages_read_in_row >= slice_Wt) {
                            row_offset += stride_Wt;
                            pages_read_in_row = 0;
                        }

                        scatter_fabric_write_unidir(
                            first_tile_id,
                            second_tile_id,
                            output_addrgens[input_idx],
                            pkt_hdr,
                            *fabric_direction_connection,
                            l1_read_addr,
                            output_page_size);
                    } else {
                        ASSERT(num_pages_to_read == 1);
                        fabric_write_unidir(
                            first_tile_id,
                            output_addrgens[input_idx],
                            pkt_hdr,
                            *fabric_direction_connection,
                            l1_read_addr,
                            output_page_size);
                    }

                    tiles_read += num_pages_to_read;
                    cb_pop_front(cb_output_id, packet_size_in_pages);
                }
                tile_id_start += output_tensor_Wt[input_idx] * output_tensor_Ht[input_idx];
                tiles_read = input_tile_id_start[input_idx];
                tiles_to_read = input_tile_id_end[input_idx];
                row_offset =
                    (input_tile_id_start[input_idx] / input_tensor_Wt[input_idx]) * output_tensor_Wt[input_idx];
                pages_read_in_row = (input_tile_id_start[input_idx] % input_tensor_Wt[input_idx]);
            }
        }

        // 2. unicast output ready semaphore forward
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_for_route_helper(pkt_hdr_sem_inc), route_info);
        tt::tt_fabric::linear::experimental::fabric_unicast_noc_unicast_atomic_inc(
            fabric_direction_connection, pkt_hdr_sem_inc, sem_inc_cmd_header, /*num_hops=*/1);

        slice_writes++;
    }

    // Drain in-flight writes BEFORE closing the EDM connections.
    noc_async_atomic_barrier();
    noc_async_write_barrier();

    fabric_connection.close();
}

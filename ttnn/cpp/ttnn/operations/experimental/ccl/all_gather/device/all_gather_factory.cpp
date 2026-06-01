// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_factory.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>

#include <bit>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::experimental::prim {

using namespace ::ttnn::ccl;

namespace {  // anonymous namespace for internal helpers

CoreRangeSet get_cores_close_to_erisc(uint32_t num_workers, bool row_wise) {
    CoreRangeSet worker_cores;
    std::vector<CoreRange> desired_core_range = {
        CoreRange({5, 3}, {6, 3}), CoreRange({2, 8}, {3, 8}), CoreRange({1, 1}, {4, 4})};
    for (const auto& cr : desired_core_range) {
        auto cores = corerange_to_cores(cr, std::nullopt, row_wise);
        for (const auto& core : cores) {
            worker_cores = worker_cores.merge(CoreRangeSet(CoreRange(core, core)));
            if (worker_cores.num_cores() == num_workers) {
                break;
            }
        }
        if (worker_cores.num_cores() == num_workers) {
            break;
        }
    }
    return worker_cores;
}

}  // namespace

AllGatherFactory::cached_mesh_workload_t AllGatherFactory::create_mesh_workload(
    const AllGatherParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const AllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();
    auto subdevice_id = mesh_device->get_sub_device_ids().at(0);
    const auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id);
    ttnn::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};

    // TODO only use sems if output not provided. need to wire this down to kernel level.
    // If output tensors are not persistent (globally allocated), we need to wait for all devices to be ready
    // before beginning our operation.
    // Since Fabric doesn't provide such capability within kernels, we need to manually sync using global semaphores.
    // tt::tt_metal::GlobalSemaphore init_barrier_semaphore;
    // tt::tt_metal::GlobalSemaphore final_barrier_semaphore;
    // if (!tensor_args.persistent_output_tensor.has_value()) {
    // Allocate semaphores in L1_SMALL to avoid fragmenting the larger L1 memory pool.
    bool l1_small_size = mesh_device->allocator()->get_bank_size(tt::tt_metal::BufferType::L1_SMALL);
    auto sem_buffer_type = l1_small_size > 0 ? tt::tt_metal::BufferType::L1_SMALL : tt::tt_metal::BufferType::L1;
    if (sem_buffer_type != tt::tt_metal::BufferType::L1_SMALL) {
        log_warning(
            tt::LogOp,
            "Allocating semaphores in L1, which may cause memory fragmentation and lead to memory allocation failures "
            "for subsequent operations. Configure an L1_SMALL region to avoid this.");
    }

    auto init_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type);
    auto final_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type);
    log_debug(tt::LogOp, "Semaphores allocated and waiting for all devices to be ready");
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevices);
    log_debug(tt::LogOp, "All devices are ready, starting program execution");
    //}

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(
            operation_attributes,
            coord,
            tensor_args.input_tensor,
            output_tensor,
            final_barrier_semaphore,
            init_barrier_semaphore);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

AllGatherFactory::cached_program_t AllGatherFactory::create_at(
    const AllGatherParams& operation_attributes,
    const ttnn::MeshCoordinate& sender_device_coord,
    const Tensor& input,
    const Tensor& output_tensor,
    const tt::tt_metal::GlobalSemaphore& semaphore,
    const tt::tt_metal::GlobalSemaphore& barrier_semaphore) {
    const auto& input_tensor = input;
    tt::tt_metal::Program program{};

    ////////////////////////////////////////////////////////////////
    // Fabric setup
    ////////////////////////////////////////////////////////////////

    uint32_t num_devices = operation_attributes.ring_size;
    uint32_t device_idx = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, sender_device_coord, operation_attributes.cluster_axis);
    // TODO verify row-major device_idx matches ShardTensorToMesh order under 2D no-cluster_axis;
    // manual (2,4) test will catch any mismatch. i.e. in 2D mesh shape, whats the device_slot?

    // Compute hops + neighbors for each mesh axis.
    // Each axis ∈ {0, 1} contributes a forward/backward pair: axis 1 -> (E=fwd, W=bwd),
    // axis 0 -> (S=fwd, N=bwd). In 1D only one axis is active; in 2D both can be.
    const auto fabric_config = tt::tt_fabric::GetFabricConfig();
    const bool fabric_is_2d = ::tt::tt_fabric::is_2d_fabric_config(fabric_config);
    const auto mesh_shape = input_tensor.device()->shape();

    std::optional<MeshCoordinate> e_coord, w_coord, n_coord, s_coord;
    uint32_t e_hops = 0, w_hops = 0, n_hops = 0, s_hops = 0;
    bool ew_load_balance = false;
    bool ns_load_balance = false;

    for (uint32_t axis = 0; axis < 2; ++axis) {
        const bool is_axis_active = mesh_shape[axis] > 1 && operation_attributes.cluster_axis.value_or(axis) == axis;
        if (!is_axis_active) {
            continue;
        }

        // Ring detection: fabric config wraps this axis AND the device set spans [0..size-1].
        // TODO consider resolving this (and replace `topology`) in device_operation.cpp
        bool axis_can_wrap;
        if (fabric_is_2d) {
            if (axis == 1) {
                axis_can_wrap = fabric_config == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X ||
                                fabric_config == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY;
            } else {
                axis_can_wrap = fabric_config == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_Y ||
                                fabric_config == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY;
            }
        } else {
            axis_can_wrap = fabric_config == tt::tt_fabric::FabricConfig::FABRIC_1D_RING;
        }
        const bool axis_is_ring =
            axis_can_wrap && ::ttnn::ccl::get_boundary_mode(input_tensor, tt::tt_fabric::Topology::Torus, axis) ==
                                 tt::tt_metal::distributed::MeshCoordinate::BoundaryMode::WRAP;
        const auto axis_topology = axis_is_ring ? tt::tt_fabric::Topology::Ring : tt::tt_fabric::Topology::Linear;

        const uint32_t axis_size = ::ttnn::ccl::get_topological_dimension(input_tensor, axis);
        const uint32_t axis_index = sender_device_coord[axis];
        auto [fwd_hops, bwd_hops] = ::ttnn::ccl::get_forward_backward_line_mcast_distance(
            axis_size, axis_index, axis_topology, /*static_alternate=*/false);
        auto fwd_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
            input_tensor, sender_device_coord, 1, axis_topology, axis);
        auto bwd_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
            input_tensor, sender_device_coord, -1, axis_topology, axis);

        const bool axis_load_balance = axis_is_ring && (axis_size % 2 == 0);
        if (axis == 1) {
            e_hops = fwd_hops;
            w_hops = bwd_hops;
            e_coord = fwd_coord;
            w_coord = bwd_coord;
            ew_load_balance = axis_load_balance;
        } else {
            s_hops = fwd_hops;
            n_hops = bwd_hops;
            s_coord = fwd_coord;
            n_coord = bwd_coord;
            ns_load_balance = axis_load_balance;
        }
    }
    TT_FATAL(
        e_coord.has_value() || w_coord.has_value() || n_coord.has_value() || s_coord.has_value(),
        "No neighboring devices");

    const uint32_t packet_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();

    auto* mesh_device = input_tensor.device();
    const auto& num_links_per_axis = operation_attributes.num_links_per_axis;

    // One worker core per link per active axis. Allocate the total up front, then split into
    // per-axis contiguous slices (axis 0 = N/S first, then axis 1 = E/W). get_cores_close_to_erisc
    // is deterministic and mesh-uniform, so a given (axis, link) maps to the same physical core on
    // every device -- required for the per-core semaphore targeting below.
    const uint32_t total_links = num_links_per_axis[0] + num_links_per_axis[1];
    TT_FATAL(total_links > 0, "AllGather: no active fabric links on any axis");
    auto sender_worker_core_range = get_cores_close_to_erisc(total_links, true);
    auto all_sender_worker_cores = corerange_to_cores(sender_worker_core_range);

    std::array<std::vector<CoreCoord>, 2> axis_cores;
    {
        uint32_t idx = 0;
        for (uint32_t a = 0; a < 2; ++a) {
            for (uint32_t l = 0; l < num_links_per_axis[a]; ++l) {
                axis_cores[a].push_back(all_sender_worker_cores[idx++]);
            }
        }
    }

    // axis 0 (N/S) is the rect (S/N spine + E/W branches); axis 1 (E/W) is a pure line. In a 1D run
    // only one axis is active (the N/S rect degenerates to a line because its E/W branch hops are 0).
    const uint32_t size0 = ::ttnn::ccl::get_topological_dimension(input_tensor, 0);
    const uint32_t size1 = ::ttnn::ccl::get_topological_dimension(input_tensor, 1);
    const bool both_active = num_links_per_axis[0] > 0 && num_links_per_axis[1] > 0;

    // Devices reached by one core's fwd+bwd mcast. axis 0 N/S rect: all other rows x full width
    // (or a pure N/S line when E/W inactive). axis 1 E/W line: my row. They sum to num_devices-1.
    auto coverage_of = [&](uint32_t a) -> uint32_t {
        return (a == 0) ? (size0 - 1) * (both_active ? size1 : 1) : (size1 - 1);
    };

    // Alternate-route hops: on an even ring, fwd/bwd swap to balance the two routes.
    const uint32_t e_alt = ew_load_balance ? w_hops : e_hops;
    const uint32_t w_alt = ew_load_balance ? e_hops : w_hops;
    const uint32_t s_alt = ns_load_balance ? n_hops : s_hops;
    const uint32_t n_alt = ns_load_balance ? s_hops : n_hops;

    // Any active axis does the local write of this device's own slice; its per-link partition is
    // complete, so all local pages are covered exactly once (avoids a double local write in 2D).
    const uint32_t local_write_axis = (num_links_per_axis[0] > 0) ? 0 : 1;

    // out_ready fan-in at the single global drain core (link 0 of the first active axis):
    //   sum over axes of coverage(axis)*links(axis)   (remote mcast hits)
    //   + 2*links(axis)                               (local incs: reader + writer per core).
    uint32_t out_ready_sem_wait_value = 0;
    for (uint32_t a = 0; a < 2; ++a) {
        out_ready_sem_wait_value += (coverage_of(a) + 2) * num_links_per_axis[a];
    }
    const CoreCoord global_first_core = all_sender_worker_cores[0];
    const CoreCoord drain_sync_core = mesh_device->worker_core_from_logical_core(global_first_core);

    ////////////////////////////////////////////////////////////////
    // Page indexing
    //
    // Glossary:
    //   input page     -- one page of the input tensor.
    //   output page    -- one page of the output tensor.
    //                     In concat mode the *kernel-visible* output page size is
    //                     smaller than the tensor's actual output page.
    //   stripe         -- a run of consecutive output page ids this device writes
    //                     before jumping past other devices' contributions.
    //   stripe jump    -- value the kernel adds to output_page_id at the stripe
    //                     boundary.
    //   stripe distance-- page-id distance from start of one of this device's
    //                     stripes to the start of the next.
    //
    // Three copy modes, picked by input vs output page sizes:
    //   matched (in == out): 1 write per input page, byte offset = 0.
    //   concat  (out > in) : 1 write per input page, byte offset = (d % concat_factor) * in.
    //   split   (in > out) : split_factor writes per input page, byte offset = 0.
    //
    // Kernel is a dumb page iterator. Iteration pattern is periodic stripes, i.e.
    // consecutive pages (stripe) followed by periodic jumps (to the next stripe).
    //
    // Host derives the iterator parameters from input/output page sizes, gather dim,
    // and device index.
    ////////////////////////////////////////////////////////////////

    const uint32_t input_page_size = input_tensor.buffer()->aligned_page_size();
    const uint32_t output_page_size = output_tensor.buffer()->aligned_page_size();

    auto input_shape = input_tensor.padded_shape();
    uint32_t rank = input_shape.rank();
    int32_t gather_dim = operation_attributes.dim;
    if (gather_dim < 0) {
        gather_dim += rank;
    }

    // --- Copy mode ---
    // Exactly one of concat_factor or split_factor is > 1 (or both = 1 in matched mode).
    const uint32_t concat_factor = std::max(1u, output_page_size / input_page_size);
    const uint32_t split_factor = std::max(1u, input_page_size / output_page_size);

    // kernel_output_page_size = bytes per write = min(input, output).
    const uint32_t kernel_output_page_size = std::min(input_page_size, output_page_size);
    const uint32_t output_page_byte_offset = (device_idx % concat_factor) * input_page_size;
    const uint32_t device_slot = device_idx / concat_factor;
    const uint32_t num_input_pages = input_tensor.buffer()->num_pages();
    const uint32_t num_output_pages = num_input_pages * split_factor;

    // --- CB sizing ---
    // cb_page_size is a multiple of input_page_size, which is itself a multiple of
    // kernel_output_page_size = min(input, output), so the kernel increments both
    // the cb_read_ptr and cb_write_ptr cleanly.
    const uint32_t pages_per_packet = std::max(1u, packet_size / input_page_size);
    uint32_t cb_page_size = input_page_size * pages_per_packet;

    // Perf hack: for tile layout, pack multiple pages into a single CB page to reduce
    // CB sync frequency between reader and writer. Don't do this for RM because of all
    // the careful handling of page sizes.
    // TODO identify the multiplier based on available L1 space
    if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
        cb_page_size *= 4;
    }

    // --- Stripe geometry ---
    // input_pages_per_stripe = num input pages along [gather_dim .. rank-1] this
    // device contributes per stripe. For RM gather_dim=-1 this is the *page* count,
    // which handles sharded RM input (> 1 input page per row).
    uint32_t input_pages_per_stripe = 1;
    for (int32_t i = gather_dim; i < rank; i++) {
        uint32_t extent;
        if (i == rank - 1) {
            if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
                extent = input_shape[i] / tt::constants::TILE_WIDTH;
            } else {
                extent = (input_shape[i] * input_tensor.element_size()) / input_page_size;
            }
        } else if (input_tensor.layout() == ttnn::TILE_LAYOUT && i == rank - 2) {
            extent = input_shape[i] / tt::constants::TILE_HEIGHT;
        } else {
            extent = input_shape[i];
        }
        input_pages_per_stripe *= extent;
    }

    // In concat mode each iter goes to a different output page (one chunk per row
    // from this device), so the stripe is 1 long. Otherwise the stripe spans this
    // device's full contribution: input_pages_per_stripe * split_factor consecutive
    // output pages.
    const uint32_t output_pages_per_stripe = (concat_factor > 1) ? 1u : input_pages_per_stripe * split_factor;
    const uint32_t output_page_stripe_distance = (num_devices * input_pages_per_stripe * split_factor) / concat_factor;
    const uint32_t output_page_stripe_jump = output_page_stripe_distance - output_pages_per_stripe + 1;
    TT_FATAL(output_pages_per_stripe > 0, "output_pages_per_stripe must be > 0");

    ////////////////////////////////////////////////////////////////
    // Circular Buffer and Kernel creation
    ////////////////////////////////////////////////////////////////

    // L1 Scratch CB Creation
    uint32_t cb0_id = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(3 * cb_page_size, {{cb0_id, df}}).set_page_size(cb0_id, cb_page_size);

    CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);

    // KERNEL CREATION + RUNTIME ARGS -- one (reader, writer) pair per active axis.
    // axis 0 = N/S rect (reader S-spine, writer N-spine; both branch full-width E/W).
    // axis 1 = E/W line (reader E, writer W). Same kernel source, different compile args.
    // Each kernel sends a single mcast described by its {e,w,n,s} hop rectangle.
    std::array<tt::tt_metal::KernelHandle, 2> reader_kernel_ids{};
    std::array<tt::tt_metal::KernelHandle, 2> writer_kernel_ids{};
    const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);

    for (uint32_t a = 0; a < 2; ++a) {
        const uint32_t n_links = num_links_per_axis[a];
        if (n_links == 0) {
            continue;
        }

        // The {e,w,n,s} hop rectangle for this axis's reader (fwd) and writer (bwd) mcasts.
        // axis 0 N/S: S/N spine with full E/W branches (e/w are 0 when E/W inactive => pure N/S line).
        // axis 1 E/W: pure line (no spine -- E/W cannot be a spine in the fabric).
        std::array<uint32_t, 4> reader_rect{}, reader_rect_alt{}, writer_rect{}, writer_rect_alt{};
        std::optional<MeshCoordinate> fwd_coord, bwd_coord;
        uint32_t fwd_hop = 0, bwd_hop = 0, coverage = 0;
        bool load_balance = false;
        if (a == 0) {  // N/S rect
            reader_rect = {e_hops, w_hops, 0, s_hops};
            reader_rect_alt = {e_alt, w_alt, 0, s_alt};
            writer_rect = {e_hops, w_hops, n_hops, 0};
            writer_rect_alt = {e_alt, w_alt, n_alt, 0};
            fwd_coord = s_coord;
            bwd_coord = n_coord;
            fwd_hop = s_hops;
            bwd_hop = n_hops;
            coverage = coverage_of(0);
            load_balance = ns_load_balance || ew_load_balance;
        } else {  // E/W line
            reader_rect = {e_hops, 0, 0, 0};
            reader_rect_alt = {e_alt, 0, 0, 0};
            writer_rect = {0, w_hops, 0, 0};
            writer_rect_alt = {0, w_alt, 0, 0};
            fwd_coord = e_coord;
            bwd_coord = w_coord;
            fwd_hop = e_hops;
            bwd_hop = w_hops;
            coverage = coverage_of(1);
            load_balance = ew_load_balance;
        }
        const bool do_local_write = (a == local_write_axis);

        std::vector<uint32_t> reader_compile_args = {
            cb0_id,                   // cb0_id
            input_page_size,          // input tensor page size
            kernel_output_page_size,  // kernel-visible page size
            output_pages_per_stripe,  // stripe length
            output_page_stripe_jump,  // stripe jump
            cb_page_size,             // cb entry size
            packet_size,              // packet_size
            load_balance,             // load_balance_across_alt_routes
            (fwd_hop > 0),            // num_connections
            reader_rect[0],
            reader_rect[1],
            reader_rect[2],
            reader_rect[3],  // hop_{e,w,n,s}
            reader_rect_alt[0],
            reader_rect_alt[1],
            reader_rect_alt[2],
            reader_rect_alt[3],  // hop_{e,w,n,s}_alt
        };
        tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_compile_args);

        std::vector<uint32_t> writer_compile_args = {
            cb0_id,                   // cb0_id
            kernel_output_page_size,  // kernel-visible page size
            output_pages_per_stripe,  // stripe length
            output_page_stripe_jump,  // stripe jump
            cb_page_size,             // cb entry size
            packet_size,              // packet_size
            load_balance,             // load_balance_across_alt_routes
            (bwd_hop > 0),            // num_connections
            writer_rect[0],
            writer_rect[1],
            writer_rect[2],
            writer_rect[3],  // hop_{e,w,n,s}
            writer_rect_alt[0],
            writer_rect_alt[1],
            writer_rect_alt[2],
            writer_rect_alt[3],  // hop_{e,w,n,s}_alt
            do_local_write,      // do_local_write
        };
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_args);

        const auto& cores = axis_cores[a];
        std::vector<CoreRange> axis_core_ranges;
        axis_core_ranges.reserve(cores.size());
        for (const auto& c : cores) {
            axis_core_ranges.emplace_back(c, c);
        }
        CoreRangeSet axis_core_range_set(axis_core_ranges);

        reader_kernel_ids[a] = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather/device/kernels/reader.cpp",
            axis_core_range_set,
            tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));
        writer_kernel_ids[a] = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather/device/kernels/writer.cpp",
            axis_core_range_set,
            tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

        for (uint32_t link = 0; link < n_links; ++link) {
            const CoreCoord core = cores[link];
            const CoreCoord barrier_core = mesh_device->worker_core_from_logical_core(core);

            // Split this device's input pages across this axis's links and map to output writes.
            // num_output_pages already accounts for split_factor, so in matched/concat modes the
            // ratio cancels back to num_input_pages.
            const uint32_t input_pages_per_link = num_input_pages / n_links;
            const uint32_t remainder = num_input_pages % n_links;
            const uint32_t input_tile_id_start = (link * input_pages_per_link) + std::min(link, remainder);
            const uint32_t input_tile_id_end = ((link + 1) * input_pages_per_link) + std::min(link + 1, remainder);
            const uint32_t local_output_start = (input_tile_id_start * num_output_pages) / num_input_pages;
            const uint32_t local_output_end = (input_tile_id_end * num_output_pages) / num_input_pages;
            const uint32_t num_worker_output_pages = local_output_end - local_output_start;
            // stripe_index*stripe_distance + device_slot*pages_per_stripe + position_in_stripe.
            const uint32_t output_page_id_start =
                (local_output_start / output_pages_per_stripe) * output_page_stripe_distance +
                device_slot * output_pages_per_stripe + local_output_start % output_pages_per_stripe;
            const uint32_t output_page_in_stripe_start = local_output_start % output_pages_per_stripe;

            // Single global drain core (link 0 of the first active axis) owns the out_ready wait+reset.
            const bool owns_out_ready_sem = (core == global_first_core);
            // Per-core barrier fan-in = this core's mcast coverage (a core only writes where it
            // mcasts, so it only needs those targets ready). Sums to num_devices-1 across axes.
            const uint32_t barrier_wait_value = coverage;

            std::vector<uint32_t> reader_rt_args = {
                input_tensor.buffer()->address(),   // input tensor address
                output_tensor.buffer()->address(),  // output tensor address
                semaphore.address(),                // out_ready_sem_bank_addr (absolute address)
                barrier_semaphore.address(),        // barrier_sem
                input_tile_id_start,                // input_page_id_start
                input_tile_id_end,                  // input_page_id_end
                output_page_id_start,               // output page start
                output_page_in_stripe_start,        // initial position within stripe
                output_page_byte_offset,            // byte offset within output page (nonzero only in page-concat)
                num_worker_output_pages,            // number of output pages for this worker
                device_idx,                         // this device's index
                owns_out_ready_sem,                 // owns_out_ready_sem (wait+reset)
                drain_sync_core.x,                  // out_ready_sem_noc0_x
                drain_sync_core.y,                  // out_ready_sem_noc0_y
                out_ready_sem_wait_value,           // out_ready_sem_wait_value
                barrier_core.x,                     // barrier_sem_noc0_x
                barrier_core.y,                     // barrier_sem_noc0_y
                barrier_wait_value,                 // barrier_wait_value
            };
            std::vector<tt::tt_fabric::FabricNodeId> reader_dst_nodes;
            if (fwd_hop > 0 && fwd_coord.has_value()) {
                reader_dst_nodes.push_back(mesh_device->get_fabric_node_id(*fwd_coord));
            }
            if (!reader_dst_nodes.empty()) {
                append_routing_plane_connection_manager_rt_args(
                    sender_fabric_node_id,
                    reader_dst_nodes,
                    {link},
                    program,
                    reader_kernel_ids[a],
                    {core},
                    reader_rt_args,
                    fabric_is_2d ? tt::tt_fabric::FabricApiType::Mesh : tt::tt_fabric::FabricApiType::Linear);
            }

            std::vector<uint32_t> writer_rt_args = {
                output_tensor.buffer()->address(),  // output tensor address
                semaphore.address(),                // out_ready_sem_bank_addr (absolute address)
                barrier_semaphore.address(),        // barrier_sem
                output_page_id_start,               // output page start
                output_page_in_stripe_start,        // initial position within stripe
                output_page_byte_offset,            // byte offset within output page (nonzero only in page-concat)
                num_worker_output_pages,            // number of output pages for this worker
                device_idx,                         // this device's index
                drain_sync_core.x,                  // out_ready_sem_noc0_x
                drain_sync_core.y,                  // out_ready_sem_noc0_y
                barrier_core.x,                     // barrier_sem_noc0_x
                barrier_core.y,                     // barrier_sem_noc0_y
            };
            std::vector<tt::tt_fabric::FabricNodeId> writer_dst_nodes;
            if (bwd_hop > 0 && bwd_coord.has_value()) {
                writer_dst_nodes.push_back(mesh_device->get_fabric_node_id(*bwd_coord));
            }
            if (!writer_dst_nodes.empty()) {
                append_routing_plane_connection_manager_rt_args(
                    sender_fabric_node_id,
                    writer_dst_nodes,
                    {link},
                    program,
                    writer_kernel_ids[a],
                    {core},
                    writer_rt_args,
                    fabric_is_2d ? tt::tt_fabric::FabricApiType::Mesh : tt::tt_fabric::FabricApiType::Linear);
            }

            tt::tt_metal::SetRuntimeArgs(program, reader_kernel_ids[a], {core}, reader_rt_args);
            tt::tt_metal::SetRuntimeArgs(program, writer_kernel_ids[a], {core}, writer_rt_args);
        }
    }

    shared_variables_t shared_variables{
        .sender_worker_cores = axis_cores,
        .worker_sender_reader_kernel_id = reader_kernel_ids,
        .worker_sender_writer_kernel_id = writer_kernel_ids,
        .semaphore = semaphore,
        .barrier_semaphore = barrier_semaphore,
        .ring_index = device_idx,
    };

    return {std::move(program), std::move(shared_variables)};
}

void AllGatherFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllGatherParams& /*operation_attributes*/,
    const AllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        log_trace(tt::LogOp, "DEBUG: semaphore: {}", shared_vars.semaphore.address());
        log_trace(tt::LogOp, "DEBUG: barrier_semaphore: {}", shared_vars.barrier_semaphore.address());
        for (uint32_t a = 0; a < 2; ++a) {
            if (shared_vars.sender_worker_cores[a].empty()) {
                continue;
            }
            auto& reader_args_by_core = GetRuntimeArgs(program, shared_vars.worker_sender_reader_kernel_id[a]);
            auto& writer_args_by_core = GetRuntimeArgs(program, shared_vars.worker_sender_writer_kernel_id[a]);
            for (const auto& core : shared_vars.sender_worker_cores[a]) {
                // reader: [0]=input_addr, [1]=output_addr, [2]=out_ready_sem, [3]=barrier_sem
                auto& reader_args = reader_args_by_core[core.x][core.y];
                reader_args[0] = tensor_args.input_tensor.buffer()->address();
                reader_args[1] = output_tensor.buffer()->address();
                reader_args[2] = shared_vars.semaphore.address();
                reader_args[3] = shared_vars.barrier_semaphore.address();
                // writer: [0]=output_addr, [1]=out_ready_sem, [2]=barrier_sem
                auto& writer_args = writer_args_by_core[core.x][core.y];
                writer_args[0] = output_tensor.buffer()->address();
                writer_args[1] = shared_vars.semaphore.address();
                writer_args[2] = shared_vars.barrier_semaphore.address();
            }
        }
    }
}

}  // namespace ttnn::experimental::prim

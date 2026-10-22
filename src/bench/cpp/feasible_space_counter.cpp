#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr std::int32_t ABI_VERSION = 1;
volatile std::sig_atomic_t interrupted = 0;

void handle_interrupt(int) {
    interrupted = 1;
}

class InterruptGuard {
public:
    InterruptGuard() : previous_(std::signal(SIGINT, handle_interrupt)) {
        interrupted = 0;
    }
    ~InterruptGuard() { std::signal(SIGINT, previous_); }

private:
    using Handler = void (*)(int);
    Handler previous_;
};

struct FeasibleSpaceInput {
    std::int32_t abi_version;
    std::uint64_t struct_size;
    std::int32_t num_rhs;
    std::int32_t num_es;
    std::int32_t num_rc;
    std::int32_t num_actions;
    const std::int32_t* action_offsets;
    const std::int32_t* action_es;
    const std::int32_t* action_rc;
    const std::int32_t* action_gateway;
    const double* action_transport_latency;
    const double* action_es_fraction;
    const double* action_rc_fraction;
    const double* es_load;
    const double* rc_load;
    const double* fronthaul_load;
    const double* midhaul_load;
    const double* direct_load;
    const double* es_capacity;
    const double* rc_capacity;
    const double* fronthaul_capacity;
    const double* midhaul_capacity;
    const double* direct_capacity;
    const double* rh_deadline_ms;
    double base_service_time_ms;
    double utilization_cap;
    double feasibility_tolerance;
};

struct FeasibleSpaceOutput {
    std::uint64_t feasible_count;
    std::uint64_t connectivity_valid_count;
    std::uint64_t rejected_by_compute;
    std::uint64_t rejected_by_bandwidth;
    std::uint64_t rejected_by_latency;
    double elapsed_seconds;
};

void copy_error(char* buffer, std::size_t size, const std::string& message) {
    if (!buffer || size == 0) return;
    std::strncpy(buffer, message.c_str(), size - 1);
    buffer[size - 1] = '\0';
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right) {
    if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right) {
        throw std::overflow_error("connectivity-valid joint-action count exceeds uint64");
    }
    return left * right;
}

void checked_add(std::uint64_t& target, std::uint64_t value) {
    if (target > std::numeric_limits<std::uint64_t>::max() - value) {
        throw std::overflow_error("feasible-space category count exceeds uint64");
    }
    target += value;
}

void validate(const FeasibleSpaceInput& input) {
    if (input.abi_version != ABI_VERSION || input.struct_size != sizeof(FeasibleSpaceInput)) {
        throw std::invalid_argument("feasible-space ABI version or struct size mismatch");
    }
    if (input.num_rhs <= 0 || input.num_es <= 0 || input.num_rc <= 0 ||
        input.num_actions <= 0) {
        throw std::invalid_argument("feasible-space dimensions must be positive");
    }
    if (!input.action_offsets || !input.action_es || !input.action_rc ||
        !input.action_gateway || !input.action_transport_latency ||
        !input.action_es_fraction || !input.action_rc_fraction || !input.es_load ||
        !input.rc_load || !input.fronthaul_load || !input.midhaul_load ||
        !input.direct_load || !input.es_capacity || !input.rc_capacity ||
        !input.fronthaul_capacity || !input.midhaul_capacity ||
        !input.direct_capacity || !input.rh_deadline_ms) {
        throw std::invalid_argument("required feasible-space input pointer is null");
    }
    if (input.action_offsets[0] != 0 ||
        input.action_offsets[input.num_rhs] != input.num_actions) {
        throw std::invalid_argument("action offsets do not span the action array");
    }
    for (int rh = 0; rh < input.num_rhs; ++rh) {
        if (input.action_offsets[rh] >= input.action_offsets[rh + 1]) {
            throw std::invalid_argument("each RH must have a connectivity-valid action");
        }
    }
    if (!(input.base_service_time_ms > 0.0) || !(input.utilization_cap > 0.0) ||
        !(input.utilization_cap < 1.0) || input.feasibility_tolerance < 0.0) {
        throw std::invalid_argument("invalid processing-delay or tolerance scalar");
    }
}

enum class RejectionState { Normal, Bandwidth, Latency };

class Counter {
public:
    Counter(const FeasibleSpaceInput& input, FeasibleSpaceOutput& output)
        : in_(input), out_(output), order_(input.num_rhs),
          selected_(input.num_rhs, -1), es_(input.num_es, 0.0),
          rc_(input.num_rc, 0.0), fh_(input.num_rhs * input.num_es, 0.0),
          mh_(input.num_es * input.num_rc, 0.0),
          direct_(input.num_rhs * input.num_rc, 0.0) {
        std::iota(order_.begin(), order_.end(), 0);
        std::stable_sort(order_.begin(), order_.end(), [&](int left, int right) {
            const int left_count = action_count(left);
            const int right_count = action_count(right);
            if (left_count != right_count) return left_count < right_count;
            const double left_bw = maximum_bandwidth(left);
            const double right_bw = maximum_bandwidth(right);
            if (left_bw != right_bw) return left_bw > right_bw;
            return left < right;
        });
        build_suffix_data();
        out_.connectivity_valid_count = suffix_count_[0];
    }

    void run() {
        recurse(0, RejectionState::Normal);
        std::uint64_t total = 0;
        checked_add(total, out_.feasible_count);
        checked_add(total, out_.rejected_by_compute);
        checked_add(total, out_.rejected_by_bandwidth);
        checked_add(total, out_.rejected_by_latency);
        if (total != out_.connectivity_valid_count) {
            throw std::logic_error("feasible-space category counts do not sum to connectivity count");
        }
    }

    std::uint64_t connectivity_count() const {
        return suffix_count_[0];
    }

    std::uint64_t prefix_task_count(int prefix_depth) const {
        if (prefix_depth < 0 || prefix_depth > in_.num_rhs) {
            throw std::invalid_argument("parallel prefix depth is out of range");
        }
        return suffix_count_[0] / suffix_count_[prefix_depth];
    }

    void run_prefix(std::uint64_t task_index, int prefix_depth) {
        const std::uint64_t completion_count = suffix_count_[prefix_depth];
        RejectionState state = RejectionState::Normal;
        std::vector<std::pair<int, int>> applied;
        applied.reserve(prefix_depth);
        bool continue_branch = true;
        for (int depth = 0; depth < prefix_depth && continue_branch; ++depth) {
            const int rh = order_[depth];
            const int count = action_count(rh);
            const int local_action = static_cast<int>(task_index % count);
            task_index /= count;
            const int action = in_.action_offsets[rh] + local_action;
            selected_[rh] = action;
            apply(rh, action, 1.0);
            applied.emplace_back(rh, action);
            continue_branch = process_action(
                depth, action, state, completion_count
            );
        }
        if (continue_branch) recurse(prefix_depth, state);
        for (auto iterator = applied.rbegin(); iterator != applied.rend(); ++iterator) {
            apply(iterator->first, iterator->second, -1.0);
            selected_[iterator->first] = -1;
        }
    }

private:
    const FeasibleSpaceInput& in_;
    FeasibleSpaceOutput& out_;
    std::vector<int> order_;
    std::vector<int> selected_;
    std::vector<double> es_, rc_, fh_, mh_, direct_;
    std::vector<std::uint64_t> suffix_count_;
    std::vector<std::vector<double>> suffix_es_, suffix_rc_, suffix_fh_, suffix_mh_, suffix_direct_;

    int action_count(int rh) const {
        return in_.action_offsets[rh + 1] - in_.action_offsets[rh];
    }

    double maximum_bandwidth(int rh) const {
        double result = 0.0;
        for (int action = in_.action_offsets[rh]; action < in_.action_offsets[rh + 1]; ++action) {
            result = std::max(result, in_.fronthaul_load[action] +
                in_.midhaul_load[action] + in_.direct_load[action]);
        }
        return result;
    }

    void build_suffix_data() {
        const int depths = in_.num_rhs + 1;
        suffix_count_.assign(depths, 1);
        suffix_es_.assign(depths, std::vector<double>(in_.num_es, 0.0));
        suffix_rc_.assign(depths, std::vector<double>(in_.num_rc, 0.0));
        suffix_fh_.assign(depths, std::vector<double>(in_.num_rhs * in_.num_es, 0.0));
        suffix_mh_.assign(depths, std::vector<double>(in_.num_es * in_.num_rc, 0.0));
        suffix_direct_.assign(depths, std::vector<double>(in_.num_rhs * in_.num_rc, 0.0));
        for (int depth = in_.num_rhs - 1; depth >= 0; --depth) {
            const int rh = order_[depth];
            suffix_count_[depth] = checked_multiply(
                suffix_count_[depth + 1], static_cast<std::uint64_t>(action_count(rh))
            );
            suffix_es_[depth] = suffix_es_[depth + 1];
            suffix_rc_[depth] = suffix_rc_[depth + 1];
            suffix_fh_[depth] = suffix_fh_[depth + 1];
            suffix_mh_[depth] = suffix_mh_[depth + 1];
            suffix_direct_[depth] = suffix_direct_[depth + 1];
            for (int action = in_.action_offsets[rh]; action < in_.action_offsets[rh + 1]; ++action) {
                const int es = in_.action_es[action];
                const int rc = in_.action_rc[action];
                suffix_es_[depth][es] = std::max(
                    suffix_es_[depth][es], suffix_es_[depth + 1][es] + in_.es_load[action]
                );
                suffix_rc_[depth][rc] = std::max(
                    suffix_rc_[depth][rc], suffix_rc_[depth + 1][rc] + in_.rc_load[action]
                );
                if (in_.action_gateway[action]) {
                    const int fh_index = rh * in_.num_es + es;
                    const int mh_index = es * in_.num_rc + rc;
                    suffix_fh_[depth][fh_index] = std::max(
                        suffix_fh_[depth][fh_index], suffix_fh_[depth + 1][fh_index] + in_.fronthaul_load[action]
                    );
                    suffix_mh_[depth][mh_index] = std::max(
                        suffix_mh_[depth][mh_index], suffix_mh_[depth + 1][mh_index] + in_.midhaul_load[action]
                    );
                } else {
                    const int direct_index = rh * in_.num_rc + rc;
                    suffix_direct_[depth][direct_index] = std::max(
                        suffix_direct_[depth][direct_index], suffix_direct_[depth + 1][direct_index] + in_.direct_load[action]
                    );
                }
            }
        }
    }

    bool exceeds(const std::vector<double>& load, const double* capacity) const {
        for (std::size_t index = 0; index < load.size(); ++index) {
            if (load[index] > capacity[index] + in_.feasibility_tolerance) return true;
        }
        return false;
    }

    bool compute_violated() const {
        return exceeds(es_, in_.es_capacity) || exceeds(rc_, in_.rc_capacity);
    }

    bool bandwidth_violated() const {
        return exceeds(fh_, in_.fronthaul_capacity) || exceeds(mh_, in_.midhaul_capacity) ||
            exceeds(direct_, in_.direct_capacity);
    }

    bool guaranteed_compute_safe(int depth) const {
        for (int d = 0; d < in_.num_es; ++d) {
            if (es_[d] + suffix_es_[depth][d] > in_.es_capacity[d] + in_.feasibility_tolerance) return false;
        }
        for (int c = 0; c < in_.num_rc; ++c) {
            if (rc_[c] + suffix_rc_[depth][c] > in_.rc_capacity[c] + in_.feasibility_tolerance) return false;
        }
        return true;
    }

    bool guaranteed_bandwidth_safe(int depth) const {
        const auto safe = [&](const std::vector<double>& load,
                              const std::vector<double>& remaining,
                              const double* capacity) {
            for (std::size_t index = 0; index < load.size(); ++index) {
                if (load[index] + remaining[index] >
                    capacity[index] + in_.feasibility_tolerance) return false;
            }
            return true;
        };
        return safe(fh_, suffix_fh_[depth], in_.fronthaul_capacity) &&
            safe(mh_, suffix_mh_[depth], in_.midhaul_capacity) &&
            safe(direct_, suffix_direct_[depth], in_.direct_capacity);
    }

    double processing_delay(double utilization) const {
        return in_.base_service_time_ms /
            (1.0 - std::min(utilization, in_.utilization_cap));
    }

    bool assigned_latency_violated() const {
        for (int rh = 0; rh < in_.num_rhs; ++rh) {
            const int action = selected_[rh];
            if (action < 0) continue;
            const int es = in_.action_es[action];
            const int rc = in_.action_rc[action];
            const double latency = in_.action_transport_latency[action] +
                in_.action_es_fraction[action] * processing_delay(es_[es] / in_.es_capacity[es]) +
                in_.action_rc_fraction[action] * processing_delay(rc_[rc] / in_.rc_capacity[rc]);
            if (latency > in_.rh_deadline_ms[rh] + in_.feasibility_tolerance) return true;
        }
        return false;
    }

    void apply(int rh, int action, double sign) {
        const int es = in_.action_es[action];
        const int rc = in_.action_rc[action];
        es_[es] += sign * in_.es_load[action];
        rc_[rc] += sign * in_.rc_load[action];
        if (in_.action_gateway[action]) {
            fh_[rh * in_.num_es + es] += sign * in_.fronthaul_load[action];
            mh_[es * in_.num_rc + rc] += sign * in_.midhaul_load[action];
        } else {
            direct_[rh * in_.num_rc + rc] += sign * in_.direct_load[action];
        }
    }

    void classify_leaf() {
        if (compute_violated()) {
            checked_add(out_.rejected_by_compute, 1);
        } else if (bandwidth_violated()) {
            checked_add(out_.rejected_by_bandwidth, 1);
        } else if (assigned_latency_violated()) {
            checked_add(out_.rejected_by_latency, 1);
        } else {
            checked_add(out_.feasible_count, 1);
        }
    }

    bool process_action(
        int depth,
        int /* action */,
        RejectionState& state,
        std::uint64_t aggregate_count
    ) {
        const int next_depth = depth + 1;
        if (compute_violated()) {
            checked_add(out_.rejected_by_compute, aggregate_count);
            return false;
        }
        if (state != RejectionState::Bandwidth && bandwidth_violated()) {
            state = RejectionState::Bandwidth;
        }
        if (state == RejectionState::Bandwidth) {
            if (guaranteed_compute_safe(next_depth)) {
                checked_add(out_.rejected_by_bandwidth, aggregate_count);
                return false;
            }
            return true;
        }
        if (state == RejectionState::Normal && assigned_latency_violated()) {
            state = RejectionState::Latency;
        }
        if (state == RejectionState::Latency &&
            guaranteed_compute_safe(next_depth) &&
            guaranteed_bandwidth_safe(next_depth)) {
            checked_add(out_.rejected_by_latency, aggregate_count);
            return false;
        }
        return true;
    }

    void recurse(int depth, RejectionState state) {
        if (interrupted) throw std::runtime_error("feasible-space count interrupted");
        if (depth == in_.num_rhs) {
            classify_leaf();
            return;
        }
        const int rh = order_[depth];
        for (int action = in_.action_offsets[rh]; action < in_.action_offsets[rh + 1]; ++action) {
            selected_[rh] = action;
            apply(rh, action, 1.0);
            RejectionState next_state = state;
            const int next_depth = depth + 1;
            if (process_action(
                depth, action, next_state, suffix_count_[next_depth]
            )) {
                recurse(next_depth, next_state);
            }
            apply(rh, action, -1.0);
            selected_[rh] = -1;
        }
    }
};

}  // namespace

extern "C" int oran_count_feasible_space(
    const FeasibleSpaceInput* input,
    FeasibleSpaceOutput* output,
    char* error_buffer,
    std::size_t error_buffer_size
) {
    if (!input || !output) {
        copy_error(error_buffer, error_buffer_size, "input and output pointers are required");
        return 1;
    }
    *output = FeasibleSpaceOutput{};
    const auto started = std::chrono::steady_clock::now();
    try {
        InterruptGuard interrupt_guard;
        validate(*input);
        FeasibleSpaceOutput planning_output{};
        Counter planning_counter(*input, planning_output);
        output->connectivity_valid_count = planning_counter.connectivity_count();
        const int prefix_depth = std::min(3, input->num_rhs);
        const std::uint64_t task_count = planning_counter.prefix_task_count(prefix_depth);
        if (task_count > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            throw std::overflow_error("parallel feasible-space task count exceeds int64");
        }
        struct alignas(64) TaskResult {
            FeasibleSpaceOutput counts{};
        };
        std::vector<TaskResult> task_results(static_cast<std::size_t>(task_count));
        std::atomic<bool> failed{false};
        std::exception_ptr first_error;

#pragma omp parallel for schedule(dynamic, 1)
        for (std::int64_t task = 0; task < static_cast<std::int64_t>(task_count); ++task) {
            if (failed.load(std::memory_order_relaxed)) continue;
            try {
                Counter local_counter(*input, task_results[task].counts);
                local_counter.run_prefix(static_cast<std::uint64_t>(task), prefix_depth);
            } catch (...) {
#pragma omp critical(oran_feasible_space_error)
                {
                    if (!first_error) first_error = std::current_exception();
                    failed.store(true, std::memory_order_relaxed);
                }
            }
        }
        if (first_error) std::rethrow_exception(first_error);
        for (const TaskResult& task : task_results) {
            checked_add(output->feasible_count, task.counts.feasible_count);
            checked_add(output->rejected_by_compute, task.counts.rejected_by_compute);
            checked_add(output->rejected_by_bandwidth, task.counts.rejected_by_bandwidth);
            checked_add(output->rejected_by_latency, task.counts.rejected_by_latency);
        }
        std::uint64_t classified = 0;
        checked_add(classified, output->feasible_count);
        checked_add(classified, output->rejected_by_compute);
        checked_add(classified, output->rejected_by_bandwidth);
        checked_add(classified, output->rejected_by_latency);
        if (classified != output->connectivity_valid_count) {
            throw std::logic_error(
                "parallel feasible-space counts do not sum to connectivity count"
            );
        }
        output->elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started
        ).count();
        return 0;
    } catch (const std::overflow_error& error) {
        copy_error(error_buffer, error_buffer_size, error.what());
        return 2;
    } catch (const std::runtime_error& error) {
        if (interrupted) {
            copy_error(error_buffer, error_buffer_size, error.what());
            return 4;
        }
        copy_error(error_buffer, error_buffer_size, error.what());
        return 3;
    } catch (const std::exception& error) {
        copy_error(error_buffer, error_buffer_size, error.what());
        return 3;
    }
}

extern "C" std::size_t oran_sizeof_feasible_space_input() {
    return sizeof(FeasibleSpaceInput);
}

extern "C" std::size_t oran_sizeof_feasible_space_output() {
    return sizeof(FeasibleSpaceOutput);
}

extern "C" std::int32_t oran_feasible_space_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

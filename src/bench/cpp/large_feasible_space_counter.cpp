#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr std::int32_t ABI_VERSION = 1;
constexpr std::uint64_t MINIMUM_SAMPLES = 10000;
constexpr std::uint64_t BATCH_SIZE = 10000;
constexpr std::uint64_t MAXIMUM_SAMPLES = 1000000;
constexpr double WILSON_Z = 1.959963984540054;
constexpr double TARGET_HALF_WIDTH = 0.0025;
volatile std::sig_atomic_t interrupted = 0;

struct LargeFeasibleSpaceInput {
    std::int32_t abi_version;
    std::uint64_t struct_size;
    std::int64_t episode_id;
    std::int64_t timestep;
    std::uint64_t sampling_seed;
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

struct LargeFeasibleSpaceOutput {
    std::uint64_t samples_used;
    std::uint64_t sampled_feasible;
    std::uint64_t sampled_rejected_by_compute;
    std::uint64_t sampled_rejected_by_bandwidth;
    std::uint64_t sampled_rejected_by_latency;
    double estimated_feasible_ratio;
    double feasible_ratio_ci_lower;
    double feasible_ratio_ci_upper;
    std::int32_t precision_target_reached;
    double elapsed_seconds;
};

void handle_interrupt(int) { interrupted = 1; }

class InterruptGuard {
public:
    InterruptGuard() : previous_(std::signal(SIGINT, handle_interrupt)) { interrupted = 0; }
    ~InterruptGuard() { std::signal(SIGINT, previous_); }
private:
    using Handler = void (*)(int);
    Handler previous_;
};

void copy_error(char* buffer, std::size_t size, const std::string& message) {
    if (!buffer || size == 0) return;
    std::strncpy(buffer, message.c_str(), size - 1);
    buffer[size - 1] = '\0';
}

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::uint64_t random_word(
    const LargeFeasibleSpaceInput& input,
    std::uint64_t sample,
    std::uint64_t rh,
    std::uint64_t retry
) {
    std::uint64_t key = splitmix64(input.sampling_seed);
    key = splitmix64(key ^ static_cast<std::uint64_t>(input.episode_id));
    key = splitmix64(key ^ static_cast<std::uint64_t>(input.timestep));
    key = splitmix64(key ^ sample);
    key = splitmix64(key ^ rh);
    return splitmix64(key ^ retry);
}

std::uint64_t uniform_bounded(
    const LargeFeasibleSpaceInput& input,
    std::uint64_t sample,
    std::uint64_t rh,
    std::uint64_t bound
) {
    const std::uint64_t threshold = (0ULL - bound) % bound;
    for (std::uint64_t retry = 0;; ++retry) {
        const std::uint64_t word = random_word(input, sample, rh, retry);
        const unsigned __int128 product =
            static_cast<unsigned __int128>(word) * static_cast<unsigned __int128>(bound);
        const std::uint64_t low = static_cast<std::uint64_t>(product);
        if (low >= threshold) return static_cast<std::uint64_t>(product >> 64U);
    }
}

void wilson(
    std::uint64_t successes,
    std::uint64_t samples,
    double& lower,
    double& upper
) {
    const double n = static_cast<double>(samples);
    const double p = static_cast<double>(successes) / n;
    const double z2 = WILSON_Z * WILSON_Z;
    const double denominator = 1.0 + z2 / n;
    const double center = (p + z2 / (2.0 * n)) / denominator;
    const double radius = WILSON_Z * std::sqrt(
        p * (1.0 - p) / n + z2 / (4.0 * n * n)
    ) / denominator;
    lower = std::max(0.0, center - radius);
    upper = std::min(1.0, center + radius);
}

void validate(const LargeFeasibleSpaceInput& in) {
    if (in.abi_version != ABI_VERSION || in.struct_size != sizeof(LargeFeasibleSpaceInput))
        throw std::invalid_argument("large feasible-space ABI version or struct size mismatch");
    if (in.num_rhs <= 0 || in.num_es <= 0 || in.num_rc <= 0 || in.num_actions <= 0)
        throw std::invalid_argument("large feasible-space dimensions must be positive");
    if (!in.action_offsets || !in.action_es || !in.action_rc || !in.action_gateway ||
        !in.action_transport_latency || !in.action_es_fraction || !in.action_rc_fraction ||
        !in.es_load || !in.rc_load || !in.fronthaul_load || !in.midhaul_load ||
        !in.direct_load || !in.es_capacity || !in.rc_capacity ||
        !in.fronthaul_capacity || !in.midhaul_capacity || !in.direct_capacity ||
        !in.rh_deadline_ms)
        throw std::invalid_argument("required large feasible-space input pointer is null");
    if (in.action_offsets[0] != 0 || in.action_offsets[in.num_rhs] != in.num_actions)
        throw std::invalid_argument("action offsets do not span the action array");
    for (int rh = 0; rh < in.num_rhs; ++rh) {
        if (in.action_offsets[rh] >= in.action_offsets[rh + 1])
            throw std::invalid_argument("each RH must have a connectivity-valid action");
    }
    if (!(in.base_service_time_ms > 0.0) || !(in.utilization_cap > 0.0) ||
        !(in.utilization_cap < 1.0) || in.feasibility_tolerance < 0.0)
        throw std::invalid_argument("invalid processing-delay or tolerance scalar");
}

enum class Category : int { Feasible, Compute, Bandwidth, Latency };

Category classify(
    const LargeFeasibleSpaceInput& in,
    std::uint64_t sample,
    std::vector<int>& selected,
    std::vector<double>& es,
    std::vector<double>& rc,
    std::vector<double>& fh,
    std::vector<double>& mh,
    std::vector<double>& direct
) {
    std::fill(es.begin(), es.end(), 0.0);
    std::fill(rc.begin(), rc.end(), 0.0);
    std::fill(fh.begin(), fh.end(), 0.0);
    std::fill(mh.begin(), mh.end(), 0.0);
    std::fill(direct.begin(), direct.end(), 0.0);
    for (int rh = 0; rh < in.num_rhs; ++rh) {
        const int begin = in.action_offsets[rh];
        const std::uint64_t count = static_cast<std::uint64_t>(in.action_offsets[rh + 1] - begin);
        const int action = begin + static_cast<int>(uniform_bounded(in, sample, rh, count));
        selected[rh] = action;
        const int es_id = in.action_es[action];
        const int rc_id = in.action_rc[action];
        es[es_id] += in.es_load[action];
        rc[rc_id] += in.rc_load[action];
        if (in.action_gateway[action]) {
            fh[rh * in.num_es + es_id] += in.fronthaul_load[action];
            mh[es_id * in.num_rc + rc_id] += in.midhaul_load[action];
        } else {
            direct[rh * in.num_rc + rc_id] += in.direct_load[action];
        }
    }
    const auto exceeds = [&](const std::vector<double>& load, const double* capacity) {
        for (std::size_t i = 0; i < load.size(); ++i)
            if (load[i] > capacity[i] + in.feasibility_tolerance) return true;
        return false;
    };
    if (exceeds(es, in.es_capacity) || exceeds(rc, in.rc_capacity)) return Category::Compute;
    if (exceeds(fh, in.fronthaul_capacity) || exceeds(mh, in.midhaul_capacity) ||
        exceeds(direct, in.direct_capacity)) return Category::Bandwidth;
    const auto delay = [&](double load, double capacity) {
        return in.base_service_time_ms /
            (1.0 - std::min(load / capacity, in.utilization_cap));
    };
    for (int rh = 0; rh < in.num_rhs; ++rh) {
        const int action = selected[rh];
        const int es_id = in.action_es[action];
        const int rc_id = in.action_rc[action];
        const double latency = in.action_transport_latency[action] +
            in.action_es_fraction[action] * delay(es[es_id], in.es_capacity[es_id]) +
            in.action_rc_fraction[action] * delay(rc[rc_id], in.rc_capacity[rc_id]);
        if (latency > in.rh_deadline_ms[rh] + in.feasibility_tolerance)
            return Category::Latency;
    }
    return Category::Feasible;
}

}  // namespace

extern "C" int oran_estimate_large_feasible_space(
    const LargeFeasibleSpaceInput* input,
    LargeFeasibleSpaceOutput* output,
    char* error_buffer,
    std::size_t error_buffer_size
) {
    if (!input || !output) {
        copy_error(error_buffer, error_buffer_size, "input and output pointers are required");
        return 1;
    }
    *output = LargeFeasibleSpaceOutput{};
    const auto started = std::chrono::steady_clock::now();
    try {
        InterruptGuard guard;
        validate(*input);
        while (output->samples_used < MAXIMUM_SAMPLES) {
            if (interrupted) return 4;
            const std::uint64_t begin = output->samples_used;
            const std::uint64_t end = std::min(begin + BATCH_SIZE, MAXIMUM_SAMPLES);
            std::uint64_t feasible = 0, compute = 0, bandwidth = 0, latency = 0;
#pragma omp parallel reduction(+:feasible,compute,bandwidth,latency)
            {
                std::vector<int> selected(input->num_rhs);
                std::vector<double> es(input->num_es), rc(input->num_rc);
                std::vector<double> fh(input->num_rhs * input->num_es);
                std::vector<double> mh(input->num_es * input->num_rc);
                std::vector<double> direct(input->num_rhs * input->num_rc);
#pragma omp for schedule(static)
                for (std::uint64_t sample = begin; sample < end; ++sample) {
                    switch (classify(*input, sample, selected, es, rc, fh, mh, direct)) {
                        case Category::Feasible: ++feasible; break;
                        case Category::Compute: ++compute; break;
                        case Category::Bandwidth: ++bandwidth; break;
                        case Category::Latency: ++latency; break;
                    }
                }
            }
            output->sampled_feasible += feasible;
            output->sampled_rejected_by_compute += compute;
            output->sampled_rejected_by_bandwidth += bandwidth;
            output->sampled_rejected_by_latency += latency;
            output->samples_used = end;
            wilson(output->sampled_feasible, output->samples_used,
                   output->feasible_ratio_ci_lower, output->feasible_ratio_ci_upper);
            output->estimated_feasible_ratio =
                static_cast<double>(output->sampled_feasible) /
                static_cast<double>(output->samples_used);
            const double half_width =
                (output->feasible_ratio_ci_upper - output->feasible_ratio_ci_lower) / 2.0;
            if (output->samples_used >= MINIMUM_SAMPLES && half_width <= TARGET_HALF_WIDTH) {
                output->precision_target_reached = 1;
                break;
            }
        }
        output->elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started
        ).count();
        return 0;
    } catch (const std::exception& error) {
        copy_error(error_buffer, error_buffer_size, error.what());
        return 1;
    }
}

extern "C" std::size_t oran_sizeof_large_feasible_space_input() {
    return sizeof(LargeFeasibleSpaceInput);
}

extern "C" std::size_t oran_sizeof_large_feasible_space_output() {
    return sizeof(LargeFeasibleSpaceOutput);
}

extern "C" std::int32_t oran_large_feasible_space_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

constexpr std::int32_t ORAN_MILP_ABI_VERSION = 3;

enum OranOptimizationMode : std::int32_t {
    ORAN_MIRROR_SOFT = 0,
    ORAN_HARD_SERVICE_FEASIBILITY = 1,
};

struct OranMilpInput {
    std::int32_t abi_version;
    std::uint64_t struct_size;
    std::int32_t num_times;
    std::int32_t num_rhs;
    std::int32_t num_es;
    std::int32_t num_rc;
    std::int32_t num_actions;
    std::int32_t num_upper_points;
    std::int64_t num_ho_terms;

    const std::int32_t* action_offsets;
    const std::int32_t* action_owner;
    const std::int32_t* action_split;
    const std::int32_t* action_es;
    const std::int32_t* action_rc;
    const std::int32_t* action_gateway;
    const double* placement_cost;
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

    const double* upper_rho_points;
    const double* upper_values;
    const double* es_utilization_upper;
    const double* rc_utilization_upper;
    double utilization_cap;

    const std::int32_t* ho_time;
    const std::int32_t* ho_action_a;
    const std::int32_t* ho_action_b;
    const double* ho_state_cost;
    const double* ho_sla_cost;
    const double* ho_control_cost;
    double compute_weight;
    double bandwidth_weight;
    double latency_weight;
    std::int32_t reconf_enabled;
    double split_change_weight;
    double es_change_weight;
    double rc_change_weight;
    const double* fixed_reconf_cost;
    const std::int32_t* mip_start_action;
    std::int64_t mip_start_size;
};

struct OranMilpOptions {
    double time_limit_seconds;
    double relative_mip_gap;
    std::int32_t threads;
    std::int32_t deterministic;
    std::int32_t random_seed;
    std::int32_t mip_emphasis;
    std::int32_t optimization_mode;
};

struct OranMilpOutput {
    std::int32_t status_code;
    std::int32_t is_proven_optimal;
    std::int32_t has_incumbent;
    double objective_value;
    double best_bound;
    double relative_gap;
    double build_time_seconds;
    double solve_time_seconds;
    std::int64_t num_binary_variables;
    std::int64_t num_continuous_variables;
    std::int64_t num_constraints;
    std::int64_t num_nonzero_coefficients;
    std::int64_t node_count;
    std::int64_t remaining_node_count;
    std::int32_t cplex_status_code;
    double resolved_time_limit_seconds;
    double resolved_relative_mip_gap;
    std::int32_t resolved_threads;
    std::int32_t resolved_random_seed;
    std::int32_t resolved_parallel_mode;
    std::int32_t resolved_mip_emphasis;
    std::int32_t* selected_action;
    std::int64_t selected_action_size;
    double component_costs[8];
    char status_name[128];
    char cplex_status_name[128];
    char termination_reason[128];
    char cplex_version[64];
    char mip_start_status[64];
};

int oran_cplex_solve(
    const OranMilpInput* input,
    const OranMilpOptions* options,
    OranMilpOutput* output,
    char* error_buffer,
    std::size_t error_buffer_size
);

void oran_cplex_free_output(OranMilpOutput* output);

std::size_t oran_sizeof_milp_input();
std::size_t oran_sizeof_milp_options();
std::size_t oran_sizeof_milp_output();

}

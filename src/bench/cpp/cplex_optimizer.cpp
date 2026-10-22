#include "cplex_optimizer.h"

#include <ilcplex/ilocplex.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <limits>
#include <sstream>
#include <streambuf>
#include <stdexcept>
#include <string>
#include <vector>

ILOSTLBEGIN

namespace {

class TeeStreamBuf final : public std::streambuf {
public:
    TeeStreamBuf(std::streambuf* first, std::streambuf* second)
        : first_(first), second_(second) {}

protected:
    int overflow(int character) override {
        if (traits_type::eq_int_type(character, traits_type::eof())) {
            return traits_type::not_eof(character);
        }
        const char value = traits_type::to_char_type(character);
        const bool first_ok = !traits_type::eq_int_type(
            first_->sputc(value), traits_type::eof()
        );
        const bool second_ok = !traits_type::eq_int_type(
            second_->sputc(value), traits_type::eof()
        );
        return first_ok && second_ok ? character : traits_type::eof();
    }

    int sync() override {
        return first_->pubsync() == 0 && second_->pubsync() == 0 ? 0 : -1;
    }

private:
    std::streambuf* first_;
    std::streambuf* second_;
};

}  // namespace

namespace {

using Clock = std::chrono::steady_clock;

double elapsed(const Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void copy_text(char* target, std::size_t size, const std::string& value) {
    if (target == nullptr || size == 0) return;
    std::strncpy(target, value.c_str(), size - 1);
    target[size - 1] = '\0';
}

int fail(int code, const std::string& message, char* buffer, std::size_t size) {
    copy_text(buffer, size, message);
    return code;
}

void validate_input(const OranMilpInput& in) {
    if (in.abi_version != ORAN_MILP_ABI_VERSION || in.struct_size != sizeof(OranMilpInput)) {
        throw std::invalid_argument("MILP input ABI version or size mismatch");
    }
    if (in.num_times <= 0 || in.num_rhs <= 0 || in.num_es <= 0 || in.num_rc <= 0 ||
        in.num_actions <= 0 || in.num_upper_points < 2 ||
        in.num_ho_terms < 0) {
        throw std::invalid_argument("MILP dimensions must be positive");
    }
    if (!in.action_offsets || !in.action_owner || !in.action_split || !in.action_es ||
        !in.action_rc || !in.action_gateway || !in.placement_cost ||
        !in.action_transport_latency || !in.action_es_fraction || !in.action_rc_fraction ||
        !in.es_load || !in.rc_load || !in.fronthaul_load || !in.midhaul_load ||
        !in.direct_load || !in.es_capacity || !in.rc_capacity ||
        !in.fronthaul_capacity || !in.midhaul_capacity || !in.direct_capacity ||
        !in.rh_deadline_ms || !in.upper_rho_points || !in.upper_values || !in.es_utilization_upper ||
        !in.rc_utilization_upper || !in.fixed_reconf_cost) {
        throw std::invalid_argument("required MILP input pointer is null");
    }
    if (in.action_offsets[0] != 0 || in.action_offsets[in.num_rhs] != in.num_actions) {
        throw std::invalid_argument("action offsets are inconsistent");
    }
    for (int r = 0; r < in.num_rhs; ++r) {
        if (in.action_offsets[r] >= in.action_offsets[r + 1]) {
            throw std::invalid_argument("each RH must have a valid action");
        }
    }
    for (int i = 1; i < in.num_upper_points; ++i) {
        if (!(in.upper_rho_points[i] > in.upper_rho_points[i - 1])) {
            throw std::invalid_argument("upper PWL rho points must be strictly increasing");
        }
    }
    if (in.upper_rho_points[0] > 1e-12 || in.utilization_cap <= 0.0) {
        throw std::invalid_argument("processing-delay PWL scalars are invalid");
    }
    if (in.num_ho_terms && (!in.ho_time || !in.ho_action_a || !in.ho_action_b ||
        !in.ho_state_cost || !in.ho_sla_cost || !in.ho_control_cost)) {
        throw std::invalid_argument("handover arrays are null");
    }
    const std::int64_t expected_start = static_cast<std::int64_t>(in.num_times) * in.num_rhs;
    if (in.mip_start_size != 0 && in.mip_start_size != expected_start) {
        throw std::invalid_argument("MIP start size is inconsistent");
    }
}

double transition_cost(const OranMilpInput& in, int a, int b) {
    double cost = in.split_change_weight * (in.action_split[a] != in.action_split[b]);
    cost += in.rc_change_weight * (in.action_rc[a] != in.action_rc[b]);
    if (in.action_gateway[a] && in.action_gateway[b]) {
        cost += in.es_change_weight * (in.action_es[a] != in.action_es[b]);
    }
    return cost;
}

}  // namespace

extern "C" int oran_cplex_solve(
    const OranMilpInput* input,
    const OranMilpOptions* options,
    OranMilpOutput* output,
    char* error_buffer,
    std::size_t error_buffer_size
) {
    if (!input || !options || !output) {
        return fail(1, "input, options, and output must be non-null", error_buffer, error_buffer_size);
    }
    std::memset(output, 0, sizeof(*output));
    output->objective_value = std::numeric_limits<double>::quiet_NaN();
    output->best_bound = std::numeric_limits<double>::quiet_NaN();
    output->relative_gap = std::numeric_limits<double>::quiet_NaN();

    try {
        validate_input(*input);
        if (options->optimization_mode < ORAN_MIRROR_SOFT ||
            options->optimization_mode > ORAN_HARD_SERVICE_FEASIBILITY) {
            throw std::invalid_argument("unknown optimization mode");
        }
        const bool hard_service = options->optimization_mode == ORAN_HARD_SERVICE_FEASIBILITY;

        const auto build_start = Clock::now();
        IloEnv env;
        try {
            IloModel model(env);
            const int T = input->num_times;
            const int R = input->num_rhs;
            const int D = input->num_es;
            const int C = input->num_rc;
            const int K = input->num_actions;
            // Every processing-delay variable is explicitly bounded by this
            // value below.  It provides a safe finite bound for the exact
            // big-M hard-deadline rows constructed per action.
            const double max_processing_delay =
                input->upper_values[input->num_upper_points - 1];
            auto x_index = [K](int t, int a) { return t * K + a; };
            auto es_index = [D](int t, int d) { return t * D + d; };
            auto rc_index = [C](int t, int c) { return t * C + c; };

            IloBoolVarArray x(env, T * K);
            IloNumVarArray es_rho(env), rc_rho(env), es_delay(env), rc_delay(env);
            IloNumVarArray ho_aux(env), reconf_aux(env);
            std::vector<int> reconf_time, reconf_action_a, reconf_action_b;
            IloExpr placement(env), compute(env), bandwidth(env), latency(env);
            IloExpr ho_state(env), ho_sla(env), ho_control(env), reconf(env);
            IloNumArray upper_points(env, input->num_upper_points);
            IloNumArray upper_values(env, input->num_upper_points);
            for (int i = 0; i < input->num_upper_points; ++i) {
                upper_points[i] = input->upper_rho_points[i];
                upper_values[i] = input->upper_values[i];
            }

            for (int t = 0; t < T; ++t) {
                for (int d = 0; d < D; ++d) {
                    const double ub = input->es_utilization_upper[es_index(t, d)];
                    es_rho.add(IloNumVar(env, 0.0, ub, ILOFLOAT));
                    es_delay.add(IloNumVar(env, 0.0, input->upper_values[input->num_upper_points - 1], ILOFLOAT));
                }
                for (int c = 0; c < C; ++c) {
                    const double ub = input->rc_utilization_upper[rc_index(t, c)];
                    rc_rho.add(IloNumVar(env, 0.0, ub, ILOFLOAT));
                    rc_delay.add(IloNumVar(env, 0.0, input->upper_values[input->num_upper_points - 1], ILOFLOAT));
                }
            }

            const double first_upper_slope =
                (input->upper_values[1] - input->upper_values[0]) /
                (input->upper_rho_points[1] - input->upper_rho_points[0]);
            auto add_delay_model = [&](IloNumVar rho, IloNumVar delay, double ub) {
                if (ub <= input->utilization_cap + 1e-12) {
                    for (int i = 0; i + 1 < input->num_upper_points; ++i) {
                        const double left = input->upper_rho_points[i];
                        const double right = input->upper_rho_points[i + 1];
                        if (left >= input->utilization_cap - 1e-12) {
                            break;
                        }
                        const double slope =
                            (input->upper_values[i + 1] - input->upper_values[i]) /
                            (right - left);
                        const double intercept = input->upper_values[i] - slope * left;
                        model.add(delay >= slope * rho + intercept);
                        if (right >= ub - 1e-12) {
                            break;
                        }
                    }
                    return;
                }
                model.add(delay == IloPiecewiseLinear(
                    rho, first_upper_slope, upper_points, upper_values, 0.0
                ));
            };

            for (int t = 0; t < T; ++t) {
                for (int r = 0; r < R; ++r) {
                    IloExpr one(env);
                    for (int a = input->action_offsets[r]; a < input->action_offsets[r + 1]; ++a) {
                        one += x[x_index(t, a)];
                        placement += input->placement_cost[a] * x[x_index(t, a)];
                        reconf += input->fixed_reconf_cost[a] * x[x_index(t, a)];
                    }
                    model.add(one == 1.0);
                    one.end();
                }

                for (int d = 0; d < D; ++d) {
                    IloExpr load(env);
                    for (int a = 0; a < K; ++a) {
                        if (input->action_es[a] == d && input->es_load[t * K + a] != 0.0) {
                            load += input->es_load[t * K + a] * x[x_index(t, a)];
                        }
                    }
                    model.add(input->es_capacity[d] * es_rho[es_index(t, d)] == load);
                    if (hard_service) {
                        model.add(load <= input->es_capacity[d]);
                    } else {
                        IloNumVar slack(env, 0.0, IloInfinity, ILOFLOAT);
                        model.add(slack >= load - input->es_capacity[d]);
                        compute += (input->compute_weight / input->es_capacity[d]) * slack;
                    }
                    add_delay_model(
                        es_rho[es_index(t, d)], es_delay[es_index(t, d)],
                        input->es_utilization_upper[es_index(t, d)]
                    );
                    load.end();
                }
                for (int c = 0; c < C; ++c) {
                    IloExpr load(env);
                    for (int a = 0; a < K; ++a) {
                        if (input->action_rc[a] == c && input->rc_load[t * K + a] != 0.0) {
                            load += input->rc_load[t * K + a] * x[x_index(t, a)];
                        }
                    }
                    model.add(input->rc_capacity[c] * rc_rho[rc_index(t, c)] == load);
                    if (hard_service) {
                        model.add(load <= input->rc_capacity[c]);
                    } else {
                        IloNumVar slack(env, 0.0, IloInfinity, ILOFLOAT);
                        model.add(slack >= load - input->rc_capacity[c]);
                        compute += (input->compute_weight / input->rc_capacity[c]) * slack;
                    }
                    add_delay_model(
                        rc_rho[rc_index(t, c)], rc_delay[rc_index(t, c)],
                        input->rc_utilization_upper[rc_index(t, c)]
                    );
                    load.end();
                }

                for (int r = 0; r < R; ++r) {
                    for (int d = 0; d < D; ++d) {
                        IloExpr load(env);
                        for (int a = input->action_offsets[r]; a < input->action_offsets[r + 1]; ++a) {
                            if (input->action_gateway[a] && input->action_es[a] == d &&
                                input->fronthaul_load[t * K + a] != 0.0) {
                                load += input->fronthaul_load[t * K + a] * x[x_index(t, a)];
                            }
                        }
                        if (hard_service) {
                            model.add(load <= input->fronthaul_capacity[r * D + d]);
                        } else {
                            IloNumVar slack(env, 0.0, IloInfinity, ILOFLOAT);
                            model.add(slack >= load - input->fronthaul_capacity[r * D + d]);
                            if (input->fronthaul_capacity[r * D + d] > 0.0) {
                                bandwidth += (input->bandwidth_weight /
                                    input->fronthaul_capacity[r * D + d]) * slack;
                            }
                        }
                        load.end();
                    }
                    for (int c = 0; c < C; ++c) {
                        IloExpr load(env);
                        for (int a = input->action_offsets[r]; a < input->action_offsets[r + 1]; ++a) {
                            if (!input->action_gateway[a] && input->action_rc[a] == c &&
                                input->direct_load[t * K + a] != 0.0) {
                                load += input->direct_load[t * K + a] * x[x_index(t, a)];
                            }
                        }
                        if (hard_service) {
                            model.add(load <= input->direct_capacity[r * C + c]);
                        } else {
                            IloNumVar slack(env, 0.0, IloInfinity, ILOFLOAT);
                            model.add(slack >= load - input->direct_capacity[r * C + c]);
                            if (input->direct_capacity[r * C + c] > 0.0) {
                                bandwidth += (input->bandwidth_weight /
                                    input->direct_capacity[r * C + c]) * slack;
                            }
                        }
                        load.end();
                    }
                }
                for (int d = 0; d < D; ++d) {
                    for (int c = 0; c < C; ++c) {
                        IloExpr load(env);
                        for (int a = 0; a < K; ++a) {
                            if (input->action_gateway[a] && input->action_es[a] == d &&
                                input->action_rc[a] == c && input->midhaul_load[t * K + a] != 0.0) {
                                load += input->midhaul_load[t * K + a] * x[x_index(t, a)];
                            }
                        }
                        if (hard_service) {
                            model.add(load <= input->midhaul_capacity[d * C + c]);
                        } else {
                            IloNumVar slack(env, 0.0, IloInfinity, ILOFLOAT);
                            model.add(slack >= load - input->midhaul_capacity[d * C + c]);
                            if (input->midhaul_capacity[d * C + c] > 0.0) {
                                bandwidth += (input->bandwidth_weight /
                                    input->midhaul_capacity[d * C + c]) * slack;
                            }
                        }
                        load.end();
                    }
                }

                for (int r = 0; r < R; ++r) {
                    IloNumVar e2e;
                    IloNumVar excess;
                    double e2e_upper = 0.0;
                    if (!hard_service) {
                        // Exactly one action is selected for this RH.  Its
                        // transport and processing components have finite
                        // bounds, so the selected E2E delay has this finite
                        // implied upper bound.  Declaring it prevents the
                        // soft-mode equality indicators from introducing a
                        // very large auxiliary binary formulation.
                        for (int a = input->action_offsets[r]; a < input->action_offsets[r + 1]; ++a) {
                            const double action_latency_upper =
                                input->action_transport_latency[a]
                                + input->action_es_fraction[a] * max_processing_delay
                                + input->action_rc_fraction[a] * max_processing_delay;
                            e2e_upper = std::max(e2e_upper, action_latency_upper);
                        }
                        e2e = IloNumVar(env, 0.0, e2e_upper, ILOFLOAT);
                        excess = IloNumVar(env, 0.0, IloInfinity, ILOFLOAT);
                        model.add(excess >= e2e - input->rh_deadline_ms[r]);
                        latency += (input->latency_weight / input->rh_deadline_ms[r]) * excess;
                    }
                    for (int a = input->action_offsets[r]; a < input->action_offsets[r + 1]; ++a) {
                        IloExpr selected_latency(env);
                        selected_latency += input->action_transport_latency[a];
                        selected_latency += input->action_es_fraction[a]
                            * es_delay[es_index(t, input->action_es[a])];
                        selected_latency += input->action_rc_fraction[a]
                            * rc_delay[rc_index(t, input->action_rc[a])];
                        const double maximum_latency =
                            input->action_transport_latency[a]
                            + input->action_es_fraction[a] * max_processing_delay
                            + input->action_rc_fraction[a] * max_processing_delay;
                        if (hard_service) {
                            // This is equivalent to
                            //   x[t,a] == 1 => selected_latency <= deadline,
                            // without Concert creating auxiliary indicator
                            // binaries.  For x[t,a] == 0, M is at least the
                            // greatest possible left-hand side minus the
                            // deadline because both processing delays have
                            // the declared upper bound above.  For x[t,a] ==
                            // 1, M vanishes and the original deadline is
                            // enforced exactly.
                            const double deadline_big_m = std::max(
                                0.0,
                                maximum_latency - input->rh_deadline_ms[r]
                            ) + 1e-9;
                            model.add(
                                selected_latency <= input->rh_deadline_ms[r]
                                + deadline_big_m * (1.0 - x[x_index(t, a)])
                            );
                        } else {
                            // This pair is equivalent to
                            //   x[t,a] == 1 => e2e == selected_latency.
                            // If x[t,a] is zero, the first row is covered by
                            // e2e <= e2e_upper and selected_latency >= its
                            // nonnegative transport component; the second is
                            // covered by selected_latency <= maximum_latency
                            // and e2e >= 0.  If x[t,a] is one, both M terms
                            // vanish and enforce the original equality.
                            const double equality_big_m = std::max(
                                e2e_upper - input->action_transport_latency[a],
                                maximum_latency
                            ) + 1e-9;
                            model.add(
                                e2e - selected_latency
                                <= equality_big_m * (1.0 - x[x_index(t, a)])
                            );
                            model.add(
                                selected_latency - e2e
                                <= equality_big_m * (1.0 - x[x_index(t, a)])
                            );
                        }
                        selected_latency.end();
                    }
                }
            }

            for (std::int64_t i = 0; i < input->num_ho_terms; ++i) {
                const int t = input->ho_time[i];
                const int a = input->ho_action_a[i];
                const int b = input->ho_action_b[i];
                IloNumVar z(env, 0.0, 1.0, ILOFLOAT);
                ho_aux.add(z);
                model.add(z >= x[x_index(t, a)] + x[x_index(t, b)] - 1.0);
                ho_state += input->ho_state_cost[i] * z;
                ho_sla += input->ho_sla_cost[i] * z;
                ho_control += input->ho_control_cost[i] * z;
            }

            if (input->reconf_enabled) {
                for (int t = 1; t < T; ++t) {
                    for (int r = 0; r < R; ++r) {
                        for (int a = input->action_offsets[r]; a < input->action_offsets[r + 1]; ++a) {
                            for (int b = input->action_offsets[r]; b < input->action_offsets[r + 1]; ++b) {
                                const double cost = transition_cost(*input, a, b);
                                if (cost == 0.0) continue;
                                IloNumVar q(env, 0.0, 1.0, ILOFLOAT);
                                reconf_aux.add(q);
                                reconf_time.push_back(t);
                                reconf_action_a.push_back(a);
                                reconf_action_b.push_back(b);
                                model.add(q >= x[x_index(t - 1, a)] + x[x_index(t, b)] - 1.0);
                                reconf += cost * q;
                            }
                        }
                    }
                }
            }

            IloExpr objective(env);
            objective += placement + ho_state + ho_sla + ho_control + reconf;
            if (!hard_service) objective += compute + bandwidth + latency;
            model.add(IloMinimize(env, objective));

            IloCplex cplex(model);
            std::ostringstream cplex_log;
            TeeStreamBuf diagnostic_log(env.out().rdbuf(), cplex_log.rdbuf());
            std::ostream diagnostic_stream(&diagnostic_log);
            if (std::getenv("ORAN_CPLEX_DIAGNOSTIC_LOG")) {
                cplex.setOut(diagnostic_stream);
                cplex.setWarning(diagnostic_stream);
            } else {
                cplex.setOut(cplex_log);
                cplex.setWarning(cplex_log);
            }
            cplex.setParam(IloCplex::Param::MIP::Tolerances::MIPGap, options->relative_mip_gap);
            cplex.setParam(IloCplex::Param::RandomSeed, options->random_seed);
            if (options->time_limit_seconds > 0) {
                cplex.setParam(IloCplex::Param::TimeLimit, options->time_limit_seconds);
            }
            if (options->threads > 0) cplex.setParam(IloCplex::Param::Threads, options->threads);
            if (options->deterministic) {
                cplex.setParam(IloCplex::Param::Parallel, CPX_PARALLEL_DETERMINISTIC);
            }
            if (options->mip_emphasis >= 0) {
                cplex.setParam(IloCplex::Param::Emphasis::MIP, options->mip_emphasis);
            }

            output->resolved_time_limit_seconds = cplex.getParam(IloCplex::Param::TimeLimit);
            output->resolved_relative_mip_gap = cplex.getParam(IloCplex::Param::MIP::Tolerances::MIPGap);
            output->resolved_threads = cplex.getParam(IloCplex::Param::Threads);
            output->resolved_random_seed = cplex.getParam(IloCplex::Param::RandomSeed);
            output->resolved_parallel_mode = cplex.getParam(IloCplex::Param::Parallel);
            output->resolved_mip_emphasis = cplex.getParam(IloCplex::Param::Emphasis::MIP);

            copy_text(output->mip_start_status, sizeof(output->mip_start_status), "not_requested");
            if (input->mip_start_size) {
                IloNumVarArray start_vars(env);
                IloNumArray start_values(env);
                // Provide the complete assignment for every exposed
                // model variable. SolveFixed does not reliably complete a
                // start containing only the selected x=1 entries.
                std::vector<unsigned char> selected(static_cast<std::size_t>(T) * K, 0);
                for (int t = 0; t < T; ++t) {
                    for (int r = 0; r < R; ++r) {
                        selected[x_index(t, input->mip_start_action[t * R + r])] = 1;
                    }
                }
                auto delay_value = [&](double rho) {
                    if (rho <= input->upper_rho_points[0]) {
                        return input->upper_values[0]
                            + first_upper_slope * (rho - input->upper_rho_points[0]);
                    }
                    for (int i = 0; i + 1 < input->num_upper_points; ++i) {
                        if (rho <= input->upper_rho_points[i + 1] + 1e-12) {
                            const double fraction = (rho - input->upper_rho_points[i])
                                / (input->upper_rho_points[i + 1] - input->upper_rho_points[i]);
                            return input->upper_values[i]
                                + fraction * (input->upper_values[i + 1] - input->upper_values[i]);
                        }
                    }
                    return input->upper_values[input->num_upper_points - 1];
                };
                for (int t = 0; t < T; ++t) {
                    for (int a = 0; a < K; ++a) {
                        start_vars.add(x[x_index(t, a)]);
                        start_values.add(selected[x_index(t, a)] ? 1.0 : 0.0);
                    }
                    for (int d = 0; d < D; ++d) {
                        double load = 0.0;
                        for (int a = 0; a < K; ++a) {
                            if (selected[x_index(t, a)] && input->action_es[a] == d) {
                                load += input->es_load[t * K + a];
                            }
                        }
                        const double rho = load / input->es_capacity[d];
                        start_vars.add(es_rho[es_index(t, d)]);
                        start_values.add(rho);
                        start_vars.add(es_delay[es_index(t, d)]);
                        start_values.add(delay_value(rho));
                    }
                    for (int c = 0; c < C; ++c) {
                        double load = 0.0;
                        for (int a = 0; a < K; ++a) {
                            if (selected[x_index(t, a)] && input->action_rc[a] == c) {
                                load += input->rc_load[t * K + a];
                            }
                        }
                        const double rho = load / input->rc_capacity[c];
                        start_vars.add(rc_rho[rc_index(t, c)]);
                        start_values.add(rho);
                        start_vars.add(rc_delay[rc_index(t, c)]);
                        start_values.add(delay_value(rho));
                    }
                }
                for (std::int64_t i = 0; i < input->num_ho_terms; ++i) {
                    const bool active = selected[x_index(input->ho_time[i], input->ho_action_a[i])]
                        && selected[x_index(input->ho_time[i], input->ho_action_b[i])];
                    start_vars.add(ho_aux[i]);
                    start_values.add(active ? 1.0 : 0.0);
                }
                for (IloInt i = 0; i < reconf_aux.getSize(); ++i) {
                    const bool active = selected[x_index(reconf_time[i] - 1, reconf_action_a[i])]
                        && selected[x_index(reconf_time[i], reconf_action_b[i])];
                    start_vars.add(reconf_aux[i]);
                    start_values.add(active ? 1.0 : 0.0);
                }
                // The action assignment has already passed the native hard-row
                // audit in Python. SolveFixed fixes the supplied action binary
                // values and lets CPLEX complete its generated PWL/SOS and
                // objective auxiliary variables deterministically.
                cplex.addMIPStart(
                    start_vars,
                    start_values,
                    IloCplex::MIPStartSolveFixed,
                    "policy_trace"
                );
                copy_text(output->mip_start_status, sizeof(output->mip_start_status),
                          "submitted_complete_solve_fixed");
                start_values.end();
                start_vars.end();
            }

            output->build_time_seconds = elapsed(build_start);
            const auto solve_start = Clock::now();
            const bool solved = cplex.solve();
            output->solve_time_seconds = elapsed(solve_start);
            (void)solved;
            if (input->mip_start_size) {
                const std::string log_text = cplex_log.str();
                const bool accepted = log_text.find(
                    "MIP start 'policy_trace' defined initial solution"
                ) != std::string::npos;
                const bool rejected = log_text.find(
                    "No solution found from 1 MIP starts"
                ) != std::string::npos;
                copy_text(
                    output->mip_start_status,
                    sizeof(output->mip_start_status),
                    accepted ? "accepted_by_cplex"
                             : (rejected ? "rejected_by_cplex"
                                         : "acceptance_unconfirmed")
                );
            }
            output->status_code = static_cast<std::int32_t>(cplex.getStatus());
            std::ostringstream status;
            status << cplex.getStatus();
            copy_text(output->status_name, sizeof(output->status_name), status.str());
            output->cplex_status_code = static_cast<std::int32_t>(cplex.getCplexStatus());
            std::ostringstream cplex_status;
            cplex_status << cplex.getCplexStatus();
            copy_text(output->cplex_status_name, sizeof(output->cplex_status_name), cplex_status.str());
            copy_text(output->termination_reason, sizeof(output->termination_reason), cplex_status.str());
            std::ostringstream version;
            version << CPX_VERSION_VERSION << '.' << CPX_VERSION_RELEASE << '.'
                    << CPX_VERSION_MODIFICATION << '.' << CPX_VERSION_FIX;
            copy_text(output->cplex_version, sizeof(output->cplex_version), version.str());
            // the authoritative status for whether it is safe to extract and
            // report an incumbent solution.
            const bool has_incumbent = cplex.isPrimalFeasible();
            output->has_incumbent = has_incumbent ? 1 : 0;
            output->is_proven_optimal = cplex.getStatus() == IloAlgorithm::Optimal ? 1 : 0;
            output->num_binary_variables = cplex.getNbinVars();
            output->num_continuous_variables = cplex.getNcols() - cplex.getNbinVars();
            output->num_constraints = cplex.getNrows();
            output->num_nonzero_coefficients = cplex.getNNZs();
            output->node_count = static_cast<std::int64_t>(cplex.getNnodes64());
            output->remaining_node_count = static_cast<std::int64_t>(cplex.getNnodesLeft64());

            output->best_bound = cplex.getBestObjValue();
            if (has_incumbent) {
                output->objective_value = cplex.getObjValue();
                output->relative_gap = cplex.getMIPRelativeGap();
                output->selected_action_size = static_cast<std::int64_t>(T) * R;
                output->selected_action = new std::int32_t[output->selected_action_size];
                for (int t = 0; t < T; ++t) {
                    for (int r = 0; r < R; ++r) {
                        int selected = -1;
                        for (int a = input->action_offsets[r]; a < input->action_offsets[r + 1]; ++a) {
                            if (cplex.getValue(x[x_index(t, a)]) > 0.5) {
                                selected = a;
                                break;
                            }
                        }
                        if (selected < 0) throw std::runtime_error("incumbent lacks an RH assignment");
                        output->selected_action[t * R + r] = selected;
                    }
                }
                output->component_costs[0] = cplex.getValue(placement);
                output->component_costs[1] = cplex.getValue(compute);
                output->component_costs[2] = cplex.getValue(bandwidth);
                output->component_costs[3] = cplex.getValue(latency);
                output->component_costs[4] = cplex.getValue(ho_state);
                output->component_costs[5] = cplex.getValue(ho_sla);
                output->component_costs[6] = cplex.getValue(ho_control);
                output->component_costs[7] = cplex.getValue(reconf);
            }

            objective.end();
            upper_values.end();
            upper_points.end();
            reconf.end();
            ho_control.end();
            ho_sla.end();
            ho_state.end();
            latency.end();
            bandwidth.end();
            compute.end();
            placement.end();
            rc_delay.end();
            es_delay.end();
            rc_rho.end();
            es_rho.end();
            reconf_aux.end();
            ho_aux.end();
            x.end();
            env.end();
            return 0;
        } catch (...) {
            env.end();
            throw;
        }
    } catch (const IloException& exc) {
        delete[] output->selected_action;
        output->selected_action = nullptr;
        output->selected_action_size = 0;
        return fail(2, std::string("CPLEX/Concert exception: ") + exc.getMessage(), error_buffer, error_buffer_size);
    } catch (const std::exception& exc) {
        delete[] output->selected_action;
        output->selected_action = nullptr;
        output->selected_action_size = 0;
        return fail(3, std::string("C++ backend exception: ") + exc.what(), error_buffer, error_buffer_size);
    } catch (...) {
        delete[] output->selected_action;
        output->selected_action = nullptr;
        output->selected_action_size = 0;
        return fail(4, "unknown C++ backend exception", error_buffer, error_buffer_size);
    }
}

extern "C" void oran_cplex_free_output(OranMilpOutput* output) {
    if (!output) return;
    delete[] output->selected_action;
    output->selected_action = nullptr;
    output->selected_action_size = 0;
}

extern "C" std::size_t oran_sizeof_milp_input() {
    return sizeof(OranMilpInput);
}

extern "C" std::size_t oran_sizeof_milp_options() {
    return sizeof(OranMilpOptions);
}

extern "C" std::size_t oran_sizeof_milp_output() {
    return sizeof(OranMilpOutput);
}

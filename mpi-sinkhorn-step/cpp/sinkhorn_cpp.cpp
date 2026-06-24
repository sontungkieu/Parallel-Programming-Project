#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

constexpr double kTiny = 1e-12;

struct Args {
    std::string mode = "sequential";
    int rows = 0;
    int cols = 0;
    std::string cost_mode = "random";
    std::string comm_mode = "double";
    std::uint64_t seed = 0;
    double epsilon = 0.1;
    int max_iters = 200;
    double tol = 1e-6;
    int check_every = 10;
    int sync_every = 1;
    double kernel_cutoff = 0.0;
    double quantization_scale = 1e10;
    std::string output;
};

struct Result {
    int iterations = 0;
    int column_syncs = 0;
    long long column_payload_bytes = 0;
    double runtime_sec = 0.0;
    int local_rows = 0;
    double local_compute_sec = 0.0;
    double local_comm_sec = 0.0;
    double row_error = std::numeric_limits<double>::infinity();
    double col_error = std::numeric_limits<double>::infinity();
    double objective = std::numeric_limits<double>::infinity();
    std::vector<int> local_rows_by_rank;
    std::vector<double> rank_compute_sec;
    std::vector<double> rank_comm_sec;
    std::vector<double> rank_total_sec;
};

std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

double unit_random(std::uint64_t seed, int row, int col) {
    std::uint64_t x = seed;
    x ^= static_cast<std::uint64_t>(row + 1) * 0x9e3779b97f4a7c15ULL;
    x ^= static_cast<std::uint64_t>(col + 1) * 0xbf58476d1ce4e5b9ULL;
    const std::uint64_t bits = splitmix64(x) >> 11;
    return static_cast<double>(bits) * (1.0 / 9007199254740992.0);
}

double linspace_value(int index, int count) {
    if (count <= 1) {
        return 0.0;
    }
    return static_cast<double>(index) / static_cast<double>(count - 1);
}

double cost_value(int row, int col, int rows, int cols, const std::string& mode, std::uint64_t seed) {
    if (mode == "random") {
        return unit_random(seed, row, col);
    }
    if (mode == "squared_distance_1d") {
        const double x = linspace_value(row, rows);
        const double y = linspace_value(col, cols);
        const double d = x - y;
        return d * d;
    }
    if (mode == "block") {
        const int row_group = static_cast<int>(std::floor(4.0 * static_cast<double>(row) / rows));
        const int col_group = static_cast<int>(std::floor(4.0 * static_cast<double>(col) / cols));
        const double x = linspace_value(row, rows);
        const double y = linspace_value(col, cols);
        const double d = x - y;
        return (row_group != col_group ? 1.0 : 0.0) + 0.05 * d * d;
    }
    throw std::runtime_error("unsupported cost mode: " + mode);
}

std::pair<int, int> split_range(int rows, int size, int rank) {
    const int base = rows / size;
    const int extra = rows % size;
    int start = 0;
    for (int r = 0; r < rank; ++r) {
        start += base + (r < extra ? 1 : 0);
    }
    const int width = base + (rank < extra ? 1 : 0);
    return {start, start + width};
}

std::vector<double> build_cost_block(
    int global_start,
    int local_rows,
    int rows,
    int cols,
    const std::string& mode,
    std::uint64_t seed
) {
    std::vector<double> cost(static_cast<std::size_t>(local_rows) * cols);
    for (int i = 0; i < local_rows; ++i) {
        const int row = global_start + i;
        for (int j = 0; j < cols; ++j) {
            cost[static_cast<std::size_t>(i) * cols + j] = cost_value(row, j, rows, cols, mode, seed);
        }
    }
    return cost;
}

double min_value(const std::vector<double>& values) {
    if (values.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    return *std::min_element(values.begin(), values.end());
}

std::vector<double> build_kernel(
    const std::vector<double>& cost,
    double global_min,
    double epsilon,
    double cutoff
) {
    std::vector<double> kernel(cost.size());
    for (std::size_t idx = 0; idx < cost.size(); ++idx) {
        const double value = std::exp(-(cost[idx] - global_min) / epsilon);
        kernel[idx] = (cutoff > 0.0 && value < cutoff) ? 0.0 : value;
    }
    return kernel;
}

double now_seconds() {
    using clock = std::chrono::steady_clock;
    static const auto origin = clock::now();
    return std::chrono::duration<double>(clock::now() - origin).count();
}

std::string hostname() {
    char buffer[256] = {};
    if (gethostname(buffer, sizeof(buffer) - 1) != 0) {
        return "unknown";
    }
    return std::string(buffer);
}

void ensure_parent_dir(const std::string& output) {
    const std::filesystem::path path(output);
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char ch : value) {
        if (ch == '"' || ch == '\\') {
            escaped.push_back('\\');
        }
        escaped.push_back(ch);
    }
    return escaped;
}

void write_int_array(std::ofstream& out, const std::vector<int>& values) {
    out << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << values[i];
    }
    out << "]";
}

void write_double_array(std::ofstream& out, const std::vector<double>& values) {
    out << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << values[i];
    }
    out << "]";
}

void write_json(
    const Args& args,
    const Result& result,
    int num_processes,
    const std::vector<std::string>& hostnames,
    const std::string& algorithm
) {
    ensure_parent_dir(args.output);
    std::ofstream out(args.output);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + args.output);
    }
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"algorithm\": \"" << algorithm << "\",\n";
    out << "  \"check_every\": " << args.check_every << ",\n";
    out << "  \"cols\": " << args.cols << ",\n";
    out << "  \"col_error\": " << result.col_error << ",\n";
    out << "  \"column_payload_bytes\": " << result.column_payload_bytes << ",\n";
    out << "  \"column_syncs\": " << result.column_syncs << ",\n";
    out << "  \"communication_mode\": \"" << args.comm_mode << "\",\n";
    out << "  \"cost_mode\": \"" << args.cost_mode << "\",\n";
    out << "  \"epsilon\": " << args.epsilon << ",\n";
    out << "  \"hostnames\": [";
    for (std::size_t i = 0; i < hostnames.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << "\"" << json_escape(hostnames[i]) << "\"";
    }
    out << "],\n";
    out << "  \"iterations\": " << result.iterations << ",\n";
    out << "  \"kernel_cutoff\": " << args.kernel_cutoff << ",\n";
    out << "  \"local_rows_by_rank\": ";
    write_int_array(out, result.local_rows_by_rank);
    out << ",\n";
    out << "  \"max_iters\": " << args.max_iters << ",\n";
    out << "  \"num_processes\": " << num_processes << ",\n";
    out << "  \"quantization_scale\": " << args.quantization_scale << ",\n";
    out << "  \"rank_comm_sec\": ";
    write_double_array(out, result.rank_comm_sec);
    out << ",\n";
    out << "  \"rank_compute_sec\": ";
    write_double_array(out, result.rank_compute_sec);
    out << ",\n";
    out << "  \"rank_total_sec\": ";
    write_double_array(out, result.rank_total_sec);
    out << ",\n";
    out << "  \"row_error\": " << result.row_error << ",\n";
    out << "  \"rows\": " << args.rows << ",\n";
    out << "  \"runtime_sec\": " << result.runtime_sec << ",\n";
    out << "  \"seed\": " << args.seed << ",\n";
    out << "  \"sync_every\": " << args.sync_every << ",\n";
    out << "  \"tol\": " << args.tol << ",\n";
    out << "  \"transport_objective\": " << result.objective << "\n";
    out << "}\n";
}

Result run_sequential(const Args& args) {
    std::vector<double> cost = build_cost_block(0, args.rows, args.rows, args.cols, args.cost_mode, args.seed);

    const double started = now_seconds();
    const double global_min = min_value(cost);
    std::vector<double> kernel = build_kernel(cost, global_min, args.epsilon, args.kernel_cutoff);
    std::vector<double> u(args.rows, 1.0);
    std::vector<double> v(args.cols, 1.0);
    std::vector<double> col_acc(args.cols, 0.0);

    const double a_value = 1.0 / static_cast<double>(args.rows);
    const double b_value = 1.0 / static_cast<double>(args.cols);
    Result result;

    for (int iteration = 1; iteration <= args.max_iters; ++iteration) {
        for (int i = 0; i < args.rows; ++i) {
            const std::size_t base = static_cast<std::size_t>(i) * args.cols;
            double sum = 0.0;
            for (int j = 0; j < args.cols; ++j) {
                const double kij = kernel[base + j];
                if (kij != 0.0) {
                    sum += kij * v[j];
                }
            }
            u[i] = a_value / (sum + kTiny);
        }

        std::fill(col_acc.begin(), col_acc.end(), 0.0);
        for (int i = 0; i < args.rows; ++i) {
            const std::size_t base = static_cast<std::size_t>(i) * args.cols;
            const double ui = u[i];
            for (int j = 0; j < args.cols; ++j) {
                const double kij = kernel[base + j];
                if (kij != 0.0) {
                    col_acc[j] += kij * ui;
                }
            }
        }
        for (int j = 0; j < args.cols; ++j) {
            v[j] = b_value / (col_acc[j] + kTiny);
        }

        result.iterations = iteration;
        if (iteration % args.check_every == 0 || iteration == args.max_iters) {
            double row_error = 0.0;
            std::fill(col_acc.begin(), col_acc.end(), 0.0);
            for (int i = 0; i < args.rows; ++i) {
                const std::size_t base = static_cast<std::size_t>(i) * args.cols;
                const double ui = u[i];
                double row_mass = 0.0;
                for (int j = 0; j < args.cols; ++j) {
                    const double kij = kernel[base + j];
                    if (kij != 0.0) {
                        const double mass = ui * kij * v[j];
                        row_mass += mass;
                        col_acc[j] += mass;
                    }
                }
                row_error += std::abs(row_mass - a_value);
            }
            double col_error = 0.0;
            for (int j = 0; j < args.cols; ++j) {
                col_error += std::abs(col_acc[j] - b_value);
            }
            result.row_error = row_error;
            result.col_error = col_error;
            if (row_error < args.tol && col_error < args.tol) {
                break;
            }
        }
    }

    double row_error = 0.0;
    double objective = 0.0;
    std::fill(col_acc.begin(), col_acc.end(), 0.0);
    for (int i = 0; i < args.rows; ++i) {
        const std::size_t base = static_cast<std::size_t>(i) * args.cols;
        const double ui = u[i];
        double row_mass = 0.0;
        for (int j = 0; j < args.cols; ++j) {
            const double kij = kernel[base + j];
            if (kij != 0.0) {
                const double mass = ui * kij * v[j];
                row_mass += mass;
                col_acc[j] += mass;
                objective += mass * cost[base + j];
            }
        }
        row_error += std::abs(row_mass - a_value);
    }
    double col_error = 0.0;
    for (int j = 0; j < args.cols; ++j) {
        col_error += std::abs(col_acc[j] - b_value);
    }
    result.row_error = row_error;
    result.col_error = col_error;
    result.objective = objective;
    result.runtime_sec = now_seconds() - started;
    result.local_rows = args.rows;
    result.local_compute_sec = result.runtime_sec;
    result.local_comm_sec = 0.0;
    result.local_rows_by_rank = {args.rows};
    result.rank_compute_sec = {result.runtime_sec};
    result.rank_comm_sec = {0.0};
    result.rank_total_sec = {result.runtime_sec};
    return result;
}

std::vector<std::string> gather_hostnames(MPI_Comm comm, int rank, int size) {
    std::string local = hostname();
    local.resize(256, '\0');
    std::vector<char> gathered;
    if (rank == 0) {
        gathered.resize(static_cast<std::size_t>(size) * 256);
    }
    MPI_Gather(local.data(), 256, MPI_CHAR, gathered.data(), 256, MPI_CHAR, 0, comm);

    std::vector<std::string> names;
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            const char* start = gathered.data() + static_cast<std::size_t>(i) * 256;
            names.emplace_back(start);
        }
    }
    return names;
}

void allreduce_columns(
    const Args& args,
    const std::vector<double>& local_col,
    std::vector<double>& global_col,
    MPI_Comm comm,
    Result& result
) {
    const double comm_started = now_seconds();
    if (args.comm_mode == "float32") {
        std::vector<float> send(local_col.size(), 0.0F);
        std::vector<float> recv(local_col.size(), 0.0F);
        for (std::size_t j = 0; j < local_col.size(); ++j) {
            send[j] = static_cast<float>(local_col[j]);
        }
        MPI_Allreduce(send.data(), recv.data(), static_cast<int>(recv.size()), MPI_FLOAT, MPI_SUM, comm);
        for (std::size_t j = 0; j < recv.size(); ++j) {
            global_col[j] = static_cast<double>(recv[j]);
        }
        result.column_payload_bytes += static_cast<long long>(local_col.size() * sizeof(float));
    } else if (args.comm_mode == "quantized") {
        std::vector<std::uint32_t> send(local_col.size(), 0);
        std::vector<std::uint32_t> recv(local_col.size(), 0);
        const double max_quantized = static_cast<double>(std::numeric_limits<std::uint32_t>::max());
        for (std::size_t j = 0; j < local_col.size(); ++j) {
            const double scaled = std::round(std::max(0.0, local_col[j]) * args.quantization_scale);
            send[j] = static_cast<std::uint32_t>(std::min(scaled, max_quantized));
        }
        MPI_Allreduce(send.data(), recv.data(), static_cast<int>(recv.size()), MPI_UINT32_T, MPI_SUM, comm);
        for (std::size_t j = 0; j < recv.size(); ++j) {
            global_col[j] = static_cast<double>(recv[j]) / args.quantization_scale;
        }
        result.column_payload_bytes += static_cast<long long>(local_col.size() * sizeof(std::uint32_t));
    } else {
        MPI_Allreduce(local_col.data(), global_col.data(), static_cast<int>(global_col.size()), MPI_DOUBLE, MPI_SUM, comm);
        result.column_payload_bytes += static_cast<long long>(local_col.size() * sizeof(double));
    }
    result.local_comm_sec += now_seconds() - comm_started;
    result.column_syncs += 1;
}

void gather_rank_metrics(Result& result, int local_rows, MPI_Comm comm, int rank, int size) {
    const double local_total_sec = result.runtime_sec;
    if (rank == 0) {
        result.local_rows_by_rank.resize(size);
        result.rank_compute_sec.resize(size);
        result.rank_comm_sec.resize(size);
        result.rank_total_sec.resize(size);
    }
    MPI_Gather(
        &local_rows,
        1,
        MPI_INT,
        rank == 0 ? result.local_rows_by_rank.data() : nullptr,
        1,
        MPI_INT,
        0,
        comm
    );
    MPI_Gather(
        &result.local_compute_sec,
        1,
        MPI_DOUBLE,
        rank == 0 ? result.rank_compute_sec.data() : nullptr,
        1,
        MPI_DOUBLE,
        0,
        comm
    );
    MPI_Gather(
        &result.local_comm_sec,
        1,
        MPI_DOUBLE,
        rank == 0 ? result.rank_comm_sec.data() : nullptr,
        1,
        MPI_DOUBLE,
        0,
        comm
    );
    MPI_Gather(
        &local_total_sec,
        1,
        MPI_DOUBLE,
        rank == 0 ? result.rank_total_sec.data() : nullptr,
        1,
        MPI_DOUBLE,
        0,
        comm
    );

    double max_runtime = 0.0;
    MPI_Reduce(&local_total_sec, &max_runtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        result.runtime_sec = max_runtime;
    }
}

Result run_mpi(const Args& args, MPI_Comm comm, int rank, int size) {
    const auto [start, end] = split_range(args.rows, size, rank);
    const int local_rows = end - start;
    std::vector<double> local_cost = build_cost_block(start, local_rows, args.rows, args.cols, args.cost_mode, args.seed);

    const double local_min = min_value(local_cost);
    double global_min = 0.0;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);

    const double started = now_seconds();
    Result result;
    const double kernel_started = now_seconds();
    std::vector<double> local_kernel = build_kernel(local_cost, global_min, args.epsilon, args.kernel_cutoff);
    result.local_compute_sec += now_seconds() - kernel_started;
    std::vector<double> local_u(local_rows, 1.0);
    std::vector<double> v(args.cols, 1.0);
    std::vector<double> local_col(args.cols, 0.0);
    std::vector<double> global_col(args.cols, 0.0);

    const double a_value = 1.0 / static_cast<double>(args.rows);
    const double b_value = 1.0 / static_cast<double>(args.cols);
    result.local_rows = local_rows;

    for (int iteration = 1; iteration <= args.max_iters; ++iteration) {
        double compute_started = now_seconds();
        for (int i = 0; i < local_rows; ++i) {
            const std::size_t base = static_cast<std::size_t>(i) * args.cols;
            double sum = 0.0;
            for (int j = 0; j < args.cols; ++j) {
                const double kij = local_kernel[base + j];
                if (kij != 0.0) {
                    sum += kij * v[j];
                }
            }
            local_u[i] = a_value / (sum + kTiny);
        }

        std::fill(local_col.begin(), local_col.end(), 0.0);
        for (int i = 0; i < local_rows; ++i) {
            const std::size_t base = static_cast<std::size_t>(i) * args.cols;
            const double ui = local_u[i];
            for (int j = 0; j < args.cols; ++j) {
                const double kij = local_kernel[base + j];
                if (kij != 0.0) {
                    local_col[j] += kij * ui;
                }
            }
        }
        result.local_compute_sec += now_seconds() - compute_started;
        const bool should_sync =
            args.comm_mode != "lazy" || iteration % args.sync_every == 0 || iteration == args.max_iters;
        if (should_sync) {
            allreduce_columns(args, local_col, global_col, comm, result);
            for (int j = 0; j < args.cols; ++j) {
                v[j] = b_value / (global_col[j] + kTiny);
            }
        }

        result.iterations = iteration;
        if (iteration % args.check_every == 0 || iteration == args.max_iters) {
            double local_row_error = 0.0;
            std::fill(local_col.begin(), local_col.end(), 0.0);
            compute_started = now_seconds();
            for (int i = 0; i < local_rows; ++i) {
                const std::size_t base = static_cast<std::size_t>(i) * args.cols;
                const double ui = local_u[i];
                double row_mass = 0.0;
                for (int j = 0; j < args.cols; ++j) {
                    const double kij = local_kernel[base + j];
                    if (kij != 0.0) {
                        const double mass = ui * kij * v[j];
                        row_mass += mass;
                        local_col[j] += mass;
                    }
                }
                local_row_error += std::abs(row_mass - a_value);
            }
            result.local_compute_sec += now_seconds() - compute_started;
            double row_error = 0.0;
            double comm_started = now_seconds();
            MPI_Allreduce(&local_row_error, &row_error, 1, MPI_DOUBLE, MPI_SUM, comm);
            MPI_Allreduce(local_col.data(), global_col.data(), args.cols, MPI_DOUBLE, MPI_SUM, comm);
            result.local_comm_sec += now_seconds() - comm_started;
            double col_error = 0.0;
            for (int j = 0; j < args.cols; ++j) {
                col_error += std::abs(global_col[j] - b_value);
            }
            result.row_error = row_error;
            result.col_error = col_error;
            if (row_error < args.tol && col_error < args.tol) {
                break;
            }
        }
    }

    double local_row_error = 0.0;
    double local_objective = 0.0;
    std::fill(local_col.begin(), local_col.end(), 0.0);
    double compute_started = now_seconds();
    for (int i = 0; i < local_rows; ++i) {
        const std::size_t base = static_cast<std::size_t>(i) * args.cols;
        const double ui = local_u[i];
        double row_mass = 0.0;
        for (int j = 0; j < args.cols; ++j) {
            const double kij = local_kernel[base + j];
            if (kij != 0.0) {
                const double mass = ui * kij * v[j];
                row_mass += mass;
                local_col[j] += mass;
                local_objective += mass * local_cost[base + j];
            }
        }
        local_row_error += std::abs(row_mass - a_value);
    }
    result.local_compute_sec += now_seconds() - compute_started;
    double row_error = 0.0;
    double objective = 0.0;
    double comm_started = now_seconds();
    MPI_Allreduce(&local_row_error, &row_error, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&local_objective, &objective, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(local_col.data(), global_col.data(), args.cols, MPI_DOUBLE, MPI_SUM, comm);
    result.local_comm_sec += now_seconds() - comm_started;

    double col_error = 0.0;
    for (int j = 0; j < args.cols; ++j) {
        col_error += std::abs(global_col[j] - b_value);
    }
    result.row_error = row_error;
    result.col_error = col_error;
    result.objective = objective;
    result.runtime_sec = now_seconds() - started;
    gather_rank_metrics(result, local_rows, comm, rank, size);
    return result;
}

void usage(const char* program) {
    std::cerr
        << "Usage: " << program << " --mode sequential|mpi --rows N --cols N --output FILE [options]\n"
        << "Options: --cost-mode random|squared_distance_1d|block --seed N --epsilon X\n"
        << "         --max-iters N --tol X --check-every N\n"
        << "         --comm-mode double|float32|quantized|lazy --sync-every N\n"
        << "         --kernel-cutoff X --quantization-scale X\n";
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + name);
            }
            return argv[++i];
        };
        if (key == "--mode") {
            args.mode = need_value(key);
        } else if (key == "--rows") {
            args.rows = std::stoi(need_value(key));
        } else if (key == "--cols") {
            args.cols = std::stoi(need_value(key));
        } else if (key == "--cost-mode") {
            args.cost_mode = need_value(key);
        } else if (key == "--comm-mode") {
            args.comm_mode = need_value(key);
        } else if (key == "--seed") {
            args.seed = static_cast<std::uint64_t>(std::stoull(need_value(key)));
        } else if (key == "--epsilon") {
            args.epsilon = std::stod(need_value(key));
        } else if (key == "--max-iters") {
            args.max_iters = std::stoi(need_value(key));
        } else if (key == "--tol") {
            args.tol = std::stod(need_value(key));
        } else if (key == "--check-every") {
            args.check_every = std::stoi(need_value(key));
        } else if (key == "--sync-every") {
            args.sync_every = std::stoi(need_value(key));
        } else if (key == "--kernel-cutoff") {
            args.kernel_cutoff = std::stod(need_value(key));
        } else if (key == "--quantization-scale") {
            args.quantization_scale = std::stod(need_value(key));
        } else if (key == "--output") {
            args.output = need_value(key);
        } else if (key == "--help" || key == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }

    if (args.mode != "sequential" && args.mode != "mpi") {
        throw std::runtime_error("--mode must be sequential or mpi");
    }
    if (
        args.comm_mode != "double" &&
        args.comm_mode != "float32" &&
        args.comm_mode != "quantized" &&
        args.comm_mode != "lazy"
    ) {
        throw std::runtime_error("--comm-mode must be double, float32, quantized, or lazy");
    }
    if (args.rows <= 0 || args.cols <= 0) {
        throw std::runtime_error("--rows and --cols must be positive");
    }
    if (args.epsilon <= 0.0) {
        throw std::runtime_error("--epsilon must be positive");
    }
    if (args.max_iters <= 0 || args.check_every <= 0) {
        throw std::runtime_error("--max-iters and --check-every must be positive");
    }
    if (args.sync_every <= 0) {
        throw std::runtime_error("--sync-every must be positive");
    }
    if (args.tol < 0.0) {
        throw std::runtime_error("--tol must be non-negative");
    }
    if (args.kernel_cutoff < 0.0) {
        throw std::runtime_error("--kernel-cutoff must be non-negative");
    }
    if (args.quantization_scale <= 0.0) {
        throw std::runtime_error("--quantization-scale must be positive");
    }
    if (args.output.empty()) {
        throw std::runtime_error("--output is required");
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        const std::string sparse_suffix = args.kernel_cutoff > 0.0 ? "_sparse_cutoff" : "";

        if (args.mode == "sequential") {
            Result result = run_sequential(args);
            write_json(args, result, 1, {hostname()}, "sinkhorn_cpp_dense" + sparse_suffix);
            std::cout << "C++ Dense Sinkhorn complete: runtime=" << std::fixed << std::setprecision(6)
                      << result.runtime_sec << "s, iterations=" << result.iterations
                      << ", row_error=" << std::scientific << result.row_error
                      << ", col_error=" << result.col_error
                      << ", objective=" << std::fixed << std::setprecision(12) << result.objective
                      << ", kernel_cutoff=" << args.kernel_cutoff
                      << std::endl;
            return 0;
        }

        MPI_Init(&argc, &argv);
        int rank = 0;
        int size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        Result result = run_mpi(args, MPI_COMM_WORLD, rank, size);
        std::vector<std::string> hostnames = gather_hostnames(MPI_COMM_WORLD, rank, size);
        if (rank == 0) {
            write_json(args, result, size, hostnames, "sinkhorn_cpp_mpi_" + args.comm_mode + sparse_suffix);
            std::cout << "C++ MPI Sinkhorn complete: runtime=" << std::fixed << std::setprecision(6)
                      << result.runtime_sec << "s, iterations=" << result.iterations
                      << ", row_error=" << std::scientific << result.row_error
                      << ", col_error=" << result.col_error
                      << ", objective=" << std::fixed << std::setprecision(12) << result.objective
                      << ", processes=" << size
                      << ", comm_mode=" << args.comm_mode
                      << ", column_syncs=" << result.column_syncs
                      << ", column_payload_bytes=" << result.column_payload_bytes
                      << ", kernel_cutoff=" << args.kernel_cutoff
                      << std::endl;
        }
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << std::endl;
        return 1;
    }
}

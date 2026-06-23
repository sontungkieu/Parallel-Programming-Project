#include <mpi.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
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
    std::string partition_mode = "contiguous";
    int chunk_rows = 256;
    std::vector<double> rank_weights;
    std::string output;
};

struct Result {
    int iterations = 0;
    int column_syncs = 0;
    long long column_payload_bytes = 0;
    double runtime_sec = 0.0;
    double queue_scheduler_sec = 0.0;
    double row_error = std::numeric_limits<double>::infinity();
    double col_error = std::numeric_limits<double>::infinity();
    double objective = std::numeric_limits<double>::infinity();
    std::vector<double> effective_rank_weights;
    std::vector<long long> local_rows_by_rank;
    std::vector<int> local_chunks_by_rank;
    std::vector<double> rank_compute_sec;
    std::vector<double> rank_rows_per_sec;
    std::vector<double> suggested_rank_weights;
    std::vector<int> queue_chunks_by_rank;
};

struct RowBlock {
    int start = 0;
    int end = 0;
};

struct PartitionPlan {
    std::vector<RowBlock> local_blocks;
    std::vector<long long> rows_by_rank;
    std::vector<int> chunks_by_rank;
};

struct QueuePhaseStats {
    long long rows = 0;
    int chunks = 0;
    double compute_sec = 0.0;
    double scheduler_sec = 0.0;
    double row_error = 0.0;
    double objective = 0.0;
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

int block_rows(const RowBlock& block) {
    return block.end - block.start;
}

std::vector<RowBlock> build_row_chunks(int rows, int chunk_rows) {
    std::vector<RowBlock> chunks;
    for (int start = 0; start < rows; start += chunk_rows) {
        chunks.push_back({start, std::min(start + chunk_rows, rows)});
    }
    return chunks;
}

std::string strip_spaces(std::string value) {
    value.erase(
        std::remove_if(
            value.begin(),
            value.end(),
            [](unsigned char ch) { return std::isspace(ch) != 0; }
        ),
        value.end()
    );
    return value;
}

std::vector<double> parse_double_csv(const std::string& csv) {
    std::vector<double> values;
    std::stringstream input(csv);
    std::string token;
    while (std::getline(input, token, ',')) {
        token = strip_spaces(token);
        if (token.empty()) {
            throw std::runtime_error("--rank-weights contains an empty value");
        }
        std::size_t parsed = 0;
        const double value = std::stod(token, &parsed);
        if (parsed != token.size() || !std::isfinite(value) || value <= 0.0) {
            throw std::runtime_error("--rank-weights values must be positive finite numbers");
        }
        values.push_back(value);
    }
    if (values.empty()) {
        throw std::runtime_error("--rank-weights cannot be empty");
    }
    return values;
}

std::vector<double> resolve_rank_weights(const Args& args, int size) {
    if (args.rank_weights.empty()) {
        return std::vector<double>(size, 1.0);
    }
    if (static_cast<int>(args.rank_weights.size()) != size) {
        throw std::runtime_error("--rank-weights length must match the MPI process count");
    }
    return args.rank_weights;
}

std::string double_csv(const std::vector<double>& values) {
    std::ostringstream out;
    out << std::setprecision(10);
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << values[i];
    }
    return out.str();
}

std::vector<double> suggest_rank_weights(const std::vector<double>& rows_per_sec) {
    if (rows_per_sec.empty()) {
        return {};
    }

    double positive_sum = 0.0;
    int positive_count = 0;
    for (double value : rows_per_sec) {
        if (value > 0.0 && std::isfinite(value)) {
            positive_sum += value;
            positive_count += 1;
        }
    }
    if (positive_count == 0) {
        return std::vector<double>(rows_per_sec.size(), 1.0);
    }

    const double mean = positive_sum / static_cast<double>(positive_count);
    std::vector<double> suggested(rows_per_sec.size(), 1.0);
    for (std::size_t i = 0; i < rows_per_sec.size(); ++i) {
        suggested[i] = rows_per_sec[i] > 0.0 && std::isfinite(rows_per_sec[i])
            ? std::max(0.05, rows_per_sec[i] / mean)
            : 0.05;
    }

    const double suggested_mean =
        std::accumulate(suggested.begin(), suggested.end(), 0.0) / static_cast<double>(suggested.size());
    for (double& value : suggested) {
        value /= suggested_mean;
    }
    return suggested;
}

PartitionPlan build_partition_plan(
    const Args& args,
    int rank,
    int size,
    const std::vector<double>& weights
) {
    PartitionPlan plan;
    plan.rows_by_rank.assign(size, 0);
    plan.chunks_by_rank.assign(size, 0);

    if (args.partition_mode == "contiguous") {
        for (int r = 0; r < size; ++r) {
            const auto [start, end] = split_range(args.rows, size, r);
            const int width = end - start;
            plan.rows_by_rank[r] = width;
            plan.chunks_by_rank[r] = width > 0 ? 1 : 0;
            if (r == rank && width > 0) {
                plan.local_blocks.push_back({start, end});
            }
        }
        return plan;
    }

    if (args.partition_mode != "weighted-chunk") {
        throw std::runtime_error("static partition builder only supports contiguous or weighted-chunk");
    }

    const std::vector<RowBlock> chunks = build_row_chunks(args.rows, args.chunk_rows);
    std::vector<double> assigned_score(size, 0.0);
    for (const RowBlock& chunk : chunks) {
        int best_rank = 0;
        double best_score = assigned_score[0] / weights[0];
        for (int r = 1; r < size; ++r) {
            const double score = assigned_score[r] / weights[r];
            if (score < best_score - 1e-12) {
                best_score = score;
                best_rank = r;
            }
        }

        const int rows = block_rows(chunk);
        assigned_score[best_rank] += static_cast<double>(rows);
        plan.rows_by_rank[best_rank] += rows;
        plan.chunks_by_rank[best_rank] += 1;
        if (best_rank == rank) {
            plan.local_blocks.push_back(chunk);
        }
    }
    return plan;
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

std::vector<double> build_cost_blocks(
    const std::vector<RowBlock>& blocks,
    int rows,
    int cols,
    const std::string& mode,
    std::uint64_t seed
) {
    long long local_rows = 0;
    for (const RowBlock& block : blocks) {
        local_rows += block_rows(block);
    }

    std::vector<double> cost(static_cast<std::size_t>(local_rows) * cols);
    std::size_t local_index = 0;
    for (const RowBlock& block : blocks) {
        for (int row = block.start; row < block.end; ++row) {
            for (int j = 0; j < cols; ++j) {
                cost[local_index * static_cast<std::size_t>(cols) + j] =
                    cost_value(row, j, rows, cols, mode, seed);
            }
            local_index += 1;
        }
    }
    return cost;
}

double min_cost_range(
    int start,
    int end,
    int rows,
    int cols,
    const std::string& mode,
    std::uint64_t seed
) {
    double minimum = std::numeric_limits<double>::infinity();
    for (int row = start; row < end; ++row) {
        for (int col = 0; col < cols; ++col) {
            minimum = std::min(minimum, cost_value(row, col, rows, cols, mode, seed));
        }
    }
    return minimum;
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

void write_double_array(std::ostream& out, const std::vector<double>& values) {
    out << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << values[i];
    }
    out << "]";
}

void write_int_array(std::ostream& out, const std::vector<int>& values) {
    out << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << values[i];
    }
    out << "]";
}

void write_long_long_array(std::ostream& out, const std::vector<long long>& values) {
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
    out << "  \"max_iters\": " << args.max_iters << ",\n";
    out << "  \"num_processes\": " << num_processes << ",\n";
    out << "  \"partition_mode\": \"" << json_escape(args.partition_mode) << "\",\n";
    out << "  \"chunk_rows\": " << args.chunk_rows << ",\n";
    out << "  \"rank_weights\": ";
    write_double_array(out, result.effective_rank_weights);
    out << ",\n";
    out << "  \"local_rows_by_rank\": ";
    write_long_long_array(out, result.local_rows_by_rank);
    out << ",\n";
    out << "  \"local_chunks_by_rank\": ";
    write_int_array(out, result.local_chunks_by_rank);
    out << ",\n";
    out << "  \"rank_compute_sec\": ";
    write_double_array(out, result.rank_compute_sec);
    out << ",\n";
    out << "  \"rank_rows_per_sec\": ";
    write_double_array(out, result.rank_rows_per_sec);
    out << ",\n";
    out << "  \"suggested_rank_weights\": ";
    write_double_array(out, result.suggested_rank_weights);
    out << ",\n";
    out << "  \"suggested_rank_weights_csv\": \"" << json_escape(double_csv(result.suggested_rank_weights)) << "\",\n";
    out << "  \"queue_chunks_by_rank\": ";
    write_int_array(out, result.queue_chunks_by_rank);
    out << ",\n";
    out << "  \"queue_scheduler_sec\": " << result.queue_scheduler_sec << ",\n";
    out << "  \"quantization_scale\": " << args.quantization_scale << ",\n";
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
    result.column_syncs += 1;
}

void gather_rank_metrics(
    Result& result,
    const std::vector<double>& weights,
    long long rows_field,
    int chunks_field,
    double local_compute_sec,
    long long processed_rows,
    int queue_chunks,
    double scheduler_sec,
    MPI_Comm comm,
    int rank,
    int size
) {
    const double local_rows_per_sec =
        local_compute_sec > 0.0 ? static_cast<double>(processed_rows) / local_compute_sec : 0.0;

    if (rank == 0) {
        result.effective_rank_weights = weights;
        result.local_rows_by_rank.resize(size);
        result.local_chunks_by_rank.resize(size);
        result.rank_compute_sec.resize(size);
        result.rank_rows_per_sec.resize(size);
        result.queue_chunks_by_rank.resize(size);
    }

    MPI_Gather(&rows_field, 1, MPI_LONG_LONG, result.local_rows_by_rank.data(), 1, MPI_LONG_LONG, 0, comm);
    MPI_Gather(&chunks_field, 1, MPI_INT, result.local_chunks_by_rank.data(), 1, MPI_INT, 0, comm);
    MPI_Gather(&local_compute_sec, 1, MPI_DOUBLE, result.rank_compute_sec.data(), 1, MPI_DOUBLE, 0, comm);
    MPI_Gather(&local_rows_per_sec, 1, MPI_DOUBLE, result.rank_rows_per_sec.data(), 1, MPI_DOUBLE, 0, comm);
    MPI_Gather(&queue_chunks, 1, MPI_INT, result.queue_chunks_by_rank.data(), 1, MPI_INT, 0, comm);

    double max_scheduler_sec = 0.0;
    MPI_Reduce(&scheduler_sec, &max_scheduler_sec, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        result.queue_scheduler_sec = max_scheduler_sec;
        result.suggested_rank_weights = suggest_rank_weights(result.rank_rows_per_sec);
    }
}

Result run_mpi_static(
    const Args& args,
    MPI_Comm comm,
    int rank,
    int size,
    const std::vector<double>& weights
) {
    const PartitionPlan partition = build_partition_plan(args, rank, size, weights);
    const int local_rows = static_cast<int>(partition.rows_by_rank[rank]);
    const int local_chunks = partition.chunks_by_rank[rank];
    std::vector<double> local_cost =
        build_cost_blocks(partition.local_blocks, args.rows, args.cols, args.cost_mode, args.seed);

    const double local_min = min_value(local_cost);
    double global_min = 0.0;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);

    const double started = now_seconds();
    double local_compute_sec = 0.0;
    double compute_started = now_seconds();
    std::vector<double> local_kernel = build_kernel(local_cost, global_min, args.epsilon, args.kernel_cutoff);
    local_compute_sec += now_seconds() - compute_started;
    std::vector<double> local_u(local_rows, 1.0);
    std::vector<double> v(args.cols, 1.0);
    std::vector<double> local_col(args.cols, 0.0);
    std::vector<double> global_col(args.cols, 0.0);

    const double a_value = 1.0 / static_cast<double>(args.rows);
    const double b_value = 1.0 / static_cast<double>(args.cols);
    Result result;

    for (int iteration = 1; iteration <= args.max_iters; ++iteration) {
        compute_started = now_seconds();
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
        local_compute_sec += now_seconds() - compute_started;

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
            compute_started = now_seconds();
            double local_row_error = 0.0;
            std::fill(local_col.begin(), local_col.end(), 0.0);
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
            local_compute_sec += now_seconds() - compute_started;

            double row_error = 0.0;
            MPI_Allreduce(&local_row_error, &row_error, 1, MPI_DOUBLE, MPI_SUM, comm);
            MPI_Allreduce(local_col.data(), global_col.data(), args.cols, MPI_DOUBLE, MPI_SUM, comm);
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
                local_objective += mass * local_cost[base + j];
            }
        }
        local_row_error += std::abs(row_mass - a_value);
    }
    local_compute_sec += now_seconds() - compute_started;

    double row_error = 0.0;
    double objective = 0.0;
    MPI_Allreduce(&local_row_error, &row_error, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&local_objective, &objective, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(local_col.data(), global_col.data(), args.cols, MPI_DOUBLE, MPI_SUM, comm);

    double col_error = 0.0;
    for (int j = 0; j < args.cols; ++j) {
        col_error += std::abs(global_col[j] - b_value);
    }
    result.row_error = row_error;
    result.col_error = col_error;
    result.objective = objective;
    result.runtime_sec = now_seconds() - started;

    const long long processed_rows =
        static_cast<long long>(local_rows) * static_cast<long long>(std::max(1, result.iterations));
    gather_rank_metrics(
        result,
        weights,
        local_rows,
        local_chunks,
        local_compute_sec,
        processed_rows,
        0,
        0.0,
        comm,
        rank,
        size
    );
    return result;
}

double kernel_value_for_row(
    const Args& args,
    int row,
    int col,
    double global_min
) {
    const double cost = cost_value(row, col, args.rows, args.cols, args.cost_mode, args.seed);
    const double value = std::exp(-(cost - global_min) / args.epsilon);
    return (args.kernel_cutoff > 0.0 && value < args.kernel_cutoff) ? 0.0 : value;
}

void accumulate_chunk_columns(
    const Args& args,
    const RowBlock& chunk,
    const std::vector<double>& v,
    double global_min,
    std::vector<double>& local_col,
    std::vector<double>& row_kernel
) {
    const double a_value = 1.0 / static_cast<double>(args.rows);
    for (int row = chunk.start; row < chunk.end; ++row) {
        double sum = 0.0;
        for (int j = 0; j < args.cols; ++j) {
            const double kij = kernel_value_for_row(args, row, j, global_min);
            row_kernel[j] = kij;
            if (kij != 0.0) {
                sum += kij * v[j];
            }
        }
        const double ui = a_value / (sum + kTiny);
        for (int j = 0; j < args.cols; ++j) {
            const double kij = row_kernel[j];
            if (kij != 0.0) {
                local_col[j] += kij * ui;
            }
        }
    }
}

void accumulate_chunk_final(
    const Args& args,
    const RowBlock& chunk,
    const std::vector<double>& v,
    double global_min,
    std::vector<double>& local_col,
    std::vector<double>& row_kernel,
    double& local_row_error,
    double& local_objective
) {
    const double a_value = 1.0 / static_cast<double>(args.rows);
    for (int row = chunk.start; row < chunk.end; ++row) {
        double sum = 0.0;
        for (int j = 0; j < args.cols; ++j) {
            const double kij = kernel_value_for_row(args, row, j, global_min);
            row_kernel[j] = kij;
            if (kij != 0.0) {
                sum += kij * v[j];
            }
        }

        const double ui = a_value / (sum + kTiny);
        double row_mass = 0.0;
        for (int j = 0; j < args.cols; ++j) {
            const double kij = row_kernel[j];
            if (kij != 0.0) {
                const double mass = ui * kij * v[j];
                row_mass += mass;
                local_col[j] += mass;
                local_objective += mass * cost_value(row, j, args.rows, args.cols, args.cost_mode, args.seed);
            }
        }
        local_row_error += std::abs(row_mass - a_value);
    }
}

void reset_queue_counter(int& counter, MPI_Win window, int rank, MPI_Comm comm, double& scheduler_sec) {
    const double started = now_seconds();
    (void)counter;
    MPI_Barrier(comm);
    if (rank == 0) {
        int zero = 0;
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, window);
        MPI_Put(&zero, 1, MPI_INT, 0, 0, 1, MPI_INT, window);
        MPI_Win_flush(0, window);
        MPI_Win_unlock(0, window);
    }
    MPI_Barrier(comm);
    scheduler_sec += now_seconds() - started;
}

int fetch_queue_chunk(MPI_Win window, double& scheduler_sec) {
    int one = 1;
    int previous = 0;
    const double started = now_seconds();
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, window);
    MPI_Fetch_and_op(&one, &previous, MPI_INT, 0, 0, MPI_SUM, window);
    MPI_Win_flush(0, window);
    MPI_Win_unlock(0, window);
    scheduler_sec += now_seconds() - started;
    return previous;
}

QueuePhaseStats run_queue_column_phase(
    const Args& args,
    const std::vector<RowBlock>& chunks,
    const std::vector<double>& v,
    double global_min,
    std::vector<double>& local_col,
    int& counter,
    MPI_Win window,
    int rank,
    MPI_Comm comm
) {
    QueuePhaseStats stats;
    reset_queue_counter(counter, window, rank, comm, stats.scheduler_sec);

    std::vector<double> row_kernel(args.cols, 0.0);
    while (true) {
        const int chunk_id = fetch_queue_chunk(window, stats.scheduler_sec);
        if (chunk_id >= static_cast<int>(chunks.size())) {
            break;
        }
        const RowBlock& chunk = chunks[chunk_id];
        const double compute_started = now_seconds();
        accumulate_chunk_columns(args, chunk, v, global_min, local_col, row_kernel);
        stats.compute_sec += now_seconds() - compute_started;
        stats.rows += block_rows(chunk);
        stats.chunks += 1;
    }
    return stats;
}

QueuePhaseStats run_queue_final_phase(
    const Args& args,
    const std::vector<RowBlock>& chunks,
    const std::vector<double>& v,
    double global_min,
    std::vector<double>& local_col,
    int& counter,
    MPI_Win window,
    int rank,
    MPI_Comm comm
) {
    QueuePhaseStats stats;
    reset_queue_counter(counter, window, rank, comm, stats.scheduler_sec);

    std::vector<double> row_kernel(args.cols, 0.0);
    while (true) {
        const int chunk_id = fetch_queue_chunk(window, stats.scheduler_sec);
        if (chunk_id >= static_cast<int>(chunks.size())) {
            break;
        }
        const RowBlock& chunk = chunks[chunk_id];
        const double compute_started = now_seconds();
        accumulate_chunk_final(
            args,
            chunk,
            v,
            global_min,
            local_col,
            row_kernel,
            stats.row_error,
            stats.objective
        );
        stats.compute_sec += now_seconds() - compute_started;
        stats.rows += block_rows(chunk);
        stats.chunks += 1;
    }
    return stats;
}

QueuePhaseStats run_local_column_phase(
    const Args& args,
    const std::vector<RowBlock>& chunks,
    const std::vector<double>& v,
    double global_min,
    std::vector<double>& local_col
) {
    QueuePhaseStats stats;
    std::vector<double> row_kernel(args.cols, 0.0);
    for (const RowBlock& chunk : chunks) {
        const double compute_started = now_seconds();
        accumulate_chunk_columns(args, chunk, v, global_min, local_col, row_kernel);
        stats.compute_sec += now_seconds() - compute_started;
        stats.rows += block_rows(chunk);
        stats.chunks += 1;
    }
    return stats;
}

QueuePhaseStats run_local_final_phase(
    const Args& args,
    const std::vector<RowBlock>& chunks,
    const std::vector<double>& v,
    double global_min,
    std::vector<double>& local_col
) {
    QueuePhaseStats stats;
    std::vector<double> row_kernel(args.cols, 0.0);
    for (const RowBlock& chunk : chunks) {
        const double compute_started = now_seconds();
        accumulate_chunk_final(
            args,
            chunk,
            v,
            global_min,
            local_col,
            row_kernel,
            stats.row_error,
            stats.objective
        );
        stats.compute_sec += now_seconds() - compute_started;
        stats.rows += block_rows(chunk);
        stats.chunks += 1;
    }
    return stats;
}

Result run_mpi_runtime_queue(
    const Args& args,
    MPI_Comm comm,
    int rank,
    int size,
    const std::vector<double>& weights
) {
    const std::vector<RowBlock> chunks = build_row_chunks(args.rows, args.chunk_rows);
    const auto [min_start, min_end] = split_range(args.rows, size, rank);
    const double local_min = min_cost_range(min_start, min_end, args.rows, args.cols, args.cost_mode, args.seed);
    double global_min = 0.0;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);

    int queue_counter = 0;
    MPI_Win queue_window = MPI_WIN_NULL;
    const bool use_rma_queue = size > 1;
    if (use_rma_queue) {
        MPI_Win_create(&queue_counter, sizeof(queue_counter), sizeof(int), MPI_INFO_NULL, comm, &queue_window);
    }

    const double started = now_seconds();
    std::vector<double> v(args.cols, 1.0);
    std::vector<double> local_col(args.cols, 0.0);
    std::vector<double> global_col(args.cols, 0.0);

    const double b_value = 1.0 / static_cast<double>(args.cols);
    Result result;
    double local_compute_sec = 0.0;
    double local_scheduler_sec = 0.0;
    long long processed_rows = 0;
    int processed_chunks = 0;

    for (int iteration = 1; iteration <= args.max_iters; ++iteration) {
        std::fill(local_col.begin(), local_col.end(), 0.0);
        QueuePhaseStats phase = use_rma_queue
            ? run_queue_column_phase(
                args,
                chunks,
                v,
                global_min,
                local_col,
                queue_counter,
                queue_window,
                rank,
                comm
            )
            : run_local_column_phase(args, chunks, v, global_min, local_col);
        local_compute_sec += phase.compute_sec;
        local_scheduler_sec += phase.scheduler_sec;
        processed_rows += phase.rows;
        processed_chunks += phase.chunks;

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
            std::fill(local_col.begin(), local_col.end(), 0.0);
            QueuePhaseStats check = use_rma_queue
                ? run_queue_final_phase(
                    args,
                    chunks,
                    v,
                    global_min,
                    local_col,
                    queue_counter,
                    queue_window,
                    rank,
                    comm
                )
                : run_local_final_phase(args, chunks, v, global_min, local_col);
            local_compute_sec += check.compute_sec;
            local_scheduler_sec += check.scheduler_sec;

            double row_error = 0.0;
            MPI_Allreduce(&check.row_error, &row_error, 1, MPI_DOUBLE, MPI_SUM, comm);
            MPI_Allreduce(local_col.data(), global_col.data(), args.cols, MPI_DOUBLE, MPI_SUM, comm);
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

    std::fill(local_col.begin(), local_col.end(), 0.0);
    QueuePhaseStats final = use_rma_queue
        ? run_queue_final_phase(
            args,
            chunks,
            v,
            global_min,
            local_col,
            queue_counter,
            queue_window,
            rank,
            comm
        )
        : run_local_final_phase(args, chunks, v, global_min, local_col);
    local_compute_sec += final.compute_sec;
    local_scheduler_sec += final.scheduler_sec;

    double row_error = 0.0;
    double objective = 0.0;
    MPI_Allreduce(&final.row_error, &row_error, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&final.objective, &objective, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(local_col.data(), global_col.data(), args.cols, MPI_DOUBLE, MPI_SUM, comm);

    double col_error = 0.0;
    for (int j = 0; j < args.cols; ++j) {
        col_error += std::abs(global_col[j] - b_value);
    }
    result.row_error = row_error;
    result.col_error = col_error;
    result.objective = objective;
    result.runtime_sec = now_seconds() - started;

    if (use_rma_queue) {
        MPI_Win_free(&queue_window);
    }

    gather_rank_metrics(
        result,
        weights,
        processed_rows,
        processed_chunks,
        local_compute_sec,
        processed_rows,
        processed_chunks,
        local_scheduler_sec,
        comm,
        rank,
        size
    );
    return result;
}

Result run_mpi(const Args& args, MPI_Comm comm, int rank, int size) {
    const std::vector<double> weights = resolve_rank_weights(args, size);
    if (args.partition_mode == "runtime-queue") {
        return run_mpi_runtime_queue(args, comm, rank, size, weights);
    }
    return run_mpi_static(args, comm, rank, size, weights);
}

void usage(const char* program) {
    std::cerr
        << "Usage: " << program << " --mode sequential|mpi --rows N --cols N --output FILE [options]\n"
        << "Options: --cost-mode random|squared_distance_1d|block --seed N --epsilon X\n"
        << "         --max-iters N --tol X --check-every N\n"
        << "         --comm-mode double|float32|quantized|lazy --sync-every N\n"
        << "         --kernel-cutoff X --quantization-scale X\n"
        << "         --partition-mode contiguous|weighted-chunk|runtime-queue\n"
        << "         --chunk-rows N --rank-weights CSV\n";
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
        } else if (key == "--partition-mode") {
            args.partition_mode = need_value(key);
        } else if (key == "--chunk-rows") {
            args.chunk_rows = std::stoi(need_value(key));
        } else if (key == "--rank-weights") {
            args.rank_weights = parse_double_csv(need_value(key));
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
        args.partition_mode != "contiguous" &&
        args.partition_mode != "weighted-chunk" &&
        args.partition_mode != "runtime-queue"
    ) {
        throw std::runtime_error("--partition-mode must be contiguous, weighted-chunk, or runtime-queue");
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
    if (args.chunk_rows <= 0) {
        throw std::runtime_error("--chunk-rows must be positive");
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
        std::string partition_suffix;
        if (args.partition_mode == "weighted-chunk") {
            partition_suffix = "_weighted_chunk";
        } else if (args.partition_mode == "runtime-queue") {
            partition_suffix = "_runtime_queue";
        }

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
            write_json(
                args,
                result,
                size,
                hostnames,
                "sinkhorn_cpp_mpi_" + args.comm_mode + partition_suffix + sparse_suffix
            );
            std::cout << "C++ MPI Sinkhorn complete: runtime=" << std::fixed << std::setprecision(6)
                      << result.runtime_sec << "s, iterations=" << result.iterations
                      << ", row_error=" << std::scientific << result.row_error
                      << ", col_error=" << result.col_error
                      << ", objective=" << std::fixed << std::setprecision(12) << result.objective
                      << ", processes=" << size
                      << ", comm_mode=" << args.comm_mode
                      << ", partition_mode=" << args.partition_mode
                      << ", column_syncs=" << result.column_syncs
                      << ", column_payload_bytes=" << result.column_payload_bytes
                      << ", kernel_cutoff=" << args.kernel_cutoff
                      << ", suggested_rank_weights=" << double_csv(result.suggested_rank_weights)
                      << std::endl;
        }
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << std::endl;
        return 1;
    }
}

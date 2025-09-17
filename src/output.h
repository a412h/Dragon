#ifndef OUTPUT_H
#define OUTPUT_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/mpi.h>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

using namespace dealii;

template <int dim, typename Number = double>
class VTUOutput {
public:
    VTUOutput(const DoFHandler<dim>& dof_handler, const std::string& basename)
        : dof_handler(dof_handler), basename(basename) {
        precompute_connectivity();
    }
    
    void write(const std::vector<std::array<Number, dim+2>>& U, 
               unsigned int cycle, 
               Number time) {
        
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        
        const unsigned int n_dofs = dof_handler.n_dofs();
        
        // Allocate all vectors at once
        Vector<Number> rho(n_dofs);
        std::vector<Vector<Number>> momentum(dim);
        for (int d = 0; d < dim; ++d) {
            momentum[d].reinit(n_dofs);
        }
        Vector<Number> E(n_dofs);
        Vector<Number> schlieren_rho(n_dofs);
        
        // Single pass to fill all vectors
        #pragma omp parallel for
        for (unsigned int i = 0; i < n_dofs; ++i) {
            rho[i] = U[i][0];
            for (int d = 0; d < dim; ++d) {
                momentum[d][i] = U[i][1 + d];
            }
            E[i] = U[i][dim + 1];
        }
        
        // Schlieren computation (expensive, remove if not needed)
        compute_schlieren(rho, schlieren_rho);
        
        data_out.add_data_vector(rho, "rho", DataOut<dim>::type_dof_data);
        
        if constexpr (dim == 1) {
            data_out.add_data_vector(momentum[0], "m_1", DataOut<dim>::type_dof_data);
        } else if constexpr (dim == 2) {
            data_out.add_data_vector(momentum[0], "m_1", DataOut<dim>::type_dof_data);
            data_out.add_data_vector(momentum[1], "m_2", DataOut<dim>::type_dof_data);
        } else if constexpr (dim == 3) {
            data_out.add_data_vector(momentum[0], "m_1", DataOut<dim>::type_dof_data);
            data_out.add_data_vector(momentum[1], "m_2", DataOut<dim>::type_dof_data);
            data_out.add_data_vector(momentum[2], "m_3", DataOut<dim>::type_dof_data);
        }
        
        data_out.add_data_vector(E, "E", DataOut<dim>::type_dof_data);
        data_out.add_data_vector(schlieren_rho, "schlieren_rho", DataOut<dim>::type_dof_data);
        
        // Build patches
        data_out.build_patches();
        
        // Write VTU file
        std::ostringstream filename;
        filename << basename << "_" << std::setfill('0') << std::setw(6) 
                 << cycle << ".vtu";
        
        std::ofstream output(filename.str());
        data_out.write_vtu(output);
        output.close();
        
        std::cout << "Output written: " << filename.str() 
                  << " at t=" << std::fixed << std::setprecision(4) << time 
                  << std::endl;
    }
    
private:
    const DoFHandler<dim>& dof_handler;
    const std::string basename;
    static constexpr Number gamma = Number(1.4);
    
    // Use precomputed connectivity for fast Schlieren computation
    struct NeighborInfo {
        std::vector<std::pair<unsigned int, std::array<Number, dim>>> neighbors;
    };
    std::vector<NeighborInfo> neighbor_connectivity;
    
    void precompute_connectivity() {
        const unsigned int n_dofs = dof_handler.n_dofs();
        neighbor_connectivity.resize(n_dofs);
        
        // Build connectivity once at construction
        for (const auto& cell : dof_handler.active_cell_iterators()) {
            for (unsigned int v = 0; v < cell->n_vertices(); ++v) {
                const unsigned int i = cell->vertex_dof_index(v, 0);
                const auto& p1 = cell->vertex(v);
                
                for (unsigned int v2 = 0; v2 < cell->n_vertices(); ++v2) {
                    if (v2 != v) {
                        const unsigned int j = cell->vertex_dof_index(v2, 0);
                        const auto& p2 = cell->vertex(v2);
                        
                        std::array<Number, dim> direction;
                        Number dist_squared = Number(0);
                        for (unsigned int d = 0; d < dim; ++d) {
                            direction[d] = p2[d] - p1[d];
                            dist_squared += direction[d] * direction[d];
                        }
                        
                        if (dist_squared > Number(1e-14)) {
                            // Normalize by distance squared for gradient computation
                            for (unsigned int d = 0; d < dim; ++d) {
                                direction[d] /= dist_squared;
                            }
                            
                            // Determine if already added
                            bool found = false;
                            for (const auto& [nj, _] : neighbor_connectivity[i].neighbors) {
                                if (nj == j) {
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) {
                                neighbor_connectivity[i].neighbors.push_back({j, direction});
                            }
                        }
                    }
                }
            }
        }
    }
    
    void compute_schlieren(const Vector<Number>& rho, Vector<Number>& schlieren) {
        /*
        Improved version
        */
        const unsigned int n_dofs = dof_handler.n_dofs();
        
        // Compute gradients in parallel
        #pragma omp parallel for
        for (unsigned int i = 0; i < n_dofs; ++i) {
            Number grad_x = Number(0);
            Number grad_y = Number(0);
            
            for (const auto& [j, direction] : neighbor_connectivity[i].neighbors) {
                const Number dr = rho[j] - rho[i];
                grad_x += dr * direction[0];
                if constexpr (dim >= 2) {
                    grad_y += dr * direction[1];
                }
            }
            
            if (!neighbor_connectivity[i].neighbors.empty()) {
                const Number inv_count = Number(1) / Number(neighbor_connectivity[i].neighbors.size());
                grad_x *= inv_count;
                grad_y *= inv_count;
            }
            
            Number grad_mag_squared = grad_x * grad_x;
            if constexpr (dim >= 2) {
                grad_mag_squared += grad_y * grad_y;
            }
            const Number grad_mag = std::sqrt(grad_mag_squared);
            
            const Number beta = Number(10.0);
            schlieren[i] = std::log(Number(1.0) + beta * grad_mag);
        }
        
        // Normalize in parallel
        Number min_val = *std::min_element(schlieren.begin(), schlieren.end());
        Number max_val = *std::max_element(schlieren.begin(), schlieren.end());
        
        if (max_val - min_val > Number(1e-14)) {
            const Number scale = Number(1) / (max_val - min_val);
            #pragma omp parallel for
            for (unsigned int i = 0; i < n_dofs; ++i) {
                schlieren[i] = (schlieren[i] - min_val) * scale;
            }
        }
    }
};

// Asyncronous writer class
template<int dim>
class AsyncVTUWriter {
private:
    std::thread writer_thread;
    std::queue<std::tuple<std::vector<std::array<double, dim+2>>, unsigned int, double>> write_queue;
    std::mutex queue_mutex;
    std::condition_variable cv_queue;
    std::condition_variable cv_empty;  // Separate CV for empty queue
    bool stop_flag = false;
    VTUOutput<dim>* output_handler;
    
public:
    AsyncVTUWriter(VTUOutput<dim>* handler) : output_handler(handler) {
        writer_thread = std::thread(&AsyncVTUWriter::writer_loop, this);
    }
    
    ~AsyncVTUWriter() {
        // Signal stop
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            stop_flag = true;
        }
        cv_queue.notify_all();
        
        // Wait for thread to finish
        if (writer_thread.joinable()) {
            writer_thread.join();
        }
    }
    
    void enqueue_write(std::vector<std::array<double, dim+2>>&& data, 
                      unsigned int cycle, double time) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            write_queue.push({std::move(data), cycle, time});
        }
        cv_queue.notify_one();
    }
    
    void wait_for_completion() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        cv_empty.wait(lock, [this] { return write_queue.empty(); });
    }
    
private:
    void writer_loop() {
        while (true) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            
            // Wait for work or stop signal
            cv_queue.wait(lock, [this] { return !write_queue.empty() || stop_flag; });
            
            // Check for exit condition
            if (stop_flag && write_queue.empty()) {
                break;
            }
            
            // Process all queued items
            while (!write_queue.empty()) {
                auto [data, cycle, time] = std::move(write_queue.front());
                write_queue.pop();
                
                // Notify if queue becomes empty
                if (write_queue.empty()) {
                    cv_empty.notify_all();
                }
                
                // Unlock while writing (allows more enqueues)
                lock.unlock();
                
                // Do the actual write
                output_handler->write(data, cycle, time);
                
                // Re-lock for next iteration
                lock.lock();
            }
        }
    }
};

#endif
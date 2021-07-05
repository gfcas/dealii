/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2021 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Authors: Fabian Castelli, Karlsruhe Institute of Technology (KIT)
 */

/**
 * Solve the Poisson equation with inhomogeneous Dirichlet boundary condition on
 * the Ldomain. Let \Omega = (-1,1)^2 \ (0,1)x(-1,0). Find a function
 * u:\Omega\to\IR, which solves the following problem:
 *    -\Delta u = f     in \Omega,
 *            u = u^D   on \ptl\Omega,
 * where f and u^D are given data:
 *            f = 0,
 *          u^D = u.
 * The exact solution is known:
 *    u(r,\phi) = r^(2/3) \sin(2/3 \pi).
 * Note the gradient of the exact solution is in the point 0 singular!
 * In the end a error analysis is performed and the eoc for the Linfty, the L2
 * and the H1 norm are calculated.
 * The solution of the linear system is calculated with a matrix-free method.
 * As solver the CG method with a geometric multigrid preconditioner is chosen.
 *
 * The code is based on dealii tutorials and ginkgo examples.
 */



#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/geometric_utilities.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/ginkgo_solver.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <ginkgo/ginkgo.hpp>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>



namespace GinkgoExample
{
  using namespace dealii;



  template <int dim>
  class Solution : public Function<dim>
  {
  public:
    Solution();

    virtual ~Solution() = default;

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;


    virtual Tensor<1, dim>
    gradient(const Point<dim> & p,
             const unsigned int component = 0) const override;
  };



  template <int dim>
  Solution<dim>::Solution()
    : Function<dim>()
  {
    AssertDimension(dim, 2);
  }



  template <int dim>
  double Solution<dim>::value(const Point<dim> & p,
                              const unsigned int component) const
  {
    AssertIndexRange(component, this->n_components);

    const std::array<double, dim> p_sphere =
      GeometricUtilities::Coordinates::to_spherical(p);

    const double &r   = p_sphere[0];
    const double &phi = p_sphere[1];

    constexpr const double alpha = 2.0 / 3.0;

    return std::pow(r, alpha) * std::sin(alpha * phi);
  }



  template <int dim>
  Tensor<1, dim> Solution<dim>::gradient(const Point<dim> & p,
                                         const unsigned int component) const
  {
    AssertIndexRange(component, this->n_components);

    const std::array<double, dim> p_sphere =
      GeometricUtilities::Coordinates::to_spherical(p);

    const double &r   = p_sphere[0];
    const double &phi = p_sphere[1];

    constexpr const double alpha = 2.0 / 3.0;

    Tensor<1, dim> return_value;

    return_value[0] =
      alpha * std::pow(r, -2.0 * alpha) *
      (p(0) * std::sin(alpha * phi) - p(1) * std::cos(alpha * phi));
    return_value[1] =
      alpha * std::pow(r, -2.0 * alpha) *
      (p(1) * std::sin(alpha * phi) + p(0) * std::cos(alpha * phi));

    return return_value;
  }



  template <int dim, int fe_degree>
  class PoissonSolver
  {
  public:
    PoissonSolver();

    void run();

  private:
    void make_grid();

    void setup_system();

    void assemble_system();

    void dealii_solve();

    void ginkgowrappers_solve();

    void ginkgo_solve();

    void solve();

    void compute_errors();

    void output_results(const unsigned int cycle) const;

    void build_convergence_table();



    const MappingQ1<dim> mapping;
    Triangulation<dim>   triangulation;
    FE_Q<dim>            fe;
    DoFHandler<dim>      dof_handler;

    AffineConstraints<double> constraints;
    SparsityPattern           sparsity_pattern;
    SparseMatrix<double>      system_matrix;

    Vector<double> solution;
    Vector<double> system_rhs;
    Vector<double> nodal_error;

    unsigned int solver_iterations;
    double       solver_time;

    ConvergenceTable convergence_table;

    TimerOutput computing_timer;
  };



  template <int dim, int fe_degree>
  PoissonSolver<dim, fe_degree>::PoissonSolver()
    : fe(fe_degree)
    , dof_handler(triangulation)
    , computing_timer(std::cout, TimerOutput::never, TimerOutput::wall_times)
  {}



  template <int dim, int fe_degree>
  void PoissonSolver<dim, fe_degree>::make_grid()
  {
    GridGenerator::hyper_L(triangulation);
    GridTools::rotate(-numbers::PI / 2, triangulation);
    triangulation.refine_global(3);
  }



  template <int dim, int fe_degree>
  void PoissonSolver<dim, fe_degree>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup system");

    // Clear system matrix
    system_matrix.clear();

    // Distribute DOFs
    dof_handler.distribute_dofs(fe);

    // Output information on cells and DOFs
    std::cout << "No. cells:     " << triangulation.n_global_active_cells()
              << std::endl;
    std::cout << "No. dofs:      " << dof_handler.n_dofs() << std::endl;

    // Set Dirichlet constraints
    constraints.clear();
    VectorTools::interpolate_boundary_values(
      mapping, dof_handler, 0, Solution<dim>(), constraints);
    constraints.close();

    // Generate sparsity pattern
    {
      DynamicSparsityPattern dsp(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      sparsity_pattern.copy_from(dsp);
    }

    // Reinit system matrix and vectors
    system_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }



  template <int dim, int fe_degree>
  void PoissonSolver<dim, fe_degree>::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "assemble system");

    // Reset system matrix and right hand side vector
    system_matrix = 0.0;
    system_rhs    = 0.0;

    // Set up FEValues
    FEValues<dim> fe_values(mapping,
                            fe,
                            QGauss<dim>(fe.degree + 1),
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    // Set up local objects
    const unsigned int dofs_per_cell = fe_values.dofs_per_cell;
    const unsigned int n_q_points    = fe_values.n_quadrature_points;

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> JxW(n_q_points);

    // Assemble system by looping over all cells
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs    = 0;

        JxW = fe_values.get_JxW_values();

        // Loop over all quadrature points
        for (const unsigned int q : fe_values.quadrature_point_indices())
          {
            const double dx = JxW[q];

            // Loop over all local DOFS
            for (const unsigned int i : fe_values.dof_indices())
              {
                // Store gradients
                const Tensor<1, dim> grad_phi_i = fe_values.shape_grad(i, q);

                // Loop over all local DOFs to assemble local system matrix
                for (const unsigned int j : fe_values.dof_indices())
                  {
                    // Store gradients
                    const Tensor<1, dim> grad_phi_j =
                      fe_values.shape_grad(j, q);

                    // Compute local system matrix contribution
                    cell_matrix(i, j) += grad_phi_i * grad_phi_j * dx;
                  }

                // Compute local right hand side contribution
                cell_rhs(i) += 0.0;
              }
          }

        // Get the local to global DOF relation
        cell->get_dof_indices(local_dof_indices);

        // Distribute the local assembly data into the global system
        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      }
  }



  template <int dim, int fe_degree>
  void PoissonSolver<dim, fe_degree>::dealii_solve()
  {
    TimerOutput::Scope t(computing_timer, "solve (dealii)");

    // Write solver information
    std::cout << "This is dealii::SolverCG<Vector<double>>" << std::endl;

    // Set up SolverControl
    SolverControl solver_control(dof_handler.n_dofs(), 1e-12);

    // Set up CG solver
    SolverCG<Vector<double>> solver(solver_control);

    // Use indentity preconditioner
    // PreconditionIdentity preconditioner;
    PreconditionBlockJacobi<SparseMatrix<double>> preconditioner;
    PreconditionBlockJacobi<SparseMatrix<double>>::AdditionalData
      preconditioner_data(fe_degree);
    preconditioner.initialize(system_matrix, preconditioner_data);
    // PreconditionSSOR<SparseMatrix<double>> preconditioner;
    // preconditioner.initialize(
    // system_matrix,
    // PreconditionJacobi<SparseMatrix<double>>::AdditionalData(1.0));

    // Reset solution vector to use zero starting value
    solution = 0.0;

    // Initialize timer
    Timer timer;

    // Solve
    solver.solve(system_matrix, solution, system_rhs, preconditioner);

    // Stop timer
    timer.stop();

    // Distribute the Dirichlet constraints
    constraints.distribute(solution);

    // Store solution time and iteration number
    solver_time       = timer.wall_time();
    solver_iterations = solver_control.last_step();

    // Write information on terminal
    std::cout << "   " << solver_control.last_step()
              << " CG iterations needed to obtain convergence." << std::endl;
    std::cout << "   " << solver_control.last_value() << " = last residuum"
              << std::endl;
    std::cout << "Time solve (" << solver_control.last_step()
              << " iterations)  (CPU/Wall) " << timer.cpu_time() << "s / "
              << timer.wall_time() << "s" << std::endl;
  }



  template <int dim, int fe_degree>
  void PoissonSolver<dim, fe_degree>::ginkgowrappers_solve()
  {
    TimerOutput::Scope t(computing_timer, "solve (ginkgowrappers)");

    // Write solver information
    std::cout << "This is GinkgoWrappers::SolverCG<double, int>" << std::endl;

    // Set up SolverControl
    SolverControl solver_control(dof_handler.n_dofs(), 1e-12);

    // Set up CG solver
    const auto executor_string = "omp";
    // std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
    //   exec_map{
    //     {"omp", [] { return gko::OmpExecutor::create(); }},
    //     {"cuda",
    //      [] {
    //        return gko::CudaExecutor::create(0,
    //                                         gko::OmpExecutor::create(),
    //                                         true);
    //      }},
    //     {"hip",
    //      [] {
    //        return gko::HipExecutor::create(0, gko::OmpExecutor::create(),
    //        true);
    //      }},
    //     {"dpcpp",
    //      [] {
    //        return gko::DpcppExecutor::create(0, gko::OmpExecutor::create());
    //      }},
    //     {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // // executor where Ginkgo will perform the computation
    // const auto exec = exec_map.at(executor_string)(); // throws if not valid

    // using bj          = gko::preconditioner::Jacobi<double, int>;
    // auto prec_factory = bj::build().on(exec);

    // auto prec = prec_factory->generate();

    // TODO preconditioner

    GinkgoWrappers::SolverCG<double, int> solver(solver_control,
                                                 executor_string);

    // Reset solution vector to use zero starting value
    solution = 0.0;

    // Initialize timer
    Timer timer;

    // Solve
    solver.solve(system_matrix, solution, system_rhs);

    // Stop timer
    timer.stop();

    // Distribute the Dirichlet constraints
    constraints.distribute(solution);

    // Store solution time and iteration number
    solver_time       = timer.wall_time();
    solver_iterations = solver_control.last_step();

    // Write information on terminal
    std::cout << "   " << solver_control.last_step()
              << " CG iterations needed to obtain convergence." << std::endl;
    std::cout << "   " << solver_control.last_value() << " = last residuum"
              << std::endl;
    std::cout << "Time solve (" << solver_control.last_step()
              << " iterations)  (CPU/Wall) " << timer.cpu_time() << "s / "
              << timer.wall_time() << "s" << std::endl;
  }



  template <int dim, int fe_degree>
  void PoissonSolver<dim, fe_degree>::ginkgo_solve()
  {
    TimerOutput::Scope t(computing_timer, "solve (ginkgo)");

    // Write solver information
    std::cout << "This is gko::solver::Cg<double,int>" << std::endl;

    // Assert that the system be symmetric.
    Assert(system_matrix.m() == system_matrix.n(), ExcNotQuadratic());
    auto num_rows = system_matrix.m();

    // Reset solution vector to use zero starting value
    solution = 0.0;

    // Make a copy of the rhs to use with Ginkgo.
    std::vector<double> rhs(num_rows);
    std::copy(system_rhs.begin(), system_rhs.begin() + num_rows, rhs.begin());

    // Ginkgo setup
    // Some shortcuts: A vector is a Dense matrix with co-dimension 1.
    // The matrix is setup in CSR. But various formats can be used. Look at
    // Ginkgo's documentation.
    using ValueType     = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType     = int;
    using vec           = gko::matrix::Dense<ValueType>;
    using real_vec      = gko::matrix::Dense<RealValueType>;
    using mtx           = gko::matrix::Csr<ValueType, IndexType>;
    using cg            = gko::solver::Cg<ValueType>;
    using bj            = gko::preconditioner::Jacobi<ValueType, IndexType>;
    using val_array     = gko::Array<double>;



    // Where the code is to be executed. Can be changed to `omp` or `cuda` to
    // run on multiple threads or on gpu's
    const auto executor_string = "omp";
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
      exec_map{
        {"omp", [] { return gko::OmpExecutor::create(); }},
        {"cuda",
         [] {
           return gko::CudaExecutor::create(0,
                                            gko::OmpExecutor::create(),
                                            true);
         }},
        {"hip",
         [] {
           return gko::HipExecutor::create(0, gko::OmpExecutor::create(), true);
         }},
        {"dpcpp",
         [] {
           return gko::DpcppExecutor::create(0, gko::OmpExecutor::create());
         }},
        {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)(); // throws if not valid

    // Setup Ginkgo's data structures
    auto             b       = vec::create(exec,
                         gko::dim<2>(num_rows, 1),
                         val_array::view(exec, num_rows, rhs.data()),
                         1);
    auto             x       = vec::create(exec, gko::dim<2>(num_rows, 1));
    auto             A       = mtx::create(exec,
                         gko::dim<2>(num_rows),
                         system_matrix.n_nonzero_elements());
    mtx::value_type *values  = A->get_values();
    mtx::index_type *row_ptr = A->get_row_ptrs();
    mtx::index_type *col_idx = A->get_col_idxs();

    // Convert to standard CSR format
    // As deal.ii does not expose its system matrix pointers, we construct them
    // individually.
    row_ptr[0] = 0;
    for (auto row = 1; row <= num_rows; ++row)
      {
        row_ptr[row] = row_ptr[row - 1] + system_matrix.get_row_length(row - 1);
      }

    std::vector<mtx::index_type> ptrs(num_rows + 1);
    std::copy(A->get_row_ptrs(),
              A->get_row_ptrs() + num_rows + 1,
              ptrs.begin());
    for (auto row = 0; row < system_matrix.m(); ++row)
      {
        for (auto p = system_matrix.begin(row); p != system_matrix.end(row);
             ++p)
          {
            // write entry into the first free one for this row
            col_idx[ptrs[row]] = p->column();
            values[ptrs[row]]  = p->value();

            // then move pointer ahead
            ++ptrs[row];
          }
      }

    // Ginkgo solve
    // Define convergence criteria
    const RealValueType reduction_factor = 1e-12;
    auto                iter_stop        = gko::stop::Iteration::build()
                       .with_max_iters(dof_handler.n_dofs())
                       .on(exec);
    auto tol_stop = gko::stop::ResidualNorm<ValueType>::build()
                      .with_reduction_factor(reduction_factor)
                      .on(exec);

    // Create logger
    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
      gko::log::Convergence<ValueType>::create(exec);
    iter_stop->add_logger(logger);
    tol_stop->add_logger(logger);

    // The stopping criteria is set at maximum iterations of n_dofs and a
    // reduction factor of 1e-12. For other options, refer to Ginkgo's
    // documentation.
    auto solver_gen =
      cg::build()
        .with_criteria(gko::share(iter_stop), gko::share(tol_stop))
        // .with_preconditioner(bj::build().with_max_block_size(8u).on(exec)) //
        // standard Jacobi
        .with_preconditioner(
          bj::build()
            .with_max_block_size(8u)
            .with_storage_optimization(gko::precision_reduction::autodetect())
            .on(exec)) // mixed precision Jacobi
        .on(exec);
    solver_gen->add_logger(logger);
    auto solver = solver_gen->generate(gko::give(A));

    // Synchronize for performance analysis
    exec->synchronize();

    // Start timer
    Timer timer;

    // Solve system
    solver->apply(gko::lend(b), gko::lend(x));

    // Stop timer and compute solution time
    timer.stop();

    // Copy the solution vector back to dealii's data structures.
    std::copy(x->get_values(), x->get_values() + num_rows, solution.begin());

    // Distribute the Dirichlet constraints
    constraints.distribute(solution);

    // Calculate residual
    auto one     = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto initres = gko::initialize<real_vec>({0.0}, exec);
    auto res     = gko::initialize<real_vec>({0.0}, exec);
    // A->apply(gko::lend(one), gko::lend(x), gko::lend(neg_one), gko::lend(b));
    // // TODO??
    // b->compute_norm2(gko::lend(res)); auto impl_res =
    // gko::as<real_vec>(logger->get_implicit_sq_resnorm());

    // std::cout << "Initial residual norm sqrt(r^T r):\n";
    // write(std::cout, gko::lend(initres));
    // std::cout << "Final residual norm sqrt(r^T r):\n";
    // write(std::cout, gko::lend(res));
    // std::cout << "Implicit residual norm squared (r^2):\n";
    // write(std::cout, gko::lend(impl_res));

    // // Print solver statistics
    // std::cout << "CG iteration count:     " << logger->get_num_iterations()
    //           << std::endl;

    // Store solution time and iteration number
    solver_time       = timer.wall_time();
    solver_iterations = logger->get_num_iterations();

    // Write information on terminal
    std::cout << "   " << solver_iterations
              << " CG iterations needed to obtain convergence." << std::endl;
    // // std::cout << "   " << solver_control.last_value() << " = last
    // residuum"
    // // << std::endl;
    std::cout << "Time solve (" << solver_iterations
              << " iterations)  (CPU/Wall) " << timer.cpu_time() << "s / "
              << timer.wall_time() << "s" << std::endl;
    // std::cout << "Time solve                  (CPU/Wall) " <<
    // timer.cpu_time()
    //           << "s / " << timer.wall_time() << "s" << std::endl;
  }



  template <int dim, int fe_degree>
  void PoissonSolver<dim, fe_degree>::solve()
  {
    TimerOutput::Scope t(computing_timer, "solve");

    // dealii_solve();

    ginkgowrappers_solve();

    // ginkgo_solve();
  }



  template <int dim, int fe_degree>
  void PoissonSolver<dim, fe_degree>::compute_errors()
  {
    TimerOutput::Scope t(computing_timer, "compute errors");

    // Interpolate exact solution
    Vector<double> exact_solution(solution);
    VectorTools::interpolate(mapping,
                             dof_handler,
                             Solution<dim>(),
                             exact_solution);

    // Compute nodal error
    nodal_error.reinit(solution);
    nodal_error.equ(1.0, solution);
    nodal_error.add(-1.0, exact_solution);
    for (auto &el : nodal_error)
      {
        el = std::abs(el);
      }

    // Set up error computation
    Vector<float> diff_per_cell(triangulation.n_active_cells());
    QGauss<dim>   quadrature(fe.degree + 2);

    // Compute Loo error
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution,
                                      Solution<dim>(),
                                      diff_per_cell,
                                      quadrature,
                                      VectorTools::Linfty_norm);
    const double Linfty_error =
      VectorTools::compute_global_error(triangulation,
                                        diff_per_cell,
                                        VectorTools::Linfty_norm);

    // Compute L2 error
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution,
                                      Solution<dim>(),
                                      diff_per_cell,
                                      quadrature,
                                      VectorTools::L2_norm);
    const double L2_error =
      VectorTools::compute_global_error(triangulation,
                                        diff_per_cell,
                                        VectorTools::L2_norm);

    // Compute H1s error
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution,
                                      Solution<dim>(),
                                      diff_per_cell,
                                      quadrature,
                                      VectorTools::H1_seminorm);
    const double H1s_error =
      VectorTools::compute_global_error(triangulation,
                                        diff_per_cell,
                                        VectorTools::H1_seminorm);

    // Compute H1 error
    const double H1_error =
      std::sqrt(L2_error * L2_error + H1s_error * H1s_error);

    // Write errors on the terminal
    std::cout << "Linfty error:  " << Linfty_error << std::endl;
    std::cout << "L2 error:      " << L2_error << std::endl;
    std::cout << "H1 error:      " << H1_error << std::endl;

    // Add errors and values to convergence table
    convergence_table.add_value("cells", triangulation.n_global_active_cells());
    convergence_table.add_value("dofs", dof_handler.n_dofs());
    // convergence_table.add_value("Linfty", Linfty_error);
    convergence_table.add_value("L2", L2_error);
    // convergence_table.add_value("H1s", H1s_error);
    convergence_table.add_value("H1", H1_error);
    convergence_table.add_value("time", solver_time);
    convergence_table.add_value("iter", solver_iterations);
    convergence_table.add_value("tpi", solver_time / solver_iterations);
  }



  template <int dim, int fe_degree>
  void
  PoissonSolver<dim, fe_degree>::output_results(const unsigned int cycle) const
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.add_data_vector(nodal_error, "nodal_error");

    data_out.build_patches(mapping,
                           fe.degree,
                           DataOut<dim>::curved_inner_cells);

    DataOutBase::VtkFlags flags;
    flags.compression_level        = DataOutBase::VtkFlags::best_speed;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    std::string filename = "solution-p" + std::to_string(fe_degree) + "-" +
                           std::to_string(dim) + "d";

    data_out.write_vtu_with_pvtu_record(
      "./", filename, cycle, MPI_COMM_WORLD, 3);
  }



  template <int dim, int fe_degree>
  void PoissonSolver<dim, fe_degree>::build_convergence_table()
  {
    TimerOutput::Scope t(computing_timer, "build convergence table");

    // convergence_table.set_precision("Linfty", 2);
    convergence_table.set_precision("L2", 2);
    // convergence_table.set_precision("H1s", 2);
    convergence_table.set_precision("H1", 2);
    convergence_table.set_precision("time", 2);

    // convergence_table.set_scientific("Linfty", true);
    convergence_table.set_scientific("L2", true);
    // convergence_table.set_scientific("H1s", true);
    convergence_table.set_scientific("H1", true);

    // convergence_table.evaluate_convergence_rates(
    //   "Linfty", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "L2", ConvergenceTable::reduction_rate_log2);
    // convergence_table.evaluate_convergence_rates(
    //   "H1s", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "H1", ConvergenceTable::reduction_rate_log2);

    // Write convergence table to console
    convergence_table.write_text(std::cout);

    // Write convergence table to file
    std::string filename = "./convergence-table-p" + std::to_string(fe_degree) +
                           "-" + std::to_string(dim) + "d.dat";
    std::ofstream out(filename);
    convergence_table.write_text(out);
  }



  template <int dim, int fe_degree>
  void PoissonSolver<dim, fe_degree>::run()
  {
    // General output before the program starts
    {
      std::string DAT_header = "START DATE: " + Utilities::System::get_date() +
                               ", TIME: " + Utilities::System::get_time();

      const unsigned int header_size = 80;

      std::cout << std::string(header_size, '=') << std::endl;
      std::cout << DAT_header << std::endl;
      std::cout << std::string(header_size, '=') << std::endl;
    }

    std::cout << "Solving problem in " << dim << " space dimensions"
              << std::endl;

    std::cout << std::string(80, '=') << std::endl;


    for (unsigned int cycle = 1; cycle <= 5; ++cycle)
      {
        std::cout << "Cycle " << cycle << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        if (cycle == 1)
          {
            make_grid();
          }
        else
          {
            triangulation.refine_global();
          }

        setup_system();

        assemble_system();

        solve();

        compute_errors();

        output_results(cycle);

        computing_timer.print_summary();

        std::cout << std::endl;
      }

    build_convergence_table();


    // General output before the program ends
    {
      std::string DAT_header = "END DATE: " + Utilities::System::get_date() +
                               ", TIME: " + Utilities::System::get_time();

      std::cout << std::string(80, '=') << std::endl;
      std::cout << DAT_header << std::endl;
      std::cout << std::string(80, '=') << std::endl;
    }
  }
} // namespace GinkgoExample



int main(int argc, char **argv)
{
  try
    {
      // Print version information
      std::cout << gko::version_info::get() << std::endl;

      using namespace GinkgoExample;

      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      {
        PoissonSolver<2, 1> pp;
        pp.run();
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}

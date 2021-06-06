/**
 * Solve the Poisson equation with inhomogeneous Dirichlet boundary condition on
 * the L-domain. Let \Omega = (-1,1)^2 \ (0,1)x(-1,0). Find a function
 * u:\Omega\to\IR, which solves the following problem:
 *    -\Delta u = f     in \Omega,
 *            u = u^D   on \ptl\Omega,
 * where f and u^D are given data:
 *            f = 0,
 *          u^D = u_ex.
 * The exact solution is assumed to be:
 *    u_ex(r,\phi) = r^(2/3) \sin(2/3 \pi).
 * Note the gradient of the exact solution is in the point 0 singular!
 * In the end a error analysis is performed and the experimental order of
 * convergence (eoc) for the Linfty, the L2 and the H1 norm are calculated.
 * The solution of the linear system is calculated with a matrix-free CG method.
 * As solver the CG method with a geometric multigrid preconditioner is chosen.
 *
 * This code contains ideas from several dealii tutorial steps and of the
 * lecture/exercise class EWR SoSe 2013 of W. Doerfler.
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/geometric_utilities.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <fstream>
#include <iostream>

namespace Step75b
{
  using namespace dealii;



  template <int dim>
  class Solution : public Function<dim>
  {
  public:
    Solution() = default;

    virtual ~Solution() = default;

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

    virtual Tensor<1, dim>
    gradient(const Point<dim> & p,
             const unsigned int component = 0) const override;
  };



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



  struct Parameters
  {
    unsigned int n_cycles           = 12;
    unsigned int initial_refinement = 3;

    struct
    {
      std::string  type      = "chebyshev";
      double       tolerance = 1e-3;
      unsigned int degree    = numbers::invalid_unsigned_int;
    } coarse_solver;

    struct
    {
      std::string  type                = "chebyshev";
      double       smoothing_range     = 15;
      unsigned int degree              = 4;
      unsigned int eig_cg_n_iterations = 10;
    } smoother;

    struct
    {
      std::string  type      = "cg";
      unsigned int max_steps = 100;
      double       tolerance = 1e-12;
    } linear_solver;

    struct
    {
      double refine_fraction  = 0.2;
      double coarsen_fraction = 0.02;

      std::string criterion = "global"; // "kelly";

      std::string strategy = "number"; // "energy";
    } adaptive_refinement;
  };



  template <int dim, int fe_degree, typename number>
  class LaplaceOperator
    : public MatrixFreeOperators::
        Base<dim, LinearAlgebra::distributed::Vector<number>>
  {
  public:
    using value_type = number;
    using FECellIntegrator =
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number>;

    LaplaceOperator() = default;

    virtual void compute_diagonal() override;

  private:
    virtual void apply_add(
      LinearAlgebra::distributed::Vector<number> &      dst,
      const LinearAlgebra::distributed::Vector<number> &src) const override;

    void
    local_apply(const MatrixFree<dim, number> &                   data,
                LinearAlgebra::distributed::Vector<number> &      dst,
                const LinearAlgebra::distributed::Vector<number> &src,
                const std::pair<unsigned int, unsigned int> &cell_range) const;

    void local_compute_diagonal(FECellIntegrator &integrator) const;
  };



  template <int dim, int fe_degree, typename number>
  void LaplaceOperator<dim, fe_degree, number>::local_apply(
    const MatrixFree<dim, number> &                   data,
    LinearAlgebra::distributed::Vector<number> &      dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    FECellIntegrator phi(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);

        phi.gather_evaluate(src, EvaluationFlags::gradients);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            phi.submit_gradient(phi.get_gradient(q), q);
          }

        phi.integrate_scatter(EvaluationFlags::gradients, dst);
      }
  }



  template <int dim, int fe_degree, typename number>
  void LaplaceOperator<dim, fe_degree, number>::apply_add(
    LinearAlgebra::distributed::Vector<number> &      dst,
    const LinearAlgebra::distributed::Vector<number> &src) const
  {
    this->data->cell_loop(&LaplaceOperator::local_apply, this, dst, src);
  }



  template <int dim, int fe_degree, typename number>
  void LaplaceOperator<dim, fe_degree, number>::compute_diagonal()
  {
    this->inverse_diagonal_entries.reset(
      new DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>());
    LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
      this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inverse_diagonal);
    MatrixFreeTools::compute_diagonal(*this->data,
                                      inverse_diagonal,
                                      &LaplaceOperator::local_compute_diagonal,
                                      this);
    for (auto &diag : inverse_diagonal)
      {
        diag = (std::abs(diag) > 1.0e-10) ? (1.0 / diag) : 1.0;
      }
  }



  template <int dim, int fe_degree, typename number>
  void LaplaceOperator<dim, fe_degree, number>::local_compute_diagonal(
    FECellIntegrator &phi) const
  {
    phi.evaluate(EvaluationFlags::gradients);

    for (unsigned int q = 0; q < phi.n_q_points; ++q)
      {
        phi.submit_gradient(phi.get_gradient(q), q);
      }
    phi.integrate(EvaluationFlags::gradients);
  }



  template <int dim, int fe_degree>
  class PoissonProblem
  {
  public:
    PoissonProblem(const Parameters &parameters);

    void run();

  private:
    void setup_system();

    void local_assemble_rhs(
      const MatrixFree<dim, double> &                   data,
      LinearAlgebra::distributed::Vector<double> &      dst,
      const LinearAlgebra::distributed::Vector<double> &src,
      const std::pair<unsigned int, unsigned int> &     cell_range) const;

    void assemble_rhs();

    void solve();

    void graphical_output(const unsigned int cycle) const;

    void compute_estimates();

    void compute_errors();

    void refine_grid();

    void build_convergence_table();



    const Parameters prm;

    const MappingQ1<dim>                      mapping;
    parallel::distributed::Triangulation<dim> triangulation;
    FE_Q<dim>                                 fe;
    DoFHandler<dim>                           dof_handler;

    AffineConstraints<double> constraints;

    LaplaceOperator<dim, fe_degree, double> system_matrix;


    MGConstrainedDoFs mg_constrained_dofs;
    using LevelMatrixType = LaplaceOperator<dim, fe_degree, float>;
    MGLevelObject<LevelMatrixType> mg_matrices;

    LinearAlgebra::distributed::Vector<double> system_rhs;
    LinearAlgebra::distributed::Vector<double> solution;

    LinearAlgebra::distributed::Vector<double> nodal_error;

    Vector<float> estimate_per_cell;

    double setup_time;

    unsigned int solver_iterations;
    double       solver_time;

    ConvergenceTable convergence_table;

    ConditionalOStream pcout;
    ConditionalOStream time_details;

    TimerOutput computing_timer;
  };



  template <int dim, int fe_degree>
  PoissonProblem<dim, fe_degree>::PoissonProblem(const Parameters &parameters)
    : prm(parameters)
    , triangulation(MPI_COMM_WORLD,
                    Triangulation<dim>::limit_level_difference_at_vertices,
                    parallel::distributed::Triangulation<
                      dim>::construct_multigrid_hierarchy)
    , fe(fe_degree)
    , dof_handler(triangulation)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , time_details(std::cout,
                   false &&
                     Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , computing_timer(MPI_COMM_WORLD,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
  {}



  template <int dim, int fe_degree>
  void PoissonProblem<dim, fe_degree>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup system");

    Timer time;
    setup_time = 0.0;

    system_matrix.clear();
    mg_matrices.clear_elements();

    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();

    pcout << "No. cells: " << triangulation.n_global_active_cells()
          << std::endl;
    pcout << "No. dofs:  " << dof_handler.n_dofs() << std::endl;


    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Solution<dim>(),
                                             constraints);
    constraints.close();

    setup_time += time.wall_time();
    time_details << "Distribute DoFs & B.C.      (CPU/Wall) " << time.cpu_time()
                 << "s / " << time.wall_time() << "s" << std::endl;
    time.restart();

    {
      typename MatrixFree<dim, double>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
        MatrixFree<dim>::AdditionalData::none;
      additional_data.mapping_update_flags =
        (update_values | update_gradients | update_JxW_values |
         update_quadrature_points);
      std::shared_ptr<MatrixFree<dim, double>> system_mf_storage(
        new MatrixFree<dim, double>());
      system_mf_storage->reinit(mapping,
                                dof_handler,
                                constraints,
                                QGauss<1>(fe_degree + 1),
                                additional_data);
      system_matrix.initialize(system_mf_storage);
    }

    system_matrix.initialize_dof_vector(system_rhs);
    system_matrix.initialize_dof_vector(solution);

    setup_time += time.wall_time();
    time_details << "Setup matrix-free system    (CPU/Wall) " << time.cpu_time()
                 << "s / " << time.wall_time() << "s" << std::endl;
    time.restart();

    const unsigned int n_levels = triangulation.n_global_levels();
    mg_matrices.resize(0, n_levels - 1);

    std::set<types::boundary_id> dirichlet_boundary;
    dirichlet_boundary.insert(0);
    mg_constrained_dofs.initialize(dof_handler);
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                       dirichlet_boundary);

    for (unsigned int level = 0; level < n_levels; ++level)
      {
        IndexSet relevant_dofs;
        DoFTools::extract_locally_relevant_level_dofs(dof_handler,
                                                      level,
                                                      relevant_dofs);

        AffineConstraints<double> level_constraints;
        level_constraints.clear();
        level_constraints.reinit(relevant_dofs);
        level_constraints.add_lines(
          mg_constrained_dofs.get_boundary_indices(level));
        level_constraints.close();

        typename MatrixFree<dim, float>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme =
          MatrixFree<dim, float>::AdditionalData::none;
        additional_data.mapping_update_flags =
          (update_values | update_gradients | update_JxW_values |
           update_quadrature_points);
        additional_data.mg_level = level;
        std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level(
          new MatrixFree<dim, float>());
        mg_mf_storage_level->reinit(mapping,
                                    dof_handler,
                                    level_constraints,
                                    QGauss<1>(fe_degree + 1),
                                    additional_data);

        mg_matrices[level].initialize(mg_mf_storage_level,
                                      mg_constrained_dofs,
                                      level);
      }

    setup_time += time.wall_time();
    time_details << "Setup matrix-free levels    (CPU/Wall) " << time.cpu_time()
                 << "s / " << time.wall_time() << "s" << std::endl;
  }



  template <int dim, int fe_degree>
  void PoissonProblem<dim, fe_degree>::local_assemble_rhs(
    const MatrixFree<dim, double> &                   data,
    LinearAlgebra::distributed::Vector<double> &      dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range) const
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> phi(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);

        phi.read_dof_values_plain(src);
        phi.evaluate(EvaluationFlags::gradients);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            phi.submit_gradient(phi.get_gradient(q), q);
          }

        phi.integrate_scatter(EvaluationFlags::gradients, dst);
      }
  }



  template <int dim, int fe_degree>
  void PoissonProblem<dim, fe_degree>::assemble_rhs()
  {
    TimerOutput::Scope t(computing_timer, "assemble right hand side");

    Timer time;

    solution = 0.0;
    constraints.distribute(solution);
    solution.update_ghost_values();

    auto matrix_free = system_matrix.get_matrix_free();

    matrix_free->cell_loop(
      &PoissonProblem::local_assemble_rhs, this, system_rhs, solution, true);

    system_rhs *= -1.0;

    setup_time += time.wall_time();
    time_details << "Assemble right hand side    (CPU/Wall) " << time.cpu_time()
                 << "s / " << time.wall_time() << "s" << std::endl;
  }



  template <int dim, int fe_degree>
  void PoissonProblem<dim, fe_degree>::solve()
  {
    TimerOutput::Scope t(computing_timer, "solve");

    AssertThrow(prm.coarse_solver.type == "chebyshev", ExcNotImplemented());
    AssertThrow(prm.smoother.type == "chebyshev", ExcNotImplemented());
    AssertThrow(prm.linear_solver.type == "cg", ExcNotImplemented());

    Timer time;

    MGTransferMatrixFree<dim, float> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(dof_handler);

    setup_time += time.wall_time();
    time_details << "MG build transfer time      (CPU/Wall) " << time.cpu_time()
                 << "s / " << time.wall_time() << "s" << std::endl;

    using SmootherType =
      PreconditionChebyshev<LevelMatrixType,
                            LinearAlgebra::distributed::Vector<float>>;
    mg::SmootherRelaxation<SmootherType,
                           LinearAlgebra::distributed::Vector<float>>
                                                         mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels() - 1);

    for (unsigned int level = 0; level < triangulation.n_global_levels();
         ++level)
      {
        if (level > 0)
          {
            smoother_data[level].smoothing_range = prm.smoother.smoothing_range;
            smoother_data[level].degree          = prm.smoother.degree;
            smoother_data[level].eig_cg_n_iterations =
              prm.smoother.eig_cg_n_iterations;
          }
        else
          {
            smoother_data[0].smoothing_range     = prm.coarse_solver.tolerance;
            smoother_data[0].degree              = prm.coarse_solver.degree;
            smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
          }
        mg_matrices[level].compute_diagonal();
        smoother_data[level].preconditioner =
          mg_matrices[level].get_matrix_diagonal_inverse();
      }
    mg_smoother.initialize(mg_matrices, smoother_data);

    MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<float>>
      mg_coarse;
    mg_coarse.initialize(mg_smoother);

    mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_matrix(
      mg_matrices);

    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>>
      mg_interface_matrices;
    mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels();
         ++level)
      mg_interface_matrices[level].initialize(mg_matrices[level]);
    mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_interface(
      mg_interface_matrices);

    Multigrid<LinearAlgebra::distributed::Vector<float>> mg(
      mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
    mg.set_edge_matrices(mg_interface, mg_interface);

    PreconditionMG<dim,
                   LinearAlgebra::distributed::Vector<float>,
                   MGTransferMatrixFree<dim, float>>
      preconditioner(dof_handler, mg, mg_transfer);


    SolverControl solver_control(prm.linear_solver.max_steps,
                                 prm.linear_solver.tolerance);

    SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

    setup_time += time.wall_time();
    time_details << "MG build smoother time      (CPU/Wall) " << time.cpu_time()
                 << "s / " << time.wall_time() << "s" << std::endl;
    time_details << "Total setup time                (Wall) " << setup_time
                 << "s" << std::endl;

    time.restart();

    // constraints.set_zero(solution);
    solution = 0.0;
    cg.solve(system_matrix, solution, system_rhs, preconditioner);

    constraints.distribute(solution);

    solver_time       = time.wall_time();
    solver_iterations = solver_control.last_step();

    time_details << "Time solve (" << solver_control.last_step()
                 << " iterations)  (CPU/Wall) " << time.cpu_time() << "s / "
                 << time.wall_time() << "s" << std::endl;
  }



  template <int dim, int fe_degree>
  void PoissonProblem<dim, fe_degree>::compute_estimates()
  {
    TimerOutput::Scope timer_section(computing_timer, "compute estimates");

    solution.update_ghost_values();
    estimate_per_cell.reinit(triangulation.n_active_cells());

    Timer time;

    if (prm.adaptive_refinement.criterion == "global")
      {
        estimate_per_cell = 0.0;
      }
    else if (prm.adaptive_refinement.criterion == "kelly")
      {
        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                locally_relevant_dofs);

        LinearAlgebra::distributed::Vector<double> copy_vector(solution);
        solution.reinit(dof_handler.locally_owned_dofs(),
                        locally_relevant_dofs,
                        triangulation.get_communicator());
        solution.copy_locally_owned_data_from(copy_vector);
        constraints.distribute(solution);
        solution.update_ghost_values();

        time.restart();

        KellyErrorEstimator<dim>::estimate(dof_handler,
                                           QGauss<dim - 1>(fe_degree + 1),
                                           {},
                                           solution,
                                           estimate_per_cell);

        time_details << "Estimate (Kelly)            (CPU/Wall) "
                     << time.cpu_time() << "s / " << time.wall_time() << "s"
                     << std::endl;
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }
  }



  template <int dim, int fe_degree>
  void PoissonProblem<dim, fe_degree>::compute_errors()
  {
    TimerOutput::Scope t(computing_timer, "compute errors");

    solution.update_ghost_values();
    LinearAlgebra::distributed::Vector<double> exact_solution(solution);
    VectorTools::interpolate(dof_handler, Solution<dim>(), exact_solution);

    nodal_error.reinit(solution);
    nodal_error.equ(1.0, solution);
    nodal_error.add(-1.0, exact_solution);

    for (auto i : nodal_error)
      {
        i = std::abs(i);
      }
    nodal_error.update_ghost_values();


    Vector<float> diff_per_cell(triangulation.n_active_cells());

    QGauss<dim> quadrature(fe_degree + 2);

    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      Solution<dim>(),
                                      diff_per_cell,
                                      quadrature,
                                      VectorTools::Linfty_norm);
    const double Linfty_error =
      VectorTools::compute_global_error(triangulation,
                                        diff_per_cell,
                                        VectorTools::Linfty_norm);

    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      Solution<dim>(),
                                      diff_per_cell,
                                      quadrature,
                                      VectorTools::L2_norm);
    const double L2_error =
      VectorTools::compute_global_error(triangulation,
                                        diff_per_cell,
                                        VectorTools::L2_norm);

    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      Solution<dim>(),
                                      diff_per_cell,
                                      quadrature,
                                      VectorTools::H1_seminorm);
    const double H1s_error =
      VectorTools::compute_global_error(triangulation,
                                        diff_per_cell,
                                        VectorTools::H1_seminorm);

    const double H1_error =
      std::sqrt(L2_error * L2_error + H1s_error * H1s_error);


    pcout << "Linfty error:  " << Linfty_error << std::endl;
    pcout << "L2 error:      " << L2_error << std::endl;
    pcout << "H1s error:     " << H1s_error << std::endl;
    pcout << "H1 error:      " << H1_error << std::endl;


    convergence_table.add_value("cells", triangulation.n_global_active_cells());
    convergence_table.add_value("dofs", dof_handler.n_dofs());
    convergence_table.add_value("Linfty", Linfty_error);
    convergence_table.add_value("L2", L2_error);
    convergence_table.add_value("H1", H1_error);
    convergence_table.add_value("time", solver_time);
    convergence_table.add_value("iter", solver_iterations);
  }



  template <int dim, int fe_degree>
  void PoissonProblem<dim, fe_degree>::refine_grid()
  {
    TimerOutput::Scope timer_section(computing_timer, "refine_grid");

    if (prm.adaptive_refinement.criterion == "global")
      {
        triangulation.refine_global();
      }
    else if (prm.adaptive_refinement.criterion == "kelly")
      {
        if (prm.adaptive_refinement.strategy == "number")
          {
            parallel::distributed::GridRefinement::
              refine_and_coarsen_fixed_number(
                triangulation,
                estimate_per_cell,
                prm.adaptive_refinement.refine_fraction,
                prm.adaptive_refinement.coarsen_fraction);
          }
        else if (prm.adaptive_refinement.strategy == "energy")
          {
            parallel::distributed::GridRefinement::
              refine_and_coarsen_fixed_fraction(
                triangulation,
                estimate_per_cell,
                prm.adaptive_refinement.refine_fraction,
                prm.adaptive_refinement.coarsen_fraction);
          }
        else
          {
            Assert(false, ExcNotImplemented());
          }

        triangulation.execute_coarsening_and_refinement();
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }
  }



  template <int dim, int fe_degree>
  void PoissonProblem<dim, fe_degree>::graphical_output(
    const unsigned int cycle) const
  {
    Timer time;

    if (triangulation.n_global_active_cells() > 1e5)
      {
        pcout << "No output written" << std::endl;
        return;
      }

    solution.update_ghost_values();
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.add_data_vector(nodal_error, "nodal_error");
    data_out.add_data_vector(estimate_per_cell, "estimate_per_cell");
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      {
        subdomain(i) = triangulation.locally_owned_subdomain();
      }
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches(mapping,
                           fe_degree,
                           DataOut<dim>::curved_inner_cells);
    DataOutBase::VtkFlags flags;
    flags.compression_level        = DataOutBase::VtkFlags::best_speed;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    std::string filename = "solution-p" + std::to_string(fe_degree) + "-" +
                           prm.adaptive_refinement.criterion + "-" +
                           prm.adaptive_refinement.strategy + "-" +
                           std::to_string(dim) + "d";

    data_out.write_vtu_with_pvtu_record(
      "./", filename, cycle, MPI_COMM_WORLD, 3);
    solution.zero_out_ghost_values();

    time.stop();

    time_details << "Time write output           (CPU/Wall) " << time.cpu_time()
                 << " / " << time.wall_time() << " s" << std::endl;
  }



  template <int dim, int fe_degree>
  void PoissonProblem<dim, fe_degree>::build_convergence_table()
  {
    TimerOutput::Scope t(computing_timer, "build convergence table");

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        convergence_table.set_precision("Linfty", 2);
        convergence_table.set_precision("L2", 2);
        convergence_table.set_precision("H1", 2);
        convergence_table.set_precision("time", 2);

        convergence_table.set_scientific("Linfty", true);
        convergence_table.set_scientific("L2", true);
        convergence_table.set_scientific("H1", true);

        if (prm.adaptive_refinement.criterion == "global")
          {
            convergence_table.evaluate_convergence_rates(
              "Linfty", ConvergenceTable::reduction_rate_log2);
            convergence_table.evaluate_convergence_rates(
              "L2", ConvergenceTable::reduction_rate_log2);
            convergence_table.evaluate_convergence_rates(
              "H1", ConvergenceTable::reduction_rate_log2);
          }
        else
          {
            convergence_table.evaluate_convergence_rates(
              "Linfty", "dofs", ConvergenceTable::reduction_rate_log2, dim);
            convergence_table.evaluate_convergence_rates(
              "L2", "dofs", ConvergenceTable::reduction_rate_log2, dim);
            convergence_table.evaluate_convergence_rates(
              "H1", "dofs", ConvergenceTable::reduction_rate_log2, dim);
          }


        convergence_table.write_text(std::cout);

        std::string filename =
          "convergence-table-p" + std::to_string(fe_degree) + "-" +
          prm.adaptive_refinement.criterion + "-" +
          prm.adaptive_refinement.strategy + "-" + std::to_string(dim) + "d";

        std::ofstream out(filename + ".dat");
        convergence_table.write_text(out);
      }
  }



  template <int dim, int fe_degree>
  void PoissonProblem<dim, fe_degree>::run()
  {
    // General output before the program starts
    {
      const unsigned int n_ranks =
        Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
      const unsigned int n_vect_doubles = VectorizedArray<double>::size();
      const unsigned int n_vect_floats  = VectorizedArray<float>::size();
      const unsigned int n_vect_bits    = 8 * sizeof(double) * n_vect_doubles;

      std::string DAT_header = "START DATE: " + Utilities::System::get_date() +
                               ", TIME: " + Utilities::System::get_time();
      std::string MPI_header = "Running with " + std::to_string(n_ranks) +
                               " MPI process" + (n_ranks > 1 ? "es" : "");
      std::string VEC_header =
        "Vectorization over " + std::to_string(n_vect_doubles) + " doubles, " +
        std::to_string(n_vect_floats) +
        " floats = " + std::to_string(n_vect_bits) + " bits (" +
        Utilities::System::get_current_vectorization_level() +
        "), VECTORIZATION_LEVEL=" +
        std::to_string(DEAL_II_COMPILER_VECTORIZATION_LEVEL);

      const unsigned int header_size = 80;

      pcout << std::string(header_size, '=') << std::endl;
      pcout << DAT_header << std::endl;
      pcout << std::string(header_size, '-') << std::endl;

      pcout << MPI_header << std::endl;
      pcout << VEC_header << std::endl;

      pcout << std::string(header_size, '=') << std::endl;
    }

    if (prm.adaptive_refinement.criterion == "global")
      {
        pcout << "Global refinement" << std::endl;
      }
    else if (prm.adaptive_refinement.criterion == "kelly")
      {
        pcout << "A posteriori refinement (Kelly + "
              << prm.adaptive_refinement.strategy << " marking)" << std::endl;
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }

    pcout << std::string(80, '=') << std::endl;

    for (unsigned int cycle = 1;
         cycle <=
         (prm.adaptive_refinement.criterion == "global" ? 5 : prm.n_cycles);
         ++cycle)
      {
        pcout << "Cycle " << cycle << std::endl;
        pcout << std::string(80, '=') << std::endl;

        if (cycle == 1)
          {
            GridGenerator::hyper_L(triangulation);
            GridTools::rotate(-numbers::PI / 2, triangulation);
            triangulation.refine_global(prm.initial_refinement);
          }
        else
          {
            refine_grid();
          }

        setup_system();

        assemble_rhs();

        solve();

        compute_estimates();

        compute_errors();

        graphical_output(cycle);

        pcout << std::endl;
      }

    build_convergence_table();

    // General output before the program ends
    {
      std::string DAT_header = "END DATE: " + Utilities::System::get_date() +
                               ", TIME: " + Utilities::System::get_time();

      pcout << std::string(80, '=') << std::endl;
      pcout << DAT_header << std::endl;
      pcout << std::string(80, '=') << std::endl;
    }
  }
} // namespace Step75b



int main(int argc, char **argv)
{
  try
    {
      using namespace Step75b;

      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      Parameters prm;

      {
        prm.adaptive_refinement.criterion = "global";

        PoissonProblem<2, 1> poisson_problem(prm);
        poisson_problem.run();
      }

      {
        prm.adaptive_refinement.criterion = "kelly";
        prm.adaptive_refinement.strategy  = "number";

        PoissonProblem<2, 1> poisson_problem(prm);
        poisson_problem.run();
      }

      {
        prm.adaptive_refinement.criterion = "kelly";
        prm.adaptive_refinement.strategy  = "energy";

        PoissonProblem<2, 1> poisson_problem(prm);
        poisson_problem.run();
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
